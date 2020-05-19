#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import config
from utils.seq2seq import data

from utils.seq2seq.batcher import *

from utils.seq2seq.train_util import *
from utils.seq2seq.rl_util import *
from utils.seq2seq.initialize import loadCheckpoint, save_model
from utils.seq2seq.write_result import *
from datetime import datetime as dt
from tqdm import tqdm
from translate.seq2seq_beam import *
from tensorboardX import SummaryWriter
import argparse
from utils.seq2seq.rl_util import *
from torch.distributions import Categorical

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = 0

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

parser = argparse.ArgumentParser()
parser.add_argument('--key_attention', type=bool, default=False, help = 'True/False')
parser.add_argument('--intra_encoder', type=bool, default=False, help = 'True/False')
parser.add_argument('--intra_decoder', type=bool, default=False, help = 'True/False')
parser.add_argument('--copy', type=bool, default=True, help = 'True/False') # for transformer

parser.add_argument('--model_type', type=str, default='seq2seq', choices=['seq2seq', 'transformer'])
parser.add_argument('--train_rl', type=bool, default=False, help = 'True/False')
parser.add_argument('--keywords', type=str, default='POS_keys', 
                    help = 'POS_keys / DEP_keys / Noun_adj_keys / TextRank_keys')

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--rand_unif_init_mag', type=float, default=0.02)
parser.add_argument('--trunc_norm_init_std', type=float, default=0.001)
parser.add_argument('--mle_weight', type=float, default=1.0)
parser.add_argument('--gound_truth_prob', type=float, default=0.1)

parser.add_argument('--max_enc_steps', type=int, default=1000)
parser.add_argument('--max_dec_steps', type=int, default=50)
parser.add_argument('--min_dec_steps', type=int, default=8)
parser.add_argument('--max_epochs', type=int, default=12)
parser.add_argument('--vocab_size', type=int, default=50000)
parser.add_argument('--beam_size', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--gradient_accum', type=int, default=1)

parser.add_argument('--load_ckpt', type=str, default=None, help='0002000')
parser.add_argument('--word_emb_type', type=str, default='word2Vec', help='word2Vec/glove/FastText')
parser.add_argument('--pre_train_emb', type=bool, default=True, help = 'True/False') # 若pre_train_emb為false, 則emb type為NoPretrain


opt = parser.parse_args(args=[])
config = re_config(opt)
loggerName, writerPath = getName(config)    
logger = getLogger(loggerName)
writer = SummaryWriter(writerPath)


# In[2]:


train_loader, validate_loader, vocab = getDataLoader(logger, config)
train_batches = len(iter(train_loader))
test_batches = len(iter(validate_loader))
save_steps = int(train_batches/1000)*1000


# In[3]:


from seq2seq import Model
import torch.nn as nn
import torch as T
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist

from parallel import DataParallelModel, DataParallelCriterion
# https://gist.github.com/thomwolf/7e2407fbd5945f07821adae3d9fd1312


load_step = None
model = Model(pre_train_emb=config.pre_train_emb, 
              word_emb_type = config.word_emb_type, 
              vocab = vocab)

# model = model.cuda()
optimizer = T.optim.Adam(model.parameters(), lr=config.lr)   
# optimizer = T.optim.Adagrad(model.parameters(),lr=config.lr, initial_accumulator_value=0.1)

load_model_path = config.save_model_path + '/%s/%s.tar' % (loggerName, config.load_ckpt)
if os.path.exists(load_model_path):
    model, optimizer, load_step = loadCheckpoint(logger, load_model_path, model, optimizer)

    
model.to('cuda:%s' % use_gpu) 


# In[4]:


class NLLLoss(nn.Module):
        """
        With label smoothing,
        KL-divergence between q_{smoothed ground truth prob.}(w)
        and p_{prob. computed by model}(w) is minimized.
        """
        def __init__(self, ignore_index):
            super(NLLLoss, self).__init__()
#             step_loss = F.nll_loss(log_probs, target, reduction="none", ignore_index=PAD)
            self.NLL = nn.NLLLoss(ignore_index=ignore_index, reduction='sum')

        def forward(self, out, tar):  
            # target dimension[0] / 2
            # tar = target.contiguous().view(-1) 
            # out = output.contiguous().view(target.size(0),-1)

            target = tar.contiguous().view(-1)
            output = out[:tar.size(0)]
            normalize = output.size(0) * output.size(1)
            output = output.contiguous().view(target.size(0),-1)
            loss = self.NLL(output, target) / normalize
            
            return loss


criterion = NLLLoss(ignore_index=PAD)

parallel_model = DataParallelModel(model) # Encapsulate the model
parallel_loss = DataParallelCriterion(criterion)


# In[5]:


def to_sents(enc_out, inds, vocab, art_oovs):
    decoded_strs = []
    for i in range(len(enc_out)):
        id_list = inds[i].tolist() # 取出每個sample sentence 的word id list
        S = output2words(id_list, vocab, art_oovs[i]) #Generate sentence corresponding to sampled words
        try:
            end_idx = S.index(data.STOP_DECODING)
            S = S[:end_idx]
        except ValueError:
            S = S
        if len(S) < 2:          #If length of sentence is less than 2 words, replace it with "xxx"; Avoids setences like "." which throws error while calculating ROUGE
            S = ["xxx"]
        S = " ".join(S)
        decoded_strs.append(S)
    return decoded_strs

def merge_res(res):
    ((inds1, log_probs1, enc_out1),(inds2, log_probs2, enc_out2)) = res
    inds = T.cat([inds1, inds2], dim = 0).cpu()
    log_probs = T.cat([log_probs1, log_probs2], dim = 0)
    enc_out = T.cat([enc_out1, enc_out2], dim = 0).cpu()
    return inds, log_probs, enc_out

def train_one_rl(package):
    config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e,                                 max_dec_len, dec_batch, target_batch = package
    
    # multinomial sampling
    parallel_res1 = parallel_model(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e,                                 max_dec_len, dec_batch, target_batch, train_rl = True, greedy=False)
    sample_inds, RL_log_probs, sample_enc_out = merge_res(parallel_res1)
    
    # greedy sampling
    with T.autograd.no_grad(): 
        parallel_res2 = parallel_model(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e,                                     max_dec_len, dec_batch, target_batch, train_rl = True, greedy=True)
        greedy_inds, _, gred_enc_out = merge_res(parallel_res2)
        
        
    art_oovs = inputs.art_oovs
    sample_sents = to_sents(sample_enc_out, sample_inds, vocab, art_oovs)
    greedy_sents = to_sents(gred_enc_out, greedy_inds, vocab, art_oovs)
    
    sample_reward = reward_function(sample_sents, inputs.original_abstract) # r(w^s):通过根据概率来随机sample词生成句子的reward值
    baseline_reward = reward_function(greedy_sents, inputs.original_abstract) # r(w^):测试阶段使用greedy decoding取概率最大的词来生成句子的reward值

    batch_reward = T.mean(sample_reward).item()
    #Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
    rl_loss = -(sample_reward - baseline_reward) * RL_log_probs  # SCST梯度計算公式     
    rl_loss = T.mean(rl_loss)  
    return rl_loss, batch_reward

def train_one(package):
    config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e,                                 max_dec_len, dec_batch, target_batch = package
    
    max_enc_len = max(T.max(enc_lens,dim=0)).tolist()[0]    

    pred_probs = parallel_model(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e,                                 max_dec_len, dec_batch, target_batch)
    target = target_batch
    loss = parallel_loss(config.mle_weight, pred_probs, target)
    return loss

def get_package(inputs):
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage,         ct_e, enc_key_batch, enc_key_mask, enc_key_lens=             get_input_from_batch(0, inputs, config, batch_first = True)

    dec_batch, dec_padding_mask, dec_lens, max_dec_len, target_batch =         get_output_from_batch(0, inputs, config, batch_first = True) # Get input and target batchs for training decoder            

    max_enc_len = max(T.max(enc_lens,dim=0)).tolist()[0]    
    # ----------------------------------------------------
    package = (config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e,                                 max_dec_len, dec_batch, target_batch)
    return package


# for inputs in train_loader:  
#     # MLE test
#     # ----------------------------------------------------
#     # pred_probs = parallel_model(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e, \
#     #                             max_dec_len, dec_batch, target_batch)
#     # # pass
#     # target = target_batch
#     # loss = parallel_loss(config.mle_weight, pred_probs, target)
#     loss = train_one(package)
# #     loss.backward() # Backward pass 
# #     optimizer.step() # Optimizer step
#     print('loss : ',loss)
#     # pass
#     # ----------------------------------------------------   
#     if config.train_rl:
#         rl_loss, batch_reward = train_one_rl(package)
#         print('rl_loss : ',rl_loss, 'batch_reward : ',batch_reward)
#     else:
#         rl_loss = T.FloatTensor([0]).cuda()        
    
#     (config.mle_weight * loss + config.rl_weight * rl_loss).backward() # Backward pass   
#     optimizer.step() # Optimizer step
#     optimizer.zero_grad() # 清空过往梯度 


# In[6]:


# @torch.no_grad()
@torch.autograd.no_grad()
def validate(validate_loader, config, model):
#     model.eval()
    losses = []
#     batch = next(iter(validate_loader))
    val_num = len(iter(validate_loader))
    for idx, batch in enumerate(validate_loader):
#         package = get_package(batch)
        loss = train_one(get_package(batch))
#         loss = train_one(model, config, batch)
        losses.append(loss.item())
        if idx>= val_num/10: break
#     model.train()
    avg_loss = sum(losses) / len(losses)
    return avg_loss

@torch.autograd.no_grad()
def calc_running_avg_loss(loss, running_avg_loss, decay=0.99):
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    return running_avg_loss


# In[7]:


import time
loss_st, loss_cost = 0,0
decode_st, decode_cost = 0,0

write_train_para(writer, config)
logger.info('------Training START--------')
running_avg_loss, running_avg_rl_loss = 0, 0
sum_total_reward = 0
step = 0
print_step = 1000

# try:
for epoch in range(config.max_epochs):
    for batch in train_loader:
        step += 1
        loss_st = time.time()
        package = get_package(batch)
        mle_loss = train_one(package)
#             if config.train_rl:
#                 rl_loss, batch_reward = train_one_rl(package)             
        
#                 if step%1000 == 0 :
#                     writer.add_scalars('scalar/RL_Loss',  
#                        {'rl_loss': rl_loss
#                        }, step)
#                     writer.add_scalars('scalar/Reward',  
#                        {'batch_reward': batch_reward
#                        }, step)
# #                     logger.info('epoch %d: %d, RL_Loss = %f, batch_reward = %f'
# #                                     % (epoch, step, rl_loss, batch_reward))
#                 sum_total_reward += batch_reward
#             else:
#                 rl_loss = T.FloatTensor([0]).cuda()
#             (config.mle_weight * mle_loss + config.rl_weight * rl_loss).backward()  # 反向传播，计算当前梯度

#             '''梯度累加就是，每次获取1个batch的数据，计算1次梯度，梯度不清空'''
#             if step % (config.gradient_accum) == 0: # gradient accumulation
#     #             clip_grad_norm_(model.parameters(), 5.0)                      
#                 optimizer.step() # 根据累计的梯度更新网络参数
#                 optimizer.zero_grad() # 清空过往梯度 
#             if step%print_step == 0 :
#                 with T.autograd.no_grad():
#                     train_batch_loss = mle_loss.item()
#                     train_batch_rl_loss = rl_loss.item()
#                     val_avg_loss = validate(validate_loader, config, model) # call batch by validate_loader
#                     running_avg_loss = calc_running_avg_loss(train_batch_loss, running_avg_loss)
#                     running_avg_rl_loss = calc_running_avg_loss(train_batch_rl_loss, running_avg_rl_loss)
#                     running_avg_reward = sum_total_reward / step
#                     if step % save_steps == 0:
#                         logger.info('epoch %d: %d, training batch loss = %f, running_avg_loss loss = %f, validation loss = %f'
#                                     % (epoch, step, train_batch_loss, running_avg_loss, val_avg_loss))
#                     writer.add_scalars('scalar/Loss',  
#                        {'train_batch_loss': train_batch_loss
#                        }, step)
#                     writer.add_scalars('scalar_avg/loss',  
#                        {'train_avg_loss': running_avg_loss,
#                         'test_avg_loss': val_avg_loss
#                        }, step)
#                     if running_avg_reward > 0:
# #                         logger.info('epoch %d: %d, running_avg_reward = %f'
# #                                 % (epoch, step, running_avg_reward))
#                         writer.add_scalars('scalar_avg/Reward',  
#                            {'running_avg_reward': running_avg_reward
#                            }, step)
#                     if running_avg_rl_loss != 0:
# #                         logger.info('epoch %d: %d, running_avg_rl_loss = %f'
# #                                 % (epoch, step, running_avg_rl_loss))
#                         writer.add_scalars('scalar_avg/RL_Loss',  
#                            {'running_avg_rl_loss': running_avg_rl_loss
#                            }, step)
#                     loss_cost = time.time() - loss_st
#                     if step % save_steps == 0: logger.info('epoch %d|step %d| compute loss cost = %f ms'
#                                 % (epoch, step, loss_cost))

#             if step % save_steps == 0:
#                 save_model(config, logger, model, optimizer, step, vocab, running_avg_loss, \
#                            r_loss=0, title = loggerName)
# #             if step%1000 == 0 and step > 0:
# #                 decode_st = time.time()
# #                 train_rouge_l_f = decode(writer, logger, step, config, model, batch, mode = 'train') # call batch by validate_loader
# #                 test_rouge_l_f = decode(writer, logger, step, config, model, validate_loader, mode = 'test') # call batch by validate_loader
# #                 decode_cost = time.time() - decode_st
# #                 if step%save_steps == 0: logger.info('epoch %d|step %d| decode cost = %f ms'% (epoch, step, decode_cost))

# #                 writer.add_scalars('scalar/Rouge-L',  
# #                    {'train_rouge_l_f': train_rouge_l_f,
# #                     'test_rouge_l_f': test_rouge_l_f
# #                    }, step)
# #                 logger.info('epoch %d: %d, train_rouge_l_f = %f, test_rouge_l_f = %f'
# #                                 % (epoch, step, train_rouge_l_f, test_rouge_l_f))
# #         break
#         logger.info('-------------------------------------------------------------')
# #         train_avg_acc = avg_acc(writer, logger, epoch, config, model, train_loader, mode = 'train')
# #         test_avg_acc = avg_acc(writer, logger, epoch, config, model, validate_loader, mode = 'test')                   
# #         logger.info('epoch %d|step %d| train_avg_acc = %f, test_avg_acc = %f' % (epoch, step, train_avg_acc, test_avg_acc))
#         if running_avg_reward > 0:
#             logger.info('epoch %d|step %d| running_avg_reward = %f'% (epoch, step, running_avg_reward))
#         if running_avg_rl_loss != 0:
#             logger.info('epoch %d|step %d| running_avg_rl_loss = %f'% (epoch, step, running_avg_rl_loss))
#         logger.info('-------------------------------------------------------------')

# except Excepation as e:
#         print(e)
# else:
#     logger.info(u'------Training SUCCESS--------')  
# finally:
#     logger.info(u'------Training END--------')    
#     del parallel_model, parallel_loss
# #     train_avg_acc, train_outFrame = decode_write_all(writer, logger, epoch, config, model, train_loader, mode = 'train')
# #     test_avg_acc, test_outFrame = decode_write_all(writer, logger, epoch, config, model, validate_loader, mode = 'test')
# #     logger.info('epoch %d: train_avg_acc = %f, test_avg_acc = %f' % (epoch, train_avg_acc, test_avg_acc))
#     removeLogger(logger)


# In[12]:


import pandas as pd
import time
from utils.seq2seq.write_result import total_evaulate, total_output

@torch.autograd.no_grad()
def decode_write_all(writer, logger, epoch, config, model, dataloader, mode):
    # 動態取batch
    num = len(dataloader)
    avg_rouge_1, avg_rouge_2, avg_rouge_l  = [], [], []
    avg_self_bleu1, avg_self_bleu2, avg_self_bleu3, avg_self_bleu4 = [], [], [], []
    avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4 = [], [], [], []
    avg_meteor = []
    outFrame = None
    avg_time = 0
        
    for idx, inputs in enumerate(dataloader):
        start = time.time() 
#         'Encoder data'
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage, \
            ct_e, enc_key_batch, enc_key_mask, enc_key_lens = get_input_from_batch(use_gpu, inputs, config, batch_first = True)
        max_enc_len = max(T.max(enc_lens,dim=0)).tolist()[0] 

        enc_batch = model.embeds(enc_batch)  # Get embeddings for encoder input    
        enc_key_batch = model.embeds(enc_key_batch)  # Get key embeddings for encoder input

        enc_out, enc_hidden = model.encoder(enc_batch, enc_lens, max_enc_len)
        
#         'Feed encoder data to predict'
        pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, 
                                enc_batch_extend_vocab, enc_key_batch, enc_key_mask, model, 
                                START, END, UNKNOWN_TOKEN)

        article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index = prepare_result(vocab, inputs, pred_ids)
        cost = (time.time() - start)
        avg_time += cost        

        
        rouge_1, rouge_2, rouge_l, self_Bleu_1, self_Bleu_2, self_Bleu_3, self_Bleu_4,             Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, batch_frame = total_evaulate(article_sents, keywords_list, decoded_sents, ref_sents)
        
        if idx %1000 ==0 and idx >0 : print(idx)
        if idx == 0: outFrame = batch_frame
        else: outFrame = pd.concat([outFrame, batch_frame], axis=0, ignore_index=True) 
        # ----------------------------------------------------
        avg_rouge_1.extend(rouge_1)
        avg_rouge_2.extend(rouge_2)
        avg_rouge_l.extend(rouge_l)   
        
        avg_self_bleu1.extend(self_Bleu_1)
        avg_self_bleu2.extend(self_Bleu_2)
        avg_self_bleu3.extend(self_Bleu_3)
        avg_self_bleu4.extend(self_Bleu_4)
        
        avg_bleu1.extend(Bleu_1)
        avg_bleu2.extend(Bleu_2)
        avg_bleu3.extend(Bleu_3)
        avg_bleu4.extend(Bleu_4)
        avg_meteor.extend(Meteor)
        # ----------------------------------------------------    
    avg_time = avg_time / (num * config.batch_size) 
    
    avg_rouge_l, outFrame = total_output(mode, writerPath, outFrame, avg_time, avg_rouge_1, avg_rouge_2, avg_rouge_l,         avg_self_bleu1, avg_self_bleu2, avg_self_bleu3, avg_self_bleu4,         avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4, avg_meteor
    )
    
    return avg_rouge_l, outFrame


# In[13]:


config.batch_size = 8
epoch = config.max_epochs
# train_loader, validate_loader, vocab = getDataLoader(logger, config)
# train_batches = len(iter(train_loader))
# test_batches = len(iter(validate_loader))
# save_steps = int(train_batches/1000)*1000

train_avg_acc, train_outFrame = decode_write_all(writer, logger, epoch, config, model, train_loader, mode = 'train')
test_avg_acc, test_outFrame = decode_write_all(writer, logger, epoch, config, model, validate_loader, mode = 'test')
logger.info('epoch %d: train_avg_acc = %f, test_avg_acc = %f' % (epoch, train_avg_acc, test_avg_acc)) 


# In[ ]:




