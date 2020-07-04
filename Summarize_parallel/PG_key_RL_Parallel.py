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
eval_gpu = 0

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

parser = argparse.ArgumentParser()
parser.add_argument('--key_attention', type=bool, default=True, help = 'True/False')
parser.add_argument('--intra_encoder', type=bool, default=False, help = 'True/False')
parser.add_argument('--intra_decoder', type=bool, default=False, help = 'True/False')
parser.add_argument('--copy', type=bool, default=True, help = 'True/False') # for transformer


parser.add_argument('--model_type', type=str, default='seq2seq', choices=['seq2seq', 'transformer'])
parser.add_argument('--train_rl', type=bool, default=True, help = 'True/False')
parser.add_argument('--keywords', type=str, default='Noun_adj_keys', 
                    help = 'POS_keys / DEP_keys / Noun_adj_keys / TextRank_keys')

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--rand_unif_init_mag', type=float, default=0.02)
parser.add_argument('--trunc_norm_init_std', type=float, default=0.001)
parser.add_argument('--mle_weight', type=float, default=1.0)
parser.add_argument('--gound_truth_prob', type=float, default=0.5)

parser.add_argument('--max_enc_steps', type=int, default=500)
parser.add_argument('--max_dec_steps', type=int, default=20)
parser.add_argument('--min_dec_steps', type=int, default=6)
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--vocab_size', type=int, default=50000)
parser.add_argument('--beam_size', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--gradient_accum', type=int, default=1)

parser.add_argument('--load_ckpt', type=str, default='', help='0002000')
parser.add_argument('--word_emb_type', type=str, default='word2Vec', help='word2Vec/glove/FastText')
parser.add_argument('--pre_train_emb', type=bool, default=True, help = 'True/False') # 若pre_train_emb為false, 則emb type為NoPretrain


opt = parser.parse_args(args=[])
config = re_config(opt)
loggerName, writerPath = getName(config)    
logger = getLogger(loggerName)
writer = SummaryWriter(writerPath)

eval_model = False


# In[2]:


train_loader, validate_loader, vocab = getDataLoader(logger, config)
train_batches = len(iter(train_loader))
test_batches = len(iter(validate_loader))
save_steps = int(train_batches/250)*250


# In[3]:


from create_model.pg import Model
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
    # 若偵測到model切換成eval
    eval_model = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)
    
else:    
    model.to('cuda:%s' % 0) #BCW


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

if not eval_model:
    criterion = NLLLoss(ignore_index=PAD)
    parallel_model = DataParallelModel(model) # Encapsulate the model
    parallel_loss = DataParallelCriterion(criterion)


# In[ ]:


# ---------------------------

# ---------------------------

def train_one_rl(package, inputs):
    config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e,                                 max_dec_len, dec_batch, target_batch = package
    
    rl_loss, batch_reward = parallel_model(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e,                                 max_dec_len, dec_batch, target_batch, train_rl = True, art_oovs = inputs.art_oovs, original_abstract = inputs.original_abstract, vocab = vocab)
    rl_loss = nn.parallel.gather(rl_loss, 0).mean() 
    return rl_loss, batch_reward

def train_one(package):
    model.train()
    config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e,                                 max_dec_len, dec_batch, target_batch = package   

    pred_probs = parallel_model(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e,                                 max_dec_len, dec_batch, target_batch)
    target = target_batch
    loss = parallel_loss(config.mle_weight, pred_probs, target)
    
    return loss, pred_probs

# ---------------------------

def get_package(inputs):
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage,         ct_e, enc_key_batch, enc_key_mask, enc_key_lens=             get_input_from_batch(inputs, config, batch_first = True)

    dec_batch, dec_padding_mask, dec_lens, max_dec_len, target_batch =         get_output_from_batch(inputs, config, batch_first = True) # Get input and target batchs for training decoder            

    max_enc_len = max(T.max(enc_lens,dim=0)).tolist()[0]    
    # ----------------------------------------------------
    package = (config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e,                                 max_dec_len, dec_batch, target_batch)
    
    inner_c = package[1] != max(package[4].tolist())[0]

    return inner_c, package


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


# In[ ]:


# @torch.no_grad()
@torch.autograd.no_grad()
def validate(validate_loader, config, model):
    model.eval()
    losses = []
#     batch = next(iter(validate_loader))
    val_num = len(iter(validate_loader))
    for idx, batch in enumerate(validate_loader):
        inner_c, package = get_package(batch)
        if inner_c: continue
        loss, _ = train_one(package)
#         loss = train_one(model, config, batch)
        losses.append(loss.item())
#         if idx>= val_num/40: break
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


# In[ ]:


# ---------------------------


# In[ ]:


# del parallel_model, parallel_loss

import pandas as pd
import time
from utils.seq2seq.write_result import total_evaulate, total_output

@torch.autograd.no_grad()
def decode(writer, dataloader, epoch):
    # 動態取batch
    num = len(dataloader)
    outFrame = None
    avg_time = 0
    total_scores = dict()   
    idx = 0 
    for _, inputs in enumerate(dataloader):
        start = time.time() 
        # 'Encoder data'
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage,             ct_e, enc_key_batch, enc_key_mask, enc_key_lens = get_input_from_batch(inputs, config, batch_first = True)
        max_enc_len = max(T.max(enc_lens,dim=0)).tolist()[0] 
        
        if (max_enc_len != max(enc_lens.tolist())[0]): continue

        enc_batch = parallel_model.module.embeds(enc_batch)  # Get embeddings for encoder input    
        enc_key_batch = parallel_model.module.embeds(enc_key_batch)  # Get key embeddings for encoder input

        enc_out, enc_hidden = parallel_model.module.encoder(enc_batch, enc_lens, max_enc_len)
        
        # 'Feed encoder data to predict'
        pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, 
                                enc_batch_extend_vocab, enc_key_batch, enc_key_mask, parallel_model.module, 
                                START, END, UNKNOWN_TOKEN)

        article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index = prepare_result(vocab, inputs, pred_ids)
        cost = (time.time() - start)
        avg_time += cost        

        multi_scores, batch_frame = total_evaulate(article_sents, keywords_list, decoded_sents, ref_sents)
        review_IDS = [review_ID for review_ID in inputs.review_IDS]
        batch_frame['review_ID'] = review_IDS        
        if idx %1000 ==0 and idx >0 : 
            print(idx); 
        if idx == 0: 
            outFrame = batch_frame; 
            total_scores = multi_scores
        else: 
            outFrame = pd.concat([outFrame, batch_frame], axis=0, ignore_index=True) 
            for key, scores in total_scores.items():
                scores.extend(multi_scores[key])
                total_scores[key] = scores
        idx += 1
        # ----------------------------------------------------    
    avg_time = avg_time / (num * config.batch_size)    

    scalar_acc = {}
    num = 0
    for key, scores in total_scores.items():
        num = len(scores)
        scalar_acc[key] = sum(scores)/len(scores)

    for scalar_name, accuracy in scalar_acc.items():
        if 'rouge_1' in scalar_name:
            writer.add_scalars('scalar/rouge_1',  
               {scalar_name: accuracy,
               }, epoch)
        elif 'rouge_2' in scalar_name:
            writer.add_scalars('scalar/rouge_2',  
               {scalar_name: accuracy,
               }, epoch)
        elif 'rouge_l' in scalar_name:
            writer.add_scalars('scalar/rouge_l',  
               {scalar_name: accuracy,
               }, epoch)
        elif 'bleu' in scalar_name:
            writer.add_scalars('scalar/bleu',  
               {scalar_name: accuracy,
               }, epoch)
        elif 'meteor' in scalar_name:
            writer.add_scalars('scalar/meteor',  
               {scalar_name: accuracy,
               }, epoch)
    
    # -----------------------------------------------------------
    total_output(epoch, 'test', writerPath, outFrame, avg_time, num , scalar_acc
    )
    # -----------------------------------------------------------
    outFrame = outFrame.sort_values(by=['rouge_l'], ascending=False)
    big_frame = outFrame.head()
    small_frame = outFrame.tail()    
    # -----------------------------------------------------------
    i = 0
    for view_item in big_frame.to_dict('records'):
        writer.add_text('BigTest/epoch_%s/##%s' % (epoch, i),
                        "### rouge_l : &nbsp;&nbsp;&nbsp;\
                        " + str(view_item['rouge_l']), epoch)
        writer.add_text('BigTest/epoch_%s/##%s' % (epoch, i),
                        "### decoded : &nbsp;&nbsp;&nbsp;\
                        " + view_item['decoded'], epoch)
        writer.add_text('BigTest/epoch_%s/##%s' % (epoch, i),
                        "### reference : &nbsp;&nbsp;&nbsp;\
                        " + view_item['reference'], epoch)
        writer.add_text('BigTest/epoch_%s/##%s' % (epoch, i),
                        "### keywords : &nbsp;&nbsp;&nbsp;\
                        " + view_item['keywords'], epoch)
        writer.add_text('BigTest/epoch_%s/##%s' % (epoch, i),
                        "### article : &nbsp;&nbsp;&nbsp;\
                        " + view_item['article'], epoch)

        i += 1
    # -----------------------------------------------------------
    i = 0
    for view_item in small_frame.to_dict('records'):
        writer.add_text('SmallTest/epoch_%s/##%s' % (epoch, i),
                        "### rouge_l : &nbsp;&nbsp;&nbsp;\
                        " + str(view_item['rouge_l']), epoch)
        writer.add_text('SmallTest/epoch_%s/##%s' % (epoch, i),
                        "### decoded : &nbsp;&nbsp;&nbsp;\
                        " + view_item['decoded'], epoch)
        writer.add_text('SmallTest/epoch_%s/##%s' % (epoch, i),
                        "### reference : &nbsp;&nbsp;&nbsp;\
                        " + view_item['reference'], epoch)
        writer.add_text('SmallTest/epoch_%s/##%s' % (epoch, i),
                        "### keywords : &nbsp;&nbsp;&nbsp;\
                        " + view_item['keywords'], epoch)
        writer.add_text('SmallTest/epoch_%s/##%s' % (epoch, i),
                        "### article : &nbsp;&nbsp;&nbsp;\
                        " + view_item['article'], epoch)
        i += 1
    return outFrame



# In[ ]:


import time
loss_st, loss_cost = 0,0
decode_st, decode_cost = 0,0
last_save_step = 0
from pytorchtools import EarlyStopping

print_step = 250
# save_steps = print_step
if not eval_model:

    write_train_para(writer, config)
    logger.info('------Training START--------')
    running_avg_loss, running_avg_rl_loss = 0, 0
    sum_total_reward = 0
    step = 0
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(config, logger, vocab, loggerName, patience=3, verbose=True)
    try:
        for epoch in range(1, config.max_epochs+1):
            for batch in train_loader:
                step += 1; 
                loss_st = time.time()
                inner_c, package = get_package(batch)
                if inner_c: continue
                parallel_model.module.train()
                mle_loss, pred_probs = train_one(package)
                if config.train_rl:
                    rl_loss, batch_reward = train_one_rl(package, batch)             

                    if step%print_step == 0 :
                        writer.add_scalars('scalar/RL_Loss',  
                           {'rl_loss': rl_loss
                           }, step)
                        writer.add_scalars('scalar/Reward',  
                           {'batch_reward': batch_reward
                           }, step)
    #                     logger.info('epoch %d: %d, RL_Loss = %f, batch_reward = %f'
    #                                     % (epoch, step, rl_loss, batch_reward))
                    sum_total_reward += batch_reward
                else:
                    rl_loss = T.FloatTensor([0]).cuda()

                (config.mle_weight * mle_loss + config.rl_weight * rl_loss).backward()  # 反向传播，计算当前梯度

                '''梯度累加就是，每次获取1个batch的数据，计算1次梯度，梯度不清空'''
                if step % (config.gradient_accum) == 0: # gradient accumulation
        #             clip_grad_norm_(model.parameters(), 5.0)                      
                    optimizer.step() # 根据累计的梯度更新网络参数
                    optimizer.zero_grad() # 清空过往梯度 
                if step%print_step == 0 :
                    with T.autograd.no_grad():
                        train_batch_loss = mle_loss.item()
                        train_batch_rl_loss = rl_loss.item()
#                         val_avg_loss = validate(validate_loader, config, model) # call batch by validate_loader
                        running_avg_loss = calc_running_avg_loss(train_batch_loss, running_avg_loss)
                        running_avg_rl_loss = calc_running_avg_loss(train_batch_rl_loss, running_avg_rl_loss)
                        running_avg_reward = sum_total_reward / step
#                         if step % save_steps == 0:
#                             logger.info('epoch %d: %d, training batch loss = %f, running_avg_loss loss = %f, validation loss = %f'
#                                         % (epoch, step, train_batch_loss, running_avg_loss, val_avg_loss))
                        writer.add_scalars('scalar/Loss',  
                           {'train_batch_loss': train_batch_loss
                           }, step)
                        writer.add_scalars('scalar_avg/loss',  
                           {'train_avg_loss': running_avg_loss
#                             'test_avg_loss': val_avg_loss
                           }, step)
                        if running_avg_reward > 0:
    #                         logger.info('epoch %d: %d, running_avg_reward = %f'
    #                                 % (epoch, step, running_avg_reward))
                            writer.add_scalars('scalar_avg/Reward',  
                               {'running_avg_reward': running_avg_reward
                               }, step)
                        if running_avg_rl_loss != 0:
    #                         logger.info('epoch %d: %d, running_avg_rl_loss = %f'
    #                                 % (epoch, step, running_avg_rl_loss))
                            writer.add_scalars('scalar_avg/RL_Loss',  
                               {'running_avg_rl_loss': running_avg_rl_loss
                               }, step)
                                                    
                
                if step % save_steps == 0:
                    parallel_model.module.eval()
                    logger.info('epoch : %s' % epoch)
                    val_avg_loss = validate(validate_loader, config, model) # call batch by validate_loader
                    logger.info('epoch %d: %d, training batch loss = %f, running_avg_loss loss = %f, validation loss = %f'
                                        % (epoch, step, train_batch_loss, running_avg_loss, val_avg_loss))
                    writer.add_scalars('scalar_avg/loss',  
                           {'train_avg_loss': running_avg_loss,
                            'test_avg_loss': val_avg_loss
                           }, step)
                    '''（讀取所儲存模型引數後，再進行並行化操作，否則無法利用之前的程式碼進行讀取）'''
                    save_model(config, logger, parallel_model, optimizer, step, vocab, val_avg_loss,                                r_loss=0, title = loggerName)
                    loss_cost = time.time() - loss_st
                    logger.info('epoch %d|step %d| compute loss cost = %f ms'
                                    % (epoch, step, loss_cost))
                    writer.add_scalars('scalar_avg/epoch_loss',  
                       {'train_avg_loss': running_avg_loss,
                        'test_avg_loss': val_avg_loss
                       }, epoch)
                    last_save_step = step
                    test_outFrame = decode(writer, validate_loader, epoch)                   

            logger.info('-------------------------------------------------------------')

            if running_avg_reward > 0:
                logger.info('epoch %d|step %d| running_avg_reward = %f'% (epoch, step, running_avg_reward))
            if running_avg_rl_loss != 0:
                logger.info('epoch %d|step %d| running_avg_rl_loss = %f'% (epoch, step, running_avg_rl_loss))
            logger.info('-------------------------------------------------------------')

            early_stopping(parallel_model, optimizer, step, val_avg_loss) # update patience
            if early_stopping.early_stop:
                logger.info("Early stopping epoch %s"%(epoch))
                break

    except Exception as e:
            print(e)
    else:
        logger.info(u'------Training SUCCESS--------')  
    finally:
        logger.info(u'------Training END--------')   
        logger.info("stopping epoch %s"%(epoch))        
        logger.info("last_save_step %s"%(last_save_step))  
        '''先將test_avg_acc調起來再decode train_'''
    #     train_avg_acc, train_outFrame = decode_write_all(writer, logger, epoch, config, model, train_loader, mode = 'train')
#         test_avg_acc, test_outFrame = decode_write_all(writer, logger, epoch, config, parallel_model.module, validate_loader, mode = 'test')
    #     logger.info('epoch %d:  test_avg_acc = %f' % (epoch,  test_avg_acc)) 
#         logger.info('epoch %d: test_avg_acc = %f' % (load_ep, test_avg_acc)) 
        removeLogger(logger)

# else: # EVAL
#     load_ep = float(config.load_ckpt) / float(save_steps)
#     config.batch_size = 32
#     train_loader, validate_loader, vocab = getDataLoader(logger, config)
#     train_batches = len(iter(train_loader))
#     test_batches = len(iter(validate_loader))
# #     save_steps = int(train_batches/250)*250
#     model.cuda(eval_gpu) 
#     model.eval()
#     '''先將test_avg_acc調起來再decode train_'''
# #     train_avg_acc, train_outFrame = decode_write_all(writer, logger, load_ep, config, model, train_loader, mode = 'train')
#     test_avg_acc, test_outFrame = decode_write_all(writer, logger, load_ep, config, model, validate_loader, mode = 'test')
# #     logger.info('epoch %d: train_avg_acc = %f, test_avg_acc = %f' % (load_ep, train_avg_acc, test_avg_acc)) 
#     logger.info('epoch %d: test_avg_acc = %f' % (load_ep, test_avg_acc)) 
#     removeLogger(logger)


# In[ ]:


test_outFrame.columns


# In[ ]:


test_outFrame[test_outFrame["rouge_1"]>=0.4][['rouge_1','article', 'reference', 'decoded', 'gen_type','overlap']]


# In[ ]:


# batch_16 epoch_6
# testing_avg_rouge_1: 0.3873426628114616 \n', 
# 'testing_avg_rouge_2: 0.25943944916828854 \n', 
# 'testing_avg_rouge_l: 0.3614074094052472 \n

