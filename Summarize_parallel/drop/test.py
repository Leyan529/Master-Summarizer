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
parser.add_argument('--beam_size', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=4)

parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--gradient_accum', type=int, default=1)

parser.add_argument('--load_ckpt', type=str, default=None, help='0002000')
parser.add_argument('--word_emb_type', type=str, default='word2Vec', help='word2Vec/glove/FastText')
parser.add_argument('--pre_train_emb', type=bool, default=False, help = 'True/False') # 若pre_train_emb為false, 則emb type為NoPretrain


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


# In[9]:


from seq2seq import Model
import torch.nn as nn
import torch as T
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist


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
model.to('cuda:0') 

# In[10]:


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

        def forward(self, output, target):  
            # target dimension[0] / 2
            # tar = target.contiguous().view(-1) 
            # out = output.contiguous().view(target.size(0),-1)

            target = target.contiguous().view(-1)
            # output = out[:tar.size(0)]
            normalize = output.size(0) * output.size(1)
            output = output.contiguous().view(target.size(0),-1)
            loss = self.NLL(output, target) / normalize
            
            return loss
        
criterion = NLLLoss(ignore_index=PAD)


# In[11]:


from parallel import DataParallelModel, DataParallelCriterion

# https://gist.github.com/thomwolf/7e2407fbd5945f07821adae3d9fd1312

parallel_model = DataParallelModel(model) # Encapsulate the model
parallel_loss = DataParallelCriterion(criterion)

# for inputs in train_loader:  

#     enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage, \
#         ct_e, enc_key_batch, enc_key_mask, enc_key_lens= \
#             get_input_from_batch(0, inputs, config, batch_first = True)

#     dec_batch, dec_padding_mask, dec_lens, max_dec_len, target_batch = \
#         get_output_from_batch(0, inputs, config, batch_first = True) # Get input and target batchs for training decoder            

#     max_enc_len = max(T.max(enc_lens,dim=0)).tolist()[0]    

#     # MLE test
#     # ----------------------------------------------------
#     # pred_probs = parallel_model(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e, \
#     #                             max_dec_len, dec_batch, target_batch)
#     # # pass
#     # target = target_batch
#     # loss = parallel_loss(config.mle_weight, pred_probs, target)
 
#     # loss.backward() # Backward pass 
#     # optimizer.step() # Optimizer step
#     # print(loss)
#     # pass
#     # ----------------------------------------------------
#     # inds, log_probs
#     ((inds1, log_probs1, enc_out1),(inds2, log_probs2, enc_out2)) = parallel_model(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e, \
#                                 max_dec_len, dec_batch, target_batch, train_rl = True)
#     inds = T.cat([inds1, inds2], dim = 0)
#     log_probs = T.cat([log_probs1, log_probs2], dim = 0)
    # enc_out = T.cat([enc_out1, enc_out2], dim = 0)
    # pass



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
    enc_out = T.cat([enc_out1, enc_out2], dim = 0).cpu()
    if (type(log_probs1) != list) and (type(log_probs2)!= list) :
        log_probs = T.cat([log_probs1, log_probs2], dim = 0)    
        return inds, log_probs, enc_out
    else:
        return inds, enc_out

def train_one_rl(package):
    config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e, \
                                max_dec_len, dec_batch, target_batch = package
    
    # multinomial sampling
    parallel_res1 = parallel_model(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e, \
                                max_dec_len, dec_batch, target_batch, train_rl = True, greedy=False)
    sample_inds, RL_log_probs, sample_enc_out = merge_res(parallel_res1)
    
    # greedy sampling
    with T.autograd.no_grad(): 
        parallel_res2 = parallel_model(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e, \
                                    max_dec_len, dec_batch, target_batch, train_rl = True, greedy=True)
        greedy_inds, gred_enc_out = merge_res(parallel_res2)
        
        
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
    config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e, \
                                max_dec_len, dec_batch, target_batch = package
    
    max_enc_len = max(T.max(enc_lens,dim=0)).tolist()[0]    

    pred_probs = parallel_model(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e, \
                                max_dec_len, dec_batch, target_batch)
    target = target_batch
    loss = parallel_loss(config.mle_weight, pred_probs, target)
    return loss

def get_package(inputs):
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage, \
        ct_e, enc_key_batch, enc_key_mask, enc_key_lens= \
            get_input_from_batch(0, inputs, config, batch_first = True)

    dec_batch, dec_padding_mask, dec_lens, max_dec_len, target_batch = \
        get_output_from_batch(0, inputs, config, batch_first = True) # Get input and target batchs for training decoder            

    max_enc_len = max(T.max(enc_lens,dim=0)).tolist()[0]    
    # ----------------------------------------------------
    package = (config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e, \
                                max_dec_len, dec_batch, target_batch)
    return package

for inputs in train_loader: 
    package = get_package(inputs)  
    # ----------------------------------------------------
    
    # MLE test
    # ----------------------------------------------------
    # pred_probs = parallel_model(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e, \
    #                             max_dec_len, dec_batch, target_batch)
    # # pass
    # target = target_batch
    # loss = parallel_loss(config.mle_weight, pred_probs, target)
    loss = train_one(package)
    # loss.backward() # Backward pass 
    # optimizer.step() # Optimizer step
    print('loss : ',loss)
    # ----------------------------------------------------   
    rl_loss, batch_reward = train_one_rl(package)
    print('rl_loss : ',rl_loss, 'batch_reward : ',batch_reward)
    (config.mle_weight * loss + config.rl_weight * rl_loss).backward() # Backward pass   
    optimizer.step() # Optimizer step
    optimizer.zero_grad() # 清空过往梯度 
    # ------------------beam search decode -----------------------------   
    # select_batch = next(iter(batch))
    # batch = select_batch
    # if type(batch) == torch.utils.data.dataloader.DataLoader:
    #     batch = next(iter(batch))
    # package =  get_package(next(iter(validate_loader)))

    # config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, extra_zeros, enc_batch_extend_vocab, ct_e, \
    #                             max_dec_len, dec_batch, target_batch = package
    # 'Encoder data'
    # enc_batch = model.embeds(enc_batch)  # Get embeddings for encoder input    
    # enc_key_batch = model.embeds(enc_key_batch)  # Get key embeddings for encoder input

    # enc_out, enc_hidden = model.encoder(enc_batch, enc_lens, max_enc_len)

    # 'Feed encoder data to predict'
    # pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, 
    #                        enc_batch_extend_vocab, enc_key_batch, enc_key_mask, model, 
    #                        START, END, UNKNOWN_TOKEN)
    # pass





# In[ ]:




