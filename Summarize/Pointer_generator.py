
# coding: utf-8

# In[1]:

from utils import config, data
from utils.batcher import *
from utils.train_util import *
from utils.rl_util import *
from utils.initialize import loadCheckpoint, save_model
from utils.write_result import *
from datetime import datetime as dt
from tqdm import tqdm
from beam.beam_search import *
from tensorboardX import SummaryWriter
import argparse
from utils.rl_util import *
from torch.distributions import Categorical

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

parser = argparse.ArgumentParser()
parser.add_argument('--key_attention', type=bool, default=False, help = 'True/False')
parser.add_argument('--intra_encoder', type=bool, default=True, help = 'True/False')
parser.add_argument('--intra_decoder', type=bool, default=True, help = 'True/False')
parser.add_argument('--copy', type=bool, default=True, help = 'True/False') # for transformer

parser.add_argument('--model_type', type=str, default='seq2seq', choices=['seq2seq', 'transformer'])
parser.add_argument('--train_rl', type=bool, default=False, help = 'True/False')
parser.add_argument('--keywords', type=str, default='POS_FOP_keywords', 
                    help = 'POS_FOP_keywords / DEP_FOP_keywords / TextRank_keywords')

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--rand_unif_init_mag', type=float, default=0.02)
parser.add_argument('--trunc_norm_init_std', type=float, default=0.001)
parser.add_argument('--mle_weight', type=float, default=1.0)
parser.add_argument('--gound_truth_prob', type=float, default=0.1)

parser.add_argument('--max_enc_steps', type=int, default=1000)
parser.add_argument('--max_dec_steps', type=int, default=50)
parser.add_argument('--min_dec_steps', type=int, default=8)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--vocab_size', type=int, default=50000)
parser.add_argument('--beam_size', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=2)

parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--gradient_accum', type=int, default=1)

parser.add_argument('--load_ckpt', type=str, default='0890000', help='0800000')
parser.add_argument('--word_emb_type', type=str, default='word2Vec', help='word2Vec/glove/FastText')
parser.add_argument('--pre_train_emb', type=bool, default=True, help = 'True/False') # 若pre_train_emb為false, 則emb type為NoPretrain

opt = parser.parse_args(args=[])
config = re_config(opt)
loggerName, writerPath = getName(config)    
logger = getLogger(loggerName)
writer = SummaryWriter(writerPath)


# In[2]:

train_loader, validate_loader, vocab = getDataLoader(logger, config)


# In[3]:

from model import Model
import torch.nn as nn
import torch as T
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

load_step = None
model = Model(pre_train_emb=config.pre_train_emb, 
              word_emb_type = config.word_emb_type, 
              vocab = vocab)

model = model.cuda()
# model = nn.parallel(model)
optimizer = T.optim.Adam(model.parameters(), lr=config.lr)   
# optimizer = T.optim.Adagrad(model.parameters(),lr=config.lr, initial_accumulator_value=0.1)

load_model_path = config.save_model_path + '/%s/%s.tar' % (loggerName, config.load_ckpt)
if os.path.exists(load_model_path):
    print(load_model_path)
    model, optimizer, load_step = loadCheckpoint(logger, load_model_path, model, optimizer)


# In[ ]:

def train_one(model, config, batch):
        ''' Calculate Negative Log Likelihood Loss for the given batch. In order to reduce exposure bias,
                pass the previous generated token as input with a probability of 0.25 instead of ground truth label
        Args:
        :param enc_out: Outputs of the encoder for all time steps (batch_size, length_input_sequence, 2*hidden_size)
        :param enc_hidden: Tuple containing final hidden state & cell state of encoder. Shape of h & c: (batch_size, hidden_size)
        :param enc_padding_mask: Mask for encoder input; Tensor of size (batch_size, length_input_sequence) with values of 0 for pad tokens & 1 for others
        :param ct_e: encoder context vector for time_step=0 (eq 5 in https://arxiv.org/pdf/1705.04304.pdf)
        :param extra_zeros: Tensor used to extend vocab distribution for pointer mechanism
        :param enc_batch_extend_vocab: Input batch that stores OOV ids
        :param batch: batch object
        '''
        'Encoder data'
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage,         ct_e, enc_key_batch, enc_key_mask, enc_key_lens=             get_input_from_batch(batch, config, batch_first = True)
 
        enc_batch = model.embeds(enc_batch)  # Get embeddings for encoder input    
        enc_key_batch = model.embeds(enc_key_batch)  # Get key embeddings for encoder input

        enc_out, enc_hidden = model.encoder(enc_batch, enc_lens)
        
        'Decoder data'
        dec_batch, dec_padding_mask, dec_lens, max_dec_len, target_batch =         get_output_from_batch(batch, config, batch_first = True) # Get input and target batchs for training decoder
        step_losses = []
        s_t = (enc_hidden[0], enc_hidden[1])  # Decoder hidden states
        x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(START))  # Input to the decoder
        prev_s = None  # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None  # Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        for t in range(min(max_dec_len, config.max_dec_steps)):
            use_gound_truth = get_cuda((T.rand(len(enc_out)) > config.gound_truth_prob)).long()  # Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
            x_t = use_gound_truth * dec_batch[:, t] + (1 - use_gound_truth) * x_t  # Select decoder input based on use_ground_truth probabilities
            x_t = model.embeds(x_t)  
            final_dist, s_t, ct_e, sum_temporal_srcs, prev_s = model.decoder(x_t, s_t, enc_out, enc_padding_mask,
                                                                                      ct_e, extra_zeros,
                                                                                      enc_batch_extend_vocab,
                                                                                      sum_temporal_srcs, prev_s, enc_key_batch, enc_key_mask)
            target = target_batch[:, t]
            log_probs = T.log(final_dist + config.eps)
            step_loss = F.nll_loss(log_probs, target, reduction="none", ignore_index=PAD)
            step_losses.append(step_loss)
            x_t = T.multinomial(final_dist,1).squeeze()  # Sample words from final distribution which can be used as input in next time step

            is_oov = (x_t >= config.vocab_size).long()  # Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t.detach() + (is_oov) * UNKNOWN_TOKEN  # Replace OOVs with [UNK] token

        losses = T.sum(T.stack(step_losses, 1), 1)  # unnormalized losses for each example in the batch; (batch_size)
        batch_avg_loss = losses / dec_lens  # Normalized losses; (batch_size)
        mle_loss = T.mean(batch_avg_loss)  # Average batch loss
        return mle_loss


# In[ ]:

# @torch.no_grad()
@torch.autograd.no_grad()
def validate(validate_loader, config, model):
    model.eval()
    losses = []
#     batch = next(iter(validate_loader))
    val_num = len(iter(validate_loader))
    for idx, batch in enumerate(validate_loader):
        loss = train_one(model, config, batch)
        losses.append(loss.item())
        if idx>= val_num/10: break
    model.train()
    avg_loss = sum(losses) / len(losses)
    return avg_loss


# In[ ]:

@torch.autograd.no_grad()
def calc_running_avg_loss(loss, running_avg_loss, decay=0.99):
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    return running_avg_loss


# In[ ]:

from random import randint
@torch.autograd.no_grad()
def decode(writer, logger, step, config, model, batch, mode):
    # 動態取batch
    if mode == 'test':
#         num = len(iter(batch))
#         select_batch = None
#         rand_b_id = randint(0,num-1)
#         logger.info('test_batch : ' + str(num)+ ' ' + str(rand_b_id))
#         for idx, b in enumerate(batch):
#             if idx == rand_b_id:
#                 select_batch = b
#                 break
        select_batch = next(iter(batch))
        batch = select_batch
        if type(batch) == torch.utils.data.dataloader.DataLoader:
            batch = next(iter(batch))
    'Encoder data'
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage,         ct_e, enc_key_batch, enc_key_mask, enc_key_lens=             get_input_from_batch(batch, config, batch_first = True)

    enc_batch = model.embeds(enc_batch)  # Get embeddings for encoder input    
    enc_key_batch = model.embeds(enc_key_batch)  # Get key embeddings for encoder input

    enc_out, enc_hidden = model.encoder(enc_batch, enc_lens)

    'Feed encoder data to predict'
    pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, 
                           enc_batch_extend_vocab, enc_key_batch, enc_key_lens, model, 
                           START, END, UNKNOWN_TOKEN)

    article_sents, decoded_sents, keywords_list,     ref_sents, long_seq_index = prepare_result(vocab, batch, pred_ids)

    rouge_1, rouge_2, rouge_l = write_rouge(writer, step, mode,article_sents, decoded_sents,                 keywords_list, ref_sents, long_seq_index)

    write_bleu(writer, step, mode, article_sents, decoded_sents,                keywords_list, ref_sents, long_seq_index)

    write_group(writer, step, mode, article_sents, decoded_sents,                keywords_list, ref_sents, long_seq_index)

    return rouge_l


# In[ ]:

from random import randint
import time
@torch.autograd.no_grad()
def avg_acc(writer, logger, epoch, config, model, dataloader, mode):
    # 動態取batch
    num = len(iter(dataloader))
    avg_rouge_l = []
    acc_st, acc_cost = 0, 0
    avg_acc_cost = []
    for idx, batch in enumerate(dataloader): 
        if idx >= num/100: break
        acc_st = time.time()
        'Encoder data'
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage,         ct_e, enc_key_batch, enc_key_mask, enc_key_lens=             get_input_from_batch(batch, config, batch_first = True)

        enc_batch = model.embeds(enc_batch)  # Get embeddings for encoder input    
        enc_key_batch = model.embeds(enc_key_batch)  # Get key embeddings for encoder input

        enc_out, enc_hidden = model.encoder(enc_batch, enc_lens)

        'Feed encoder data to predict'
        pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, 
                               enc_batch_extend_vocab, enc_key_batch, enc_key_lens, model, 
                               START, END, UNKNOWN_TOKEN)

        article_sents, decoded_sents, keywords_list,         ref_sents, long_seq_index = prepare_result(vocab, batch, pred_ids)

        rouge_1, rouge_2, rouge_l = write_rouge(writer, None, None, article_sents, decoded_sents,                     keywords_list, ref_sents, long_seq_index, write = False)
        avg_rouge_l.append(rouge_l)
        acc_cost = time.time() - acc_st
        avg_acc_cost.append(acc_cost)


    avg_rouge_l = sum(avg_rouge_l) / len(avg_rouge_l)
    writer.add_scalars('scalar_avg/acc',  
                   {'%sing_avg_acc'%(mode): avg_rouge_l
                   }, epoch)
    avg_acc_cost = sum(avg_acc_cost) / len(avg_acc_cost)
#     print(avg_acc_cost)
#     avg_acc_cost = avg_acc_cost / len(avg_rouge_l)
#     print('decode 1% batches %s data, cost time %s ms' % (mode, avg_acc_cost ))
    return avg_rouge_l


# In[ ]:

def RL(model, config, batch, greedy):    
        '''Generate sentences from decoder entirely using sampled tokens as input. These sentences are used for ROUGE evaluation
        Args
        :param enc_out: Outputs of the encoder for all time steps (batch_size, length_input_sequence, 2*hidden_size)
        :param enc_hidden: Tuple containing final hidden state & cell state of encoder. Shape of h & c: (batch_size, hidden_size)
        :param enc_padding_mask: Mask for encoder input; Tensor of size (batch_size, length_input_sequence) with values of 0 for pad tokens & 1 for others
        :param ct_e: encoder context vector for time_step=0 (eq 5 in https://arxiv.org/pdf/1705.04304.pdf)
        :param extra_zeros: Tensor used to extend vocab distribution for pointer mechanism
        :param enc_batch_extend_vocab: Input batch that stores OOV ids
        :param article_oovs: Batch containing list of OOVs in each example
        :param greedy: If true, performs greedy based sampling, else performs multinomial sampling
        Returns:
        :decoded_strs: List of decoded sentences
        :log_probs: Log probabilities of sampled words
        '''
        'Encoder data'
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage,         ct_e, enc_key_batch, enc_key_mask, enc_key_lens=             get_input_from_batch(batch, config, batch_first = True)
        
        enc_batch = model.embeds(enc_batch)  # Get embeddings for encoder input    
        enc_key_batch = model.embeds(enc_key_batch)  # Get key embeddings for encoder input

        enc_out, enc_hidden = model.encoder(enc_batch, enc_lens)
        
        s_t = enc_hidden                                                                            #Decoder hidden states
        x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(START))  # Input to the decoder
        prev_s = None                                                                               #Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None                                                                    #Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        inds = []                       # Stores sampled indices for each time step
        decoder_padding_mask = []       # Stores padding masks of generated samples
        log_probs = []                                                                              #Stores log probabilites of generated samples
        mask = get_cuda(T.LongTensor(len(enc_out)).fill_(1))                                        #Values that indicate whether [STOP] token has already been encountered; 1 => Not encountered, 0 otherwise
        # Generate RL tokens and compute rl-log-loss
        # ----------------------------------------------------------------------
        for t in range(config.max_dec_steps):
            x_t = model.embeds(x_t)
            
            probs, s_t, ct_e, sum_temporal_srcs, prev_s = model.decoder(x_t, s_t, enc_out, enc_padding_mask,
                                                                                      ct_e, extra_zeros,
                                                                                      enc_batch_extend_vocab,
                                                                                      sum_temporal_srcs, prev_s, enc_key_batch, enc_key_mask)
            
            if greedy is False:
                multi_dist = Categorical(probs) # 建立以參數probs為標準的類別分佈
                # perform multinomial sampling
                x_t = multi_dist.sample()  # 將下一個時間點的x_t，視為下一個action   
                # 使用log_prob实施梯度方法 Policy Gradient，构造一个等价類別分佈的损失函数
                log_prob = multi_dist.log_prob(x_t)  
                log_probs.append(log_prob) #
            else:
                # perform greedy sampling distribution
                _, x_t = T.max(probs, dim=1)  # 因greedy以機率最大進行取樣，視為其中一個action   
            x_t = x_t.detach() # detach返回的 Variable 永远不会需要梯度
            inds.append(x_t)
            mask_t = get_cuda(T.zeros(len(enc_out)))                                                #Padding mask of batch for current time step
            mask_t[mask == 1] = 1                                                                   #If [STOP] is not encountered till previous time step, mask_t = 1 else mask_t = 0
            mask[(mask == 1) + (x_t == END) == 2] = 0                                       #If [STOP] is not encountered till previous time step and current word is [STOP], make mask = 0
            decoder_padding_mask.append(mask_t)
            is_oov = (x_t>=config.vocab_size).long()                                                #Mask indicating whether sampled word is OOV
            x_t = (1-is_oov)*x_t + (is_oov)*UNKNOWN_TOKEN                                             #Replace OOVs with [UNK] token
        # -----------------------------------End loop -----------------------------------
        inds = T.stack(inds, dim=1)
        decoder_padding_mask = T.stack(decoder_padding_mask, dim=1)
        if greedy is False:                                                                         #If multinomial based sampling, compute log probabilites of sampled words
            log_probs = T.stack(log_probs, dim=1) # 在第1个维度上stack, 增加新的维度进行堆叠
            log_probs = log_probs * decoder_padding_mask # 遮罩掉為[END] or [STOP]不計算損失           #Not considering sampled words with padding mask = 0
            lens = T.sum(decoder_padding_mask, dim=1) # 計算每個sample words生成的總長度               #Length of sampled sentence
            log_probs = T.sum(log_probs, dim=1) / lens  # 計算平均的每個句子的log loss # (bs,1)        #compute normalizied log probability of a sentence
        decoded_strs = []
        for i in range(len(enc_out)):
            id_list = inds[i].cpu().numpy() # 取出每個sample sentence 的word id list
            S = output2words(id_list, vocab, batch.art_oovs[i]) #Generate sentence corresponding to sampled words
            try:
                end_idx = S.index(data.STOP_DECODING)
                S = S[:end_idx]
            except ValueError:
                S = S
            if len(S) < 2:          #If length of sentence is less than 2 words, replace it with "xxx"; Avoids setences like "." which throws error while calculating ROUGE
                S = ["xxx"]
            S = " ".join(S)
            decoded_strs.append(S)
        return decoded_strs, log_probs


# In[ ]:

def train_one_RL(model, config, batch):
    # Self-Critical sequence training(SCST)
    sample_sents, RL_log_probs = RL(model, config, batch, greedy=False)   # multinomial sampling
    with T.autograd.no_grad():        
        greedy_sents, _ = RL(model, config, batch, greedy=True)  # greedy sampling

    sample_reward = reward_function(sample_sents, batch.original_abstract) # r(w^s):通过根据概率来随机sample词生成句子的reward值
    baseline_reward = reward_function(greedy_sents, batch.original_abstract) # r(w^):测试阶段使用greedy decoding取概率最大的词来生成句子的reward值

    batch_reward = T.mean(sample_reward).item()
    #Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
    rl_loss = -(sample_reward - baseline_reward) * RL_log_probs  # SCST梯度計算公式     
    rl_loss = T.mean(rl_loss)  
    '''
    公式的意思就是：对于如果当前sample到的词比测试阶段生成的词好，那么在这次词的维度上，整个式子的值就是负的（因为后面那一项一定为负），
    这样梯度就会上升，从而提高这个词的分数st；而对于其他词，后面那一项为正，梯度就会下降，从而降低其他词的分数
    '''                 
    return rl_loss, batch_reward


# In[ ]:

import pandas as pd
import time
from utils.write_result import *

@torch.autograd.no_grad()
def decode_write_all(writer, logger, epoch, config, model, dataloader, mode):
    # 動態取batch
    num = len(iter(dataloader))
    avg_rouge_1, avg_rouge_2, avg_rouge_l,  = [], [], []
    avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4 = [], [], [], []
    outFrame = None
    avg_time = 0
    for idx, batch in enumerate(dataloader):
        start = time.time() 
#         'Encoder data'
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage,         ct_e, enc_key_batch, enc_key_mask, enc_key_lens=             get_input_from_batch(batch, config, batch_first = True)

        enc_batch = model.embeds(enc_batch)  # Get embeddings for encoder input    
        enc_key_batch = model.embeds(enc_key_batch)  # Get key embeddings for encoder input

        enc_out, enc_hidden = model.encoder(enc_batch, enc_lens)
        
#         'Feed encoder data to predict'
        pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, 
                                enc_batch_extend_vocab, enc_key_batch, enc_key_lens, model, 
                                START, END, UNKNOWN_TOKEN)

        article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index = prepare_result(vocab, batch, pred_ids)
        cost = (time.time() - start)
        avg_time += cost
        logger.info('decode batch cost time : %s ms'%(cost / (config.batch_size)))        
        # ----------------------------------------------------
        batch_frame = {
            'article':article_sents,
            'keywords':keywords_list,
            'reference':ref_sents,
            'decoded':decoded_sents
        }
        batch_frame = pd.DataFrame(batch_frame)
        if idx == 0: outFrame = batch_frame 
        else: outFrame = pd.concat([outFrame, batch_frame], axis=0, ignore_index=True) 
        # ----------------------------------------------------
        rouge_1, rouge_2, rouge_l = write_rouge(writer, None, None, article_sents, decoded_sents,                     keywords_list, ref_sents, long_seq_index, write = False)
        Bleu_1, Bleu_2, Bleu_3, Bleu_4 = write_bleu(writer, None, None, article_sents, decoded_sents,             keywords_list, ref_sents, long_seq_index, write = False)
        # ----------------------------------------------------
        avg_rouge_1.append(rouge_1)
        avg_rouge_2.append(rouge_2)
        avg_rouge_l.append(rouge_l)
        
        avg_bleu1.append(Bleu_1)
        avg_bleu2.append(Bleu_2)
        avg_bleu3.append(Bleu_3)
        avg_bleu4.append(Bleu_4)
        # ----------------------------------------------------
    avg_rouge_1 = sum(avg_rouge_1) / num
    avg_rouge_2 = sum(avg_rouge_2) / num
    avg_rouge_l = sum(avg_rouge_l) / num
    writer.add_scalars('Rouge_avg/mode',  
                    {'avg_rouge_1': avg_rouge_1,
                    'avg_rouge_2': avg_rouge_2,
                    'avg_rouge_l': avg_rouge_l
                    }, epoch)
    # --------------------------------------               
    avg_bleu1 = sum(avg_bleu1)/num
    avg_bleu2 = sum(avg_bleu2)/num
    avg_bleu3 = sum(avg_bleu3)/num
    avg_bleu4 = sum(avg_bleu4)/num
    
    writer.add_scalars('BLEU_avg/mode',  
                    {
                    '%sing_avg_bleu1'%(mode): avg_bleu1,
                    '%sing_avg_bleu1'%(mode): avg_bleu2,
                    '%sing_avg_bleu1'%(mode): avg_bleu3,
                    '%sing_avg_bleu1'%(mode): avg_bleu4,                   
                    }, epoch)
    # --------------------------------------      
    outFrame.to_excel(writerPath + '/%s_output.xls'% mode)
    avg_time = avg_time / (num * config.batch_size) 
    with open(writerPath + '/%s_res.txt'% mode, 'w', encoding='utf-8') as f:
        f.write('Accuracy result:\n')
        f.write('##-- Rouge --##\n')
        f.write('%sing_avg_rouge_1: %s \n'%(mode, avg_rouge_1))
        f.write('%sing_avg_rouge_2: %s \n'%(mode, avg_rouge_2))
        f.write('%sing_avg_rouge_l: %s \n'%(mode, avg_rouge_l))

        f.write('##-- BLEU --##\n')
        f.write('%sing_avg_bleu1: %s \n'%(mode, avg_bleu1))
        f.write('%sing_avg_bleu2: %s \n'%(mode, avg_bleu2))
        f.write('%sing_avg_bleu3: %s \n'%(mode, avg_bleu3))
        f.write('%sing_avg_bleu4: %s \n'%(mode, avg_bleu4))

        f.write('Execute Time: %s \n' % avg_time)        
    # --------------------------------------              
    return avg_rouge_l


# In[ ]:

import time
loss_st, loss_cost = 0,0
decode_st, decode_cost = 0,0

write_train_para(writer, config)
logger.info('------Training START--------')
running_avg_loss, running_avg_rl_loss = 0, 0
sum_total_reward = 0
step = 0

try:
    for epoch in range(config.max_epochs):
        for batch in train_loader:
            step += 1
            loss_st = time.time()
            mle_loss = train_one(model, config, batch)
            if config.train_rl:
                rl_loss, batch_reward = train_one_RL(model, config, batch)             
        
                if step%1000 == 0 :
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
            if step%1000 == 0 :
                with T.autograd.no_grad():
                    train_batch_loss = mle_loss.item()
                    train_batch_rl_loss = rl_loss.item()
                    val_avg_loss = validate(validate_loader, config, model) # call batch by validate_loader
                    running_avg_loss = calc_running_avg_loss(train_batch_loss, running_avg_loss)
                    running_avg_rl_loss = calc_running_avg_loss(train_batch_rl_loss, running_avg_rl_loss)
                    running_avg_reward = sum_total_reward / step
#                     logger.info('epoch %d: %d, training batch loss = %f, running_avg_loss loss = %f, validation loss = %f'
#                                 % (epoch, step, train_batch_loss, running_avg_loss, val_avg_loss))
                    writer.add_scalars('scalar/Loss',  
                       {'train_batch_loss': train_batch_loss
                       }, step)
                    writer.add_scalars('scalar_avg/loss',  
                       {'train_avg_loss': running_avg_loss,
                        'test_avg_loss': val_avg_loss
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
                    loss_cost = time.time() - loss_st
                    if step%10000 == 0: logger.info('epoch %d|step %d| compute loss cost = %f ms'
                                % (epoch, step, loss_cost))

            if step%10 == 0:
                save_model(config, logger, model, optimizer, step, vocab, running_avg_loss,                            r_loss=0, title = loggerName)
            if step%1000 == 0 and step > 0:
                decode_st = time.time()
                train_rouge_l_f = decode(writer, logger, step, config, model, batch, mode = 'train') # call batch by validate_loader
                test_rouge_l_f = decode(writer, logger, step, config, model, validate_loader, mode = 'test') # call batch by validate_loader
                decode_cost = time.time() - decode_st
                if step%10000 == 0: logger.info('epoch %d|step %d| decode cost = %f ms'% (epoch, step, decode_cost))

                writer.add_scalars('scalar/Rouge-L',  
                   {'train_rouge_l_f': train_rouge_l_f,
                    'test_rouge_l_f': test_rouge_l_f
                   }, step)
#                 logger.info('epoch %d: %d, train_rouge_l_f = %f, test_rouge_l_f = %f'
#                                 % (epoch, step, train_rouge_l_f, test_rouge_l_f))
#         break
        logger.info('-------------------------------------------------------------')
        train_avg_acc = avg_acc(writer, logger, epoch, config, model, train_loader, mode = 'train')
        test_avg_acc = avg_acc(writer, logger, epoch, config, model, validate_loader, mode = 'test')
        logger.info('epoch %d|step %d| training batch loss = %f, running_avg_loss loss = %f, validation loss = %f'
                     % (epoch, step, train_batch_loss, running_avg_loss, val_avg_loss))
                    
        logger.info('epoch %d|step %d| train_avg_acc = %f, test_avg_acc = %f' % (epoch, step, train_avg_acc, test_avg_acc))
        if running_avg_reward > 0:
            logger.info('epoch %d|step %d| running_avg_reward = %f'% (epoch, step, running_avg_reward))
        if running_avg_rl_loss != 0:
            logger.info('epoch %d|step %d| running_avg_rl_loss = %f'% (epoch, step, running_avg_rl_loss))
        logger.info('-------------------------------------------------------------')

except Excepation as e:
        print(e)
else:
    logger.info(u'------Training SUCCESS--------')  
finally:
    logger.info(u'------Training END--------')    
#     train_avg_acc = decode_write_all(writer, logger, epoch, config, model, train_loader, mode = 'train')
    test_avg_acc = decode_write_all(writer, logger, epoch, config, model, validate_loader, mode = 'test')
    logger.info('epoch %d: train_avg_acc = %f, test_avg_acc = %f' % (epoch, train_avg_acc, test_avg_acc))
    removeLogger(logger)


# In[4]:

# config.load_ckpt = '0200000'

# logger = getLogger(loggerName)
# writer = SummaryWriter(writerPath)

# load_model_path = config.save_model_path + '/%s/%s.tar' % (logger, config.load_ckpt)

# train_loader, validate_loader, vocab = getDataLoader(logger, config)

# if os.path.exists(load_model_path):
#     model, optimizer, load_step = loadCheckpoint(logger, load_model_path, model, optimizer)
    
# train_avg_acc = decode_write_all(writer, logger, epoch, config, model, train_loader, mode = 'train')
# test_avg_acc = decode_write_all(writer, logger, epoch, config, model, validate_loader, mode = 'test')
# logger.info('epoch %d: train_avg_acc = %f, test_avg_acc = %f' % (epoch, train_avg_acc, test_avg_acc))

# ipython nbconvert --to script Pointer_generator.ipynb

