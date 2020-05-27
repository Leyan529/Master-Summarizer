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
from utils.seq2seq.data import output2words
import argparse
from utils.seq2seq.rl_util import *
from torch.distributions import Categorical

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

parser = argparse.ArgumentParser()
parser.add_argument('--key_attention', type=bool, default=False, help = 'True/False')
parser.add_argument('--intra_encoder', type=bool, default=True, help = 'True/False')
parser.add_argument('--intra_decoder', type=bool, default=True, help = 'True/False')
parser.add_argument('--copy', type=bool, default=True, help = 'True/False') # for transformer

parser.add_argument('--model_type', type=str, default='seq2seq', choices=['seq2seq', 'transformer'])
parser.add_argument('--train_rl', type=bool, default=True, help = 'True/False')
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
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--vocab_size', type=int, default=50000)
parser.add_argument('--beam_size', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--gradient_accum', type=int, default=1)

parser.add_argument('--load_ckpt', type=str, default='0378000', help='0002000')
parser.add_argument('--word_emb_type', type=str, default='word2Vec', help='word2Vec/glove/FastText')
parser.add_argument('--pre_train_emb', type=bool, default=True)

opt = parser.parse_args(args=[])
config = re_config(opt)
# loggerName, writerPath = getName(config)    
loggerName = 'TextRank'
writerPath = 'runs/%s/%s/exp'% (config.data_type, loggerName)
if not os.path.exists(writerPath): os.makedirs(writerPath)
logger = getLogger(loggerName)
# writer = SummaryWriter(writerPath)
writer = None


# In[2]:


train_loader, validate_loader, vocab = getDataLoader(logger, config)
train_batches = len(iter(train_loader))
test_batches = len(iter(validate_loader))
save_steps = int(train_batches/1000)*1000


# In[3]:


import pandas as pd
import time
from summa.summarizer import summarize
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

    for idx, batch in enumerate(dataloader):
        start = time.time() 
        article_sents = [article for article in batch.original_article]
        ref_sents = [ref for ref in batch.original_abstract ]
        decoded_sents = [summarize(article, words=30) for article in article_sents]
        decoded_sents = [sent if len(sent) > 5 else "xxx xxx xxx xxx xxx" for sent in decoded_sents]
        
        keywords_list = [str(word_list) for word_list in batch.key_words]
        cost = (time.time() - start)
        avg_time += cost        
        try:
            rouge_1, rouge_2, rouge_l, self_Bleu_1, self_Bleu_2, self_Bleu_3, self_Bleu_4,                 Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, batch_frame = total_evaulate(article_sents, keywords_list, decoded_sents, ref_sents)
        except Exception as e :
            continue
            
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


# In[4]:


epoch = 0
model = None
# model    
train_avg_acc, train_outFrame = decode_write_all(writer, logger, epoch, config, model, train_loader, mode = 'train')
logger.info('-----------------------------------------------------------')
test_avg_acc, test_outFrame = decode_write_all(writer, logger, epoch, config, model, validate_loader, mode = 'test')
logger.info('epoch %d: train_avg_acc = %f, test_avg_acc = %f' % (epoch, train_avg_acc, test_avg_acc))

# !ipython nbconvert --to script Pointer_generator.ipynb
train_outFrame.head()
test_outFrame.head()
removeLogger(logger)


# In[5]:


# centroid
# textrank

