#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import config
from utils.bert import data

from utils.bert.batcher import *
from utils.bert.train_util import *
from utils.bert.initialize import loadCheckpoint, save_model
from utils.bert.write_result import *

from datetime import datetime as dt
from tqdm import tqdm

from tensorboardX import SummaryWriter
import argparse
from torch.distributions import Categorical

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='transformer', choices=['seq2seq', 'transformer'])
parser.add_argument('--copy', type=bool, default=True, choices=[True, False])
parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'Transformer'])
parser.add_argument("-max_pos", default=1000, type=int)
parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=True, choices=[False, True])

parser.add_argument("-lr_bert", default=2e-2, type=float, help='2e-3')
parser.add_argument("-lr_dec", default=2e-2, type=float, help='2e-3')
parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-finetune_bert", type=bool, default=True)
    
'''
原 Bert Base paper核心參數
dropout = 0.1
num_layers = 12
num_heads = 8
emb_dim(d_model) : 768
ff_embed_dim = 4*emb_dim = 3072

bert_config = BertConfig(self.encoder.model.config.vocab_size, hidden_size=768,
                                     num_hidden_layers=12, num_attention_heads=8,
                                     intermediate_size= 3072,
                                     hidden_dropout_prob=0.1,
                                     attention_probs_dropout_prob=0.1)
'''
parser.add_argument("-enc_dropout", default=0.1, type=float)
parser.add_argument("-enc_layers", default=10, type=int)
parser.add_argument("-enc_hidden_size", default=768, type=int)
parser.add_argument("-enc_heads", default=8, type=int)
parser.add_argument("-enc_ff_size", default=3072, type=int)

parser.add_argument("-dec_dropout", default=0.1, type=float)
parser.add_argument("-dec_layers", default=10, type=int)
parser.add_argument("-dec_hidden_size", default=768, type=int)
parser.add_argument("-dec_heads", default=8, type=int)
parser.add_argument("-dec_ff_size", default=2048, type=int)
parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=True, choices=[False, True])

parser.add_argument("-param_init", default=0, type=float)
parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-optim", default='adam', type=str)
parser.add_argument("-lr", default=1, type=float)
parser.add_argument("-beta1", default= 0.9, type=float)
parser.add_argument("-beta2", default=0.999, type=float)
parser.add_argument("-warmup_steps", default=8000, type=int)
parser.add_argument("-warmup_steps_bert", default=8000, type=int)
parser.add_argument("-warmup_steps_dec", default=8000, type=int)
parser.add_argument("-max_grad_norm", default=0, type=float)

parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)


parser.add_argument('--train_rl', type=bool, default=False, help = 'True/False')
parser.add_argument('--keywords', type=str, default='POS_keys', 
                    help = 'POS_keys / DEP_keys / Noun_adj_keys / TextRank_keys')

parser.add_argument('--mle_weight', type=float, default=1.0)
parser.add_argument("-label_smoothing", default=0.0, type=float)
parser.add_argument("-generator_shard_size", default=32, type=int)
parser.add_argument("-alpha",  default=0.6, type=float)

parser.add_argument('--max_enc_steps', type=int, default=1000)
parser.add_argument('--max_dec_steps', type=int, default=40)
parser.add_argument('--min_dec_steps', type=int, default=10)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--vocab_size', type=int, default=50000)
parser.add_argument('--beam_size', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=4)

# parser.add_argument('--hidden_dim', type=int, default=512)
# parser.add_argument('--emb_dim', type=int, default=512)
parser.add_argument('--gradient_accum', type=int, default=1)

parser.add_argument('--load_ckpt', type=str, default='0000010', help='0000010')
# parser.add_argument('--word_emb_type', type=str, default='glove', help='word2Vec/glove/FastText')
# parser.add_argument('--pre_train_emb', type=bool, default=False, help = 'True/False') # 若pre_train_emb為false, 則emb type為NoPretrain

opt = parser.parse_args(args=[])
config = re_config(opt)

loggerName, writerPath = getName(config)    
logger = getLogger(loggerName)
writer = SummaryWriter(writerPath)


# In[2]:


train_loader, validate_loader, vocab, symbols = getDataLoader(logger, config)
tokenizer = vocab.tokenizer
train_batches = len(iter(train_loader))
test_batches = len(iter(validate_loader))
save_steps = int(train_batches/1000)*1000


# In[3]:


from utils.transformer.loss import *
from utils.transformer.optimizers import Optimizer
from transformer import *
from utils.transformer.predictor import build_predictor
import torch.nn as nn
import torch
from parallel import DataParallelModel, DataParallelCriterion
# https://gist.github.com/thomwolf/7e2407fbd5945f07821adae3d9fd1312

model = AbsSummarizer(config)

load_model_path = config.save_model_path + '/%s/%s.tar' % (loggerName, config.load_ckpt)
if os.path.exists(load_model_path):
    model, optimizer, load_step = loadCheckpoint(config, logger, load_model_path, model)
else:    
    if (config.sep_optim):
        optim_bert = Optimizer(
            config.optim, config.lr_bert, config.max_grad_norm,
            beta1=config.beta1, beta2=config.beta2,
            decay_method='noam',
            warmup_steps=config.warmup_steps_bert)

        optim_dec = Optimizer(
            config.optim, config.lr_dec, config.max_grad_norm,
            beta1=config.beta1, beta2=config.beta2,
            decay_method='noam',
            warmup_steps=config.warmup_steps_dec)
        
        params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('encoder.model')]
        optim_bert.set_parameters(params)

        params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('encoder.model')]
        optim_dec.set_parameters(params)

        optimizer = [optim_bert, optim_dec]
    else:
        optimizer = Optimizer(
            config.optim, config.lr, config.max_grad_norm,
            beta1=config.beta1, beta2=config.beta2,
            decay_method='noam',
            warmup_steps=config.warmup_steps)
        optimizer.set_parameters(list(model.named_parameters()))
        optimizer = [optimizer]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
setattr(config, 'device_ids', [0])

model = get_cuda(model)
# net = nn.DataParallel(model, device_ids=config.device_ids)
# model = nn.DataParallel(model).cuda()
model.to(device) 


# In[4]:


parallel_model = DataParallelModel(model) # Encapsulate the model

criterion = choose_criterion(config, model.vocab_size)
parallel_loss = DataParallelCriterion(criterion)


# In[5]:



# loss, num_correct, target = compute_loss(None, criterion, pred, dec_batch[:,1:], num_tokens, tokenizer)
# # --------------------------------------------------------------------------------
# acc = accuracy(num_correct, num_tokens)
# cross_entropy = xent(loss, num_tokens)
# perplexity = ppl(loss, num_tokens)


# In[6]:


def merge_res(res):
    ((pred1, attn1),(pred2, attn2)) = res
    merge_pred = torch.cat([pred1, pred2], dim = 0).cuda(pred1.device.index)
    attn = torch.cat([attn1, attn2], dim = 0)
    return (pred1, pred2), attn, merge_pred

def compute_loss(preds, target, merge_pred, num_tokens, tokenizer):
    gtruth = target   
    loss = parallel_loss(config.mle_weight , preds, gtruth) 
    num_correct = compute_correct(merge_pred, target, num_tokens, tokenizer)  
    return loss, num_correct, target

def get_package(inputs):    
    # ----------------------------------------------------
    normalization = 0
    'Encoder data'
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, _, _, _, _, _, enc_seg, enc_cls, enc_cls_mask = get_input_from_batch(inputs, config, batch_first = True)
    # ----------------------------------------------------
    'Decoder data'
    dec_batch, dec_padding_mask, dec_lens, max_dec_len, target_batch = get_output_from_batch(inputs, config, batch_first = True) # Get input and target 
    num_tokens = dec_batch[:, 1:].ne(0).sum()
    normalization += num_tokens.item() 
    return (enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, enc_seg, enc_cls, enc_cls_mask,            dec_batch, dec_padding_mask, dec_lens, max_dec_len, target_batch,             num_tokens, normalization)
    
    
for inputs in train_loader:  
#     package = get_package(inputs)
#     mle_loss = train_one(package)
#     print(loss)

    gold_tgt_len = inputs.dec_tgt.size(1)
    setattr(config, 'min_length',gold_tgt_len + 20)
    setattr(config, 'max_length',gold_tgt_len + 60)
    predictor = build_predictor(config, tokenizer, symbols, model, logger)
    
    # 'Encoder data'
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, _, \
    _, _, _, _, enc_seg, enc_cls, enc_cls_mask = \
        get_input_from_batch(inputs, config, batch_first = True)

    # 'Decoder data'
    dec_batch, dec_padding_mask, dec_lens, max_dec_len, target_batch = \
    get_output_from_batch(inputs, config, batch_first = True) # Get input and target
    
    setattr(inputs, 'src',enc_batch)
    setattr(inputs, 'segs',enc_seg)
    setattr(inputs, 'mask_src',enc_padding_mask)

    inputs_data = predictor.translate_batch(inputs)
    translations = predictor.from_batch(inputs_data) # translation = (pred_sents, gold_sent, raw_src)
    article_sents = [t[2] for t in translations]
    decoded_sents = [t[0] for t in translations]
    ref_sents = [t[1] for t in translations]


# In[ ]:




