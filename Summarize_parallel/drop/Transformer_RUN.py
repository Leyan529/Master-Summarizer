
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

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
parser.add_argument("-encoder", default='Transformer', type=str, choices=['bert', 'Transformer'])
parser.add_argument("-max_pos", default=1000, type=int)
parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False, choices=[False, True])

parser.add_argument("-lr_bert", default=2e-3, type=float)
parser.add_argument("-lr_dec", default=2e-3, type=float)
parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-finetune_bert", type=bool, default=True)
    
'''
bert_config = BertConfig(self.encoder.model.config.vocab_size, hidden_size=768,
                                     num_hidden_layers=12, num_attention_heads=8,
                                     intermediate_size= 3072,
                                     hidden_dropout_prob=0.1,
                                     attention_probs_dropout_prob=0.1)
'''
parser.add_argument("-enc_dropout", default=0.2, type=float)
parser.add_argument("-enc_layers", default=6, type=int)
parser.add_argument("-enc_hidden_size", default=768, type=int)
parser.add_argument("-enc_heads", default=8, type=int)
parser.add_argument("-enc_ff_size", default=2048, type=int)

parser.add_argument("-dec_dropout", default=0.2, type=float)
parser.add_argument("-dec_layers", default=6, type=int)
parser.add_argument("-dec_hidden_size", default=768, type=int)
parser.add_argument("-dec_heads", default=8, type=int)
parser.add_argument("-dec_ff_size", default=2048, type=int)
parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False, choices=[False, True])

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
parser.add_argument("-label_smoothing", default=0.1, type=float)
parser.add_argument("-generator_shard_size", default=32, type=int)
parser.add_argument("-alpha",  default=0.6, type=float)

parser.add_argument('--max_enc_steps', type=int, default=500)
parser.add_argument('--max_dec_steps', type=int, default=50)
parser.add_argument('--min_dec_steps', type=int, default=8)
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--vocab_size', type=int, default=50000)
parser.add_argument('--beam_size', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=2)

parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--emb_dim', type=int, default=512)
parser.add_argument('--gradient_accum', type=int, default=1)

parser.add_argument('--load_ckpt', type=str, default='0000010', help='0000010')
# parser.add_argument('--word_emb_type', type=str, default='glove', help='word2Vec/glove/FastText')
parser.add_argument('--pre_train_emb', type=bool, default=False, help = 'True/False') # 若pre_train_emb為false, 則emb type為NoPretrain

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

model = AbsSummarizer(config)

load_model_path = config.save_model_path + '/%s/%s.tar' % (loggerName, config.load_ckpt)
if os.path.exists(load_model_path):
    model, optimizer, load_step = loadCheckpoint(config, logger, load_model_path, model)
else:    
    if (config.sep_optim):
        optim_bert = optim = Optimizer(
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
model = get_cuda(model)


# In[4]:

model


# In[ ]:

def train_one(model, config, batch):
    normalization = 0
    'Encoder data'
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, _, _, _, _, _, enc_seg, enc_cls, enc_cls_mask = get_input_from_batch(batch, config, batch_first = True)

    'Decoder data'
    dec_batch, dec_padding_mask, dec_lens, max_dec_len, target_batch = get_output_from_batch(batch, config, batch_first = True) # Get input and target 

    
    num_tokens = dec_batch[:, 1:].ne(0).sum()
    normalization += num_tokens.item()    

    pred, state = model(enc_batch, dec_batch, enc_seg, 
        enc_cls, enc_padding_mask, dec_padding_mask, enc_cls_mask, 
        extra_zeros, enc_batch_extend_vocab)
    criterion = choose_criterion(config, model.vocab_size)
    loss, num_correct, target = compute_loss(model, criterion, pred, dec_batch[:,1:], num_tokens, tokenizer)
    # loss = loss / normalization  # Normalized losses; (batch_size)
    # --------------------------------------------------------------------------------
    acc = accuracy(num_correct, num_tokens)
    cross_entropy = xent(loss, num_tokens)
    perplexity = ppl(loss, num_tokens)

    print("num_tokens:%s; acc: %6.2f; perplexity: %5.2f; cross entropy loss: %4.2f" 
                            % (num_tokens,
                            acc,
                            perplexity,
                            cross_entropy
                            ))

    # if step % 100 == 0:
    #     if acc > 0:
    #         print("Step %5d; num_tokens:%s; acc: %6.2f; perplexity: %5.2f; cross entropy loss: %4.2f" 
    #                         % (step,num_tokens,
    #                         acc,
    #                         perplexity,
    #                         cross_entropy
    #                         ))
            # print('scores',scores.shape)
            # print('target',target)
    #     # >>>>>>>> DEBUG Session <<<<<<<<<
    # print('------------------------------------')
    # print("ENC\n")
    # print(enc_batch.shape)
    # print("DEC\n")
    # print(dec_batch.shape)
    # print("TGT\n")
    # print(target_batch.shape)
    # print("ENCP\n")
    # print(enc_padding_mask.shape)
    # print("DECP\n")
    # print(dec_padding_mask.shape)
    # print("enc_seg\n")
    # print(enc_seg.shape)
    # print("enc_cls\n")
    # print(enc_cls.shape)
    # print("enc_cls_mask\n")
    # print(enc_cls_mask.shape)

    return loss.div(float(normalization))    
    # return loss



# In[ ]:

# @torch.no_grad()
@torch.autograd.no_grad()
def validate(validate_loader, config, model):
    model.eval()
    losses = []
    # batch = next(iter(validate_loader))
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
        # num = len(iter(batch))
        # select_batch = None
        # rand_b_id = randint(0,num-1)
        # logger.info('test_batch : ' + str(num)+ ' ' + str(rand_b_id))
        # for idx, b in enumerate(batch):
        #     if idx == rand_b_id:
        #         select_batch = b
        #         break
        select_batch = next(iter(batch))
        batch = select_batch
        if type(batch) == torch.utils.data.dataloader.DataLoader:
            batch = next(iter(batch))

    # ---------------------------------------------------------------------------
    '''
    batch_data = self.translate_batch(batch)
    translations = self.from_batch(batch_data)
    '''
    gold_tgt_len = batch.dec_tgt.size(1)
    setattr(config, 'min_length',gold_tgt_len + 20)
    setattr(config, 'max_length',gold_tgt_len + 60)
    predictor = build_predictor(config, tokenizer, symbols, model, logger)

    # 'Encoder data'
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, _, _, _, _, _, enc_seg, enc_cls, enc_cls_mask =         get_input_from_batch(batch, config, batch_first = True)

    # 'Decoder data'
    dec_batch, dec_padding_mask, dec_lens, max_dec_len, target_batch = get_output_from_batch(batch, config, batch_first = True) # Get input and target 

    setattr(batch, 'src',enc_batch)
    setattr(batch, 'segs',enc_seg)
    setattr(batch, 'mask_src',enc_padding_mask)

    batch_data = predictor.translate_batch(batch)
    translations = predictor.from_batch(batch_data) # translation = (pred_sents, gold_sent, raw_src)
    article_sents = [t[2] for t in translations]
    decoded_sents = [t[0] for t in translations]
    ref_sents = [t[1] for t in translations]
    keywords_list = [str(word_list) for word_list in batch.key_words]
#     print('decoded_sents',decoded_sents)
    # ---------------------------------------------------------------------------
    rouge_1, rouge_2, rouge_l = write_rouge(writer, None, None, article_sents, decoded_sents,                     keywords_list, ref_sents, 0, write = False)
    write_bleu(writer, step, mode, article_sents, decoded_sents,                keywords_list, ref_sents, 0)

    write_group(writer, step, mode, article_sents, decoded_sents,                keywords_list, ref_sents, 0)

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
        if idx >= num/10000: break
        acc_st = time.time()
        # ---------------------------------------------------------------------------
        '''
        batch_data = self.translate_batch(batch)
        translations = self.from_batch(batch_data)
        '''
        gold_tgt_len = batch.dec_tgt.size(1)
        setattr(config, 'min_length',gold_tgt_len + 20)
        setattr(config, 'max_length',gold_tgt_len + 60)
        predictor = build_predictor(config, tokenizer, symbols, model, logger)

        # 'Encoder data'
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, _,         _, _, _, _, enc_seg, enc_cls, enc_cls_mask =             get_input_from_batch(batch, config, batch_first = True)

        # 'Decoder data'
        dec_batch, dec_padding_mask, dec_lens, max_dec_len, target_batch =         get_output_from_batch(batch, config, batch_first = True) # Get input and target 

        setattr(batch, 'src',enc_batch)
        setattr(batch, 'segs',enc_seg)
        setattr(batch, 'mask_src',enc_padding_mask)

        batch_data = predictor.translate_batch(batch)
        translations = predictor.from_batch(batch_data) # translation = (pred_sents, gold_sent, raw_src)
        article_sents = [t[2] for t in translations]
        decoded_sents = [t[0] for t in translations]
        ref_sents = [t[1] for t in translations]
        keywords_list = [str(word_list) for word_list in batch.key_words]


        rouge_1, rouge_2, rouge_l = write_rouge(writer, None, None, article_sents, decoded_sents,                         keywords_list, ref_sents, 0, write = False)
        # ---------------------------------------------------------------------------
        avg_rouge_l.append(rouge_l)
        acc_cost = time.time() - acc_st
        avg_acc_cost.append(acc_cost)


    avg_rouge_l = sum(avg_rouge_l) / len(avg_rouge_l)
    writer.add_scalars('scalar_avg/acc',  
                   {'%sing_avg_acc'%(mode): avg_rouge_l
                   }, epoch)
    avg_acc_cost = sum(avg_acc_cost) / len(avg_acc_cost)
    return avg_rouge_l


# In[ ]:

import pandas as pd
import time

@torch.autograd.no_grad()
def decode_write_all(writer, logger, epoch, config, model, dataloader, mode):
    # 動態取batch
    num = len(iter(dataloader))
    avg_rouge_1, avg_rouge_2, avg_rouge_l,  = [], [], []
    avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4 = [], [], [], []
    outFrame = None
    avg_time = 0
    
    rouge = Rouge()  
    
    for idx, batch in enumerate(dataloader):
        start = time.time() 
        gold_tgt_len = batch.dec_tgt.size(1)
        setattr(config, 'min_length',gold_tgt_len + 20)
        setattr(config, 'max_length',gold_tgt_len + 60)
        predictor = build_predictor(config, tokenizer, symbols, model, logger)

        # 'Encoder data'
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, _,  _, _, _, _, enc_seg, enc_cls, enc_cls_mask =             get_input_from_batch(batch, config, batch_first = True)

        # 'Decoder data'
        dec_batch, dec_padding_mask, dec_lens, max_dec_len, target_batch =         get_output_from_batch(batch, config, batch_first = True) # Get input and target 

        setattr(batch, 'src',enc_batch)
        setattr(batch, 'segs',enc_seg)
        setattr(batch, 'mask_src',enc_padding_mask)

        batch_data = predictor.translate_batch(batch)
        translations = predictor.from_batch(batch_data) # translation = (pred_sents, gold_sent, raw_src)
        article_sents = [t[2] for t in translations]
        decoded_sents = [t[0] for t in translations]
        ref_sents = [t[1] for t in translations]
        keywords_list = [str(word_list) for word_list in batch.key_words]
        cost = (time.time() - start)

        avg_time += cost        

        overlap = [len(set(article_sents[i].split(" ")) & set(ref_sents[i].split(" "))) for i in range(len(article_sents))]
        too_overlap = [overlap[i] > len(set(ref_sents[i].split(" ")))-3 for i in range(len(article_sents))]
        scores = rouge.get_scores(decoded_sents, ref_sents, avg = False)
        rouge_1 = [score['rouge-1']['f'] for score in scores]
        rouge_2 = [score['rouge-2']['f'] for score in scores]
        rouge_l = [score['rouge-l']['f'] for score in scores]

        batch_frame = {
            'article':article_sents,
            'keywords':keywords_list,
            'reference':ref_sents,
            'decoded':decoded_sents,
            'ref_lens': [len(r.split(" ")) for r in ref_sents],
            'overlap': overlap,
            'too_overlap': too_overlap,
            'rouge_1':rouge_1,
            'rouge_2':rouge_2,
            'rouge_l':rouge_l
#             'Bleu_1':Bleu_1,
#             'Bleu_2':Bleu_2,
#             'Bleu_3':Bleu_3,
#             'Bleu_4':Bleu_4,
        }
        batch_frame = pd.DataFrame(batch_frame)
        if idx %1000 == 0: print(idx)
        if idx == 0: outFrame = batch_frame
        elif idx < 100 : outFrame = pd.concat([outFrame, batch_frame], axis=0, ignore_index=True) 
        # ----------------------------------------------------
        avg_rouge_1.extend(rouge_1)
        avg_rouge_2.extend(rouge_2)
        avg_rouge_l.extend(rouge_l)        
#         avg_bleu1.append(Bleu_1)
#         avg_bleu2.append(Bleu_2)
#         avg_bleu3.append(Bleu_3)
#         avg_bleu4.append(Bleu_4)
        # ----------------------------------------------------
#     print(avg_rouge_1)
    avg_rouge_1 = sum(avg_rouge_1) / len(avg_rouge_1)
    avg_rouge_2 = sum(avg_rouge_2) / len(avg_rouge_2)
    avg_rouge_l = sum(avg_rouge_l) / len(avg_rouge_l)
    writer.add_scalars('Rouge_avg/mode',  
                    {'avg_rouge_1': avg_rouge_1,
                    'avg_rouge_2': avg_rouge_2,
                    'avg_rouge_l': avg_rouge_l
                    }, epoch)
    # --------------------------------------               
#     avg_bleu1 = sum(avg_bleu1)/len(avg_bleu1)
#     avg_bleu2 = sum(avg_bleu2)/len(avg_bleu2)
#     avg_bleu3 = sum(avg_bleu3)/len(avg_bleu3)
#     avg_bleu4 = sum(avg_bleu4)/len(avg_bleu4)
    
#     writer.add_scalars('BLEU_avg/mode',  
#                     {
#                     '%sing_avg_bleu1'%(mode): avg_bleu1,
#                     '%sing_avg_bleu1'%(mode): avg_bleu2,
#                     '%sing_avg_bleu1'%(mode): avg_bleu3,
#                     '%sing_avg_bleu1'%(mode): avg_bleu4,                   
#                     }, epoch)
    # --------------------------------------      
    outFrame.to_excel(writerPath + '/%s_output.xls'% mode)
    avg_time = avg_time / (num * config.batch_size) 
    with open(writerPath + '/%s_res.txt'% mode, 'w', encoding='utf-8') as f:
        f.write('Accuracy result:\n')
        f.write('##-- Rouge --##\n')
        f.write('%sing_avg_rouge_1: %s \n'%(mode, avg_rouge_1))
        f.write('%sing_avg_rouge_2: %s \n'%(mode, avg_rouge_2))
        f.write('%sing_avg_rouge_l: %s \n'%(mode, avg_rouge_l))

#         f.write('##-- BLEU --##\n')
#         f.write('%sing_avg_bleu1: %s \n'%(mode, avg_bleu1))
#         f.write('%sing_avg_bleu2: %s \n'%(mode, avg_bleu2))
#         f.write('%sing_avg_bleu3: %s \n'%(mode, avg_bleu3))
#         f.write('%sing_avg_bleu4: %s \n'%(mode, avg_bleu4))

        f.write('Execute Time: %s \n' % avg_time)        
    # --------------------------------------              
    return avg_rouge_l, outFrame


# In[ ]:

import time
loss_st, loss_cost = 0,0
decode_st, decode_cost = 0,0

write_train_para(writer, config)
logger.info('------Training START--------')
running_avg_loss, running_avg_rl_loss = 0, 0
sum_total_reward = 0
step = 0
save_steps = 2
# try:
for epoch in range(config.max_epochs):
    for batch in train_loader:
        step += 1
        loss_st = time.time()
        mle_loss = train_one(model, config, batch)
        if config.train_rl:
            rl_loss, batch_reward = train_one_RL(model, config, batch)             
        else:
            rl_loss = T.FloatTensor([0]).cuda()
        (config.mle_weight * mle_loss + config.rl_weight * rl_loss).backward()  # 反向传播，计算当前梯度

        model.zero_grad() # 清空过往梯度
        '''梯度累加就是，每次获取1个batch的数据，计算1次梯度，梯度不清空'''
        if step % (config.gradient_accum) == 0: # gradient accumulation
                # clip_grad_norm_(model.parameters(), 5.0)                     
            for o in optimizer:
                o.step() # 根据累计的梯度更新网络参数
            
                
        if step%1000 == 0 :
            with T.autograd.no_grad():
                train_batch_loss = mle_loss.item()
                train_batch_rl_loss = rl_loss.item()
                val_avg_loss = validate(validate_loader, config, model) # call batch by validate_loader
                running_avg_loss = calc_running_avg_loss(train_batch_loss, running_avg_loss)
                running_avg_rl_loss = calc_running_avg_loss(train_batch_rl_loss, running_avg_rl_loss)
                running_avg_reward = sum_total_reward / step
                if step % save_steps == 0:
                    logger.info('epoch %d: %d, training batch loss = %f, running_avg_loss loss = %f, validation loss = %f'
                                % (epoch, step, train_batch_loss, running_avg_loss, val_avg_loss))
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
                if step % save_steps == 0: logger.info('epoch %d|step %d| compute loss cost = %f ms'
                            % (epoch, step, loss_cost))
        if step % 500 == 0:
            save_model(config, logger, model, optimizer, step, vocab, running_avg_loss,                            r_loss=0, title = loggerName)
        if step%save_steps == 0 and step > 0:
            decode_st = time.time()
            train_rouge_l_f = decode(writer, logger, step, config, model, batch, mode = 'train') # call batch by validate_loader
            test_rouge_l_f = decode(writer, logger, step, config, model, validate_loader, mode = 'test') # call batch by validate_loader
            decode_cost = time.time() - decode_st
            if step%save_steps == 0: logger.info('epoch %d|step %d| decode cost = %f ms'% (epoch, step, decode_cost))

            writer.add_scalars('scalar/Rouge-L',  
                {'train_rouge_l_f': train_rouge_l_f,
                'test_rouge_l_f': test_rouge_l_f
                }, step)
            if step%save_steps == 0:
                logger.info('epoch %d: %d, train_rouge_l_f = %f, test_rouge_l_f = %f'
                            % (epoch, step, train_rouge_l_f, test_rouge_l_f))
#         break
    logger.info('-------------------------------------------------------------')
    train_avg_acc = avg_acc(writer, logger, epoch, config, model, train_loader, mode = 'train')
    test_avg_acc = avg_acc(writer, logger, epoch, config, model, validate_loader, mode = 'test')                   
    logger.info('epoch %d|step %d| train_avg_acc = %f, test_avg_acc = %f' % (epoch, step, train_avg_acc, test_avg_acc))
    if running_avg_reward > 0:
        logger.info('epoch %d|step %d| running_avg_reward = %f'% (epoch, step, running_avg_reward))
    if running_avg_rl_loss != 0:
        logger.info('epoch %d|step %d| running_avg_rl_loss = %f'% (epoch, step, running_avg_rl_loss))
    logger.info('-------------------------------------------------------------')

# except Excepation as e:
#         print(e)
# else:
#     logger.info(u'------Training SUCCESS--------')  
# finally:
#     logger.info(u'------Training END--------')    
#     train_avg_acc, train_outFrame = decode_write_all(writer, logger, epoch, config, model, train_loader, mode = 'train')
#     test_avg_acc, test_outFrame = decode_write_all(writer, logger, epoch, config, model, validate_loader, mode = 'test')
#     logger.info('epoch %d: train_avg_acc = %f, test_avg_acc = %f' % (epoch, train_avg_acc, test_avg_acc))
#     removeLogger(logger)


# In[ ]:

# train_outFrame.head()
# test_outFrame.head()

# !ipython nbconvert --to script Transformer.ipynb

