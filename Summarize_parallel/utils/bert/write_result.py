from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from utils.bert.data import output2words
import random

import warnings
import pandas as pd
import time

warnings.filterwarnings(action='ignore', category=UserWarning, module='nltk')

def prepare_result(vocab, batch, pred_ids,rand = False, clean = False):
    decoded_sents = []
    ref_sents = []
    # ref_sents2 = []
    article_sents = []
    keywords_list = []
    
    summary_len = max_summary_len = long_seq_index = 0
    for i in range(len(pred_ids)):  
        try: 
            decoded_words = output2words(pred_ids[i], vocab, batch.art_oovs[i])
        except:
            continue
        if len(decoded_words) < 2:
            decoded_words = "xxx"
        else:
            decoded_words = " ".join(decoded_words)
        if clean : 
            decoded_words = decoded_words.replace("[UNK]","")
        decoded_sents.append(decoded_words)
        summary = batch.original_abstract[i]
        if clean :  
            summary = summary.replace("<s>","").replace("</s>","")
        review = batch.original_article[i]
        ref_sents.append(summary)
        article_sents.append(review) 
        keywords = batch.key_words[i]
        keywords_list.append(str(keywords))
        summary_len = len(summary.split(" "))
        if max_summary_len < summary_len: 
            max_summary_len = summary_len
            long_seq_index = i
    try:
        if rand: long_seq_index = random.randint(0,len(pred_ids)-1)
    except Exception as e:
        # print(e)
        long_seq_index = 0
    # print(long_seq_index,random.randint(0,len(pred_ids)-1))
    return article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index

def write_rouge(writer,step,mode,article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index, write=True):
    rouge = Rouge()
    # print(decoded_sents)    
    # print(ref_sents)    
    score = rouge.get_scores(decoded_sents, ref_sents, avg = True) 
    if write:   
        writer.add_scalars('%s/rouge-1' % mode,  # 'rouge-2' , 'rouge-l'
                {'f': score['rouge-1']['f'],
                'p': score['rouge-1']['p'],
                'r': score['rouge-1']['r']}
                , step)
        writer.add_scalars('%s/rouge-2' % mode,  # 'rouge-2' , 'rouge-l'
                {'f': score['rouge-2']['f'],
                'p': score['rouge-2']['p'],
                'r': score['rouge-2']['r']}
                , step)
        writer.add_scalars('%s/rouge-l' % mode,  # 'rouge-2' , 'rouge-l'
                {'f': score['rouge-l']['f'],
                'p': score['rouge-l']['p'],
                'r': score['rouge-l']['r']}
                , step)
    return score['rouge-1']['f'], score['rouge-2']['f'], score['rouge-l']['f']
    
def write_group(writer,step,mode,article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index):
    writer.add_text('Group/%s/%s' % (step,mode), 
                    "### decoded : &nbsp;&nbsp;&nbsp;\
                    %s" % (decoded_sents[long_seq_index]), step)
    writer.add_text('Group/%s/%s' % (step,mode), 
                    "### keywords : &nbsp;&nbsp;&nbsp;\
                    " + keywords_list[long_seq_index], step)
    writer.add_text('Group/%s/%s' % (step,mode), 
                    "### ref_summary : &nbsp;&nbsp;&nbsp;\
                    " + ref_sents[long_seq_index], step)
    writer.add_text('Group/%s/%s' % (step,mode), 
                    "### review : &nbsp;&nbsp;&nbsp;\
                    " + article_sents[long_seq_index], step)        

def write_bleu(writer,step, mode, article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index, write=True):
        
    bleu_decode_sents = [decode.split(" ") for decode in decoded_sents]
    bleu_ref_sents = [[ref.split(" ")] for ref in ref_sents]
    Bleu_1 = corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(1, 0, 0, 0))
    Bleu_2 = corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0, 1, 0, 0))
    Bleu_3 = corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0, 0, 1, 0))
    Bleu_4 = corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0, 0, 0, 1))
    if write:
        writer.add_scalars('%s/BLEU' % mode,  
                {'BLEU-1': Bleu_1,
                'BLEU-2': Bleu_2,
                'BLEU-3': Bleu_3,
                'BLEU-4': Bleu_4}
                , step)
        
        # writer.add_scalars('%s/Cumulative' % mode,  # 'rouge-2' , 'rouge-l'
        #         {'BLEU-1': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(1, 0, 0, 0)),
        #         'BLEU-2': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0.5, 0.5, 0, 0)),
        #         'BLEU-3': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0.33, 0.33, 0.33, 0)),
        #         'BLEU-4': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0.25, 0.25, 0.25, 0.25))}
        #         , step)
    return Bleu_1, Bleu_2, Bleu_3, Bleu_4
    
def write_scalar(writer, logger, step, train_rouge_l_f, test_rouge_l_f):
    writer.add_scalars('scalar/Rouge-L',  
               {'train_rouge_l_f': train_rouge_l_f,
                'test_rouge_l_f': test_rouge_l_f
               }, step)
    # logger.info('step: %s train_rouge_l_f: %.3f test_rouge_l_f: %.3f \n' \
    #             % (step, train_rouge_l_f, test_rouge_l_f))
    
def write_train_para(writer, exp_config):
    data_paras = ['data_type', 'batch_size','beam_size',
    'vocab_size',
    'pre_train_emb','word_emb_type']
    data_info_str = ""

    transformer_paras = ['encoder','max_pos','use_bert_emb',
    'enc_dropout','enc_layers','enc_hidden_size','enc_heads','enc_ff_size',
    'dec_dropout','dec_layers','dec_hidden_size','dec_heads','dec_ff_size'
    ]

    model_paras = ['emb_dim','hidden_dim','emb_grad']
    model_info_str = ""
    transformer_info_str = ""

    train_paras = ['lr', 'max_dec_steps','min_dec_steps','max_enc_steps','max_epochs','mle_weight',
    'param_init','param_init_glorot','optim','lr_bert','lr_dec','beta1','beta2',
    'warmup_steps','warmup_steps_bert','warmup_steps_dec','max_grad_norm','sep_optim'
    #'rl_weight', 'train_rl'
    ]
    train_info_str = ""

    for a in dir(exp_config):
        if type(getattr(exp_config, a)) in [str,int,float,bool] \
        and 'path' not in str(a) \
        and '__' not in str(a) \
        and 'info' not in str(a):
            if a in data_paras:
                data_info_str += '## %s : %s\n'%(a,getattr(exp_config, a))
            if a in (transformer_paras + model_paras):
                if a in transformer_paras:
                    transformer_info_str += '## %s : %s\n'%(a,getattr(exp_config, a))
                if a in model_paras:
                    model_info_str += '## %s : %s\n'%(a,getattr(exp_config, a))
            if a in train_paras:
                train_info_str += '## %s : %s\n'%(a,getattr(exp_config, a))
            
                 

    model_info_str = model_info_str + transformer_info_str

    writer.add_text('Data-Parameters',data_info_str,0)
    writer.add_text('Model-Parameters',model_info_str,0)
    writer.add_text('Train-Parameters',train_info_str,0)


# @torch.autograd.no_grad()
# def decode_write_all(writer, logger, epoch, config, model, dataloader, mode):
#     # 動態取batch
#     num = len(iter(dataloader))
#     avg_rouge_1, avg_rouge_2, avg_rouge_l,  = [], [], []
#     avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4 = [], [], [], []
#     outFrame = None
#     avg_time = 0
#     for idx, batch in enumerate(dataloader):
#         start = time.time() 
#         'Encoder data'
#         enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage, \
#         ct_e, enc_key_batch, enc_key_mask, enc_key_lens= \
#             get_input_from_batch(batch, config, batch_first = True)

#         enc_batch = model.embeds(enc_batch)  # Get embeddings for encoder input    
#         enc_key_batch = model.embeds(enc_key_batch)  # Get key embeddings for encoder input

#         enc_out, enc_hidden = model.encoder(enc_batch, enc_lens)

#         'Feed encoder data to predict'
#         pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, 
#                                 enc_batch_extend_vocab, enc_key_batch, enc_key_lens, model, 
#                                 START, END, UNKNOWN_TOKEN)

#         article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index = prepare_result(vocab, batch, pred_ids)
#         cost = (time.time() - start) / 1000
#         logger.info('decode batch cost time : %s ms'%(cost / (config.batch_size)))
#         avg_time += cost
#         # ----------------------------------------------------
#         batch_frame = {
#             'article':article_sents,
#             'keywords':keywords_list,
#             'reference':ref_sents,
#             'decoded':decoded_sents
#         }
#         batch_frame = pd.DataFrame(batch_frame)
#         if idx == 0: outFrame = batch_frame 
#         else: outFrame = pd.concat([outFrame, batch_frame], axis=0, ignore_index=True) 
#         # ----------------------------------------------------
#         rouge_l, rouge_1, rouge_2 = write_rouge(writer, None, None, article_sents, decoded_sents, \
#                     keywords_list, ref_sents, long_seq_index, write = False)
#         Bleu_1, Bleu_2, Bleu_3, Bleu_4 = write_bleu(writer, None, None, article_sents, decoded_sents, \
#             keywords_list, ref_sents, long_seq_index, write = False)
#         # ----------------------------------------------------
#         avg_rouge_1.append(rouge_1)
#         avg_rouge_2.append(rouge_2)
#         avg_rouge_l.append(rouge_l)
        
#         avg_bleu1.append(Bleu_1)
#         avg_bleu2.append(Bleu_2)
#         avg_bleu3.append(Bleu_3)
#         avg_bleu4.append(Bleu_4)
#         # ----------------------------------------------------
#     avg_rouge_1 = sum(avg_rouge_1) / num
#     avg_rouge_2 = sum(avg_rouge_2) / num
#     avg_rouge_l = sum(avg_rouge_l) / num
#     writer.add_scalars('Rouge_avg/mode',  
#                     {'avg_rouge_1': avg_rouge_1,
#                     'avg_rouge_2': avg_rouge_2,
#                     'avg_rouge_l': avg_rouge_l
#                     }, epoch)
#     # --------------------------------------               
#     avg_bleu1 = sum(avg_bleu1)/num
#     avg_bleu2 = sum(avg_bleu2)/num
#     avg_bleu3 = sum(avg_bleu3)/num
#     avg_bleu4 = sum(avg_bleu4)/num
    
#     writer.add_scalars('BLEU_avg/mode',  
#                     {
#                     '%sing_avg_bleu1'%(mode): avg_bleu1,
#                     '%sing_avg_bleu1'%(mode): avg_bleu2,
#                     '%sing_avg_bleu1'%(mode): avg_bleu3,
#                     '%sing_avg_bleu1'%(mode): avg_bleu4,                   
#                     }, epoch)
#     # --------------------------------------      
#     outFrame.to_excel(writerPath + '/%s_output.xls'% mode)
#     avg_time = avg_time / (num * config.batch_size) 
#     with open(writerPath + '/%s_res.txt'% mode, 'w', encoding='utf-8') as f:
#         f.write('Accuracy result:\n')
#         f.write('##-- Rouge --##\n')
#         f.write('%sing_avg_rouge_1: %s \n'%(mode, avg_rouge_1))
#         f.write('%sing_avg_rouge_2: %s \n'%(mode, avg_rouge_2))
#         f.write('%sing_avg_rouge_l: %s \n'%(mode, avg_rouge_l))

#         f.write('##-- BLEU --##\n')
#         f.write('%sing_avg_bleu1: %s \n'%(mode, avg_bleu1))
#         f.write('%sing_avg_bleu2: %s \n'%(mode, avg_bleu2))
#         f.write('%sing_avg_bleu3: %s \n'%(mode, avg_bleu3))
#         f.write('%sing_avg_bleu4: %s \n'%(mode, avg_bleu4))

#         f.write('Execute Time:\n')
        
#     # --------------------------------------              
      
#     return avg_rouge_1, avg_rouge_2, avg_rouge_l, avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4