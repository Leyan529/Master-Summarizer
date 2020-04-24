from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from utils.data import output2words
import random

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='nltk')

import pandas as pd
import time

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
        decoded_sents.append(decoded_words.strip())
        summary = batch.original_abstract[i].strip()
        if clean :  
            summary = summary.replace("<s>","").replace("</s>","")
        review = batch.original_article[i].strip()
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

def write_rouge(writer,step,mode,article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index,write=True):
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
    writer.add_text('Compose/%s/%s' % (step,mode), 
                    "### decoded : &nbsp;&nbsp;&nbsp;\
                    %s" % (decoded_sents[long_seq_index]), step)
    writer.add_text('Compose/%s/%s' % (step,mode), 
                    "### keywords : &nbsp;&nbsp;&nbsp;\
                    " + keywords_list[long_seq_index], step)
    writer.add_text('Compose/%s/%s' % (step,mode), 
                    "### ref_summary : &nbsp;&nbsp;&nbsp;\
                    " + ref_sents[long_seq_index], step)
    writer.add_text('Compose/%s/%s' % (step,mode), 
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
    'key_attention','max_key_num','keywords','vocab_size',
    'pre_train_emb','word_emb_type']
    data_info_str = ""

    transformer_paras = ['copy', 'coverage']
    pg_paras = ['intra_decoder', 'intra_encoder','eps',
    'gound_truth_prob','trunc_norm_init_std','rand_unif_init_mag']
    model_paras = ['emb_dim','hidden_dim','emb_grad']
    model_info_str = ""
    transformer_info_str = ""
    pg_info_str = ""

    train_paras = ['lr', 'max_dec_steps','min_dec_steps',
    'max_enc_steps','max_epochs','mle_weight','rl_weight',
    'train_rl']
    train_info_str = ""

    for a in dir(exp_config):
        if type(getattr(exp_config, a)) in [str,int,float,bool] \
        and 'path' not in str(a) \
        and '__' not in str(a) \
        and 'info' not in str(a):
            if a in data_paras:
                data_info_str += '## %s : %s\n'%(a,getattr(exp_config, a))
            if a in (transformer_paras + pg_paras + model_paras):
                if a in transformer_paras:
                    transformer_info_str += '## %s : %s\n'%(a,getattr(exp_config, a))
                if a in pg_paras:
                    pg_info_str += '## %s : %s\n'%(a,getattr(exp_config, a))
                if a in model_paras:
                    model_info_str += '## %s : %s\n'%(a,getattr(exp_config, a))
            if a in train_paras:
                train_info_str += '## %s : %s\n'%(a,getattr(exp_config, a))
            
                 

    if getattr(exp_config, 'model_type') == 'seq2seq':
        model_info_str = model_info_str + pg_info_str
    else:
        model_info_str = model_info_str + transformer_info_str

    writer.add_text('Data-Parameters',data_info_str,0)
    writer.add_text('Model-Parameters',model_info_str,0)
    writer.add_text('Train-Parameters',train_info_str,0)