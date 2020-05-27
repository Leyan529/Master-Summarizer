from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score

from utils.seq2seq.data import output2words
import random

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='nltk')

import pandas as pd
import time

def longest_common_subsequence(main_string, comparing_string):

    # main_string = main_string.split(" ")
    # comparing_string = comparing_string.split(" ")
    columns_length = len(main_string)  # Get the length of the first word or base word
    rows_length = len(comparing_string)  # Get the length of the second word or comparing word

    # MAKE A 2D LIST (MATRIX)
    dynamic_table = [[0] * (columns_length + 1) for i in range(rows_length + 1)]

    # rows_length = NUMBER OF ROWS
    # columns_length = NUMBER OF COLUMNS
    
    # FILL THE MATRIX FOLLOWING LCS ALGORITHM.
    for i in range(1, rows_length + 1):
        for j in range(1, columns_length + 1):
            if main_string[j - 1] == comparing_string[i - 1]:
                dynamic_table[i][j] = 1 + dynamic_table[i - 1][j - 1]

            else:
                dynamic_table[i][j] = max(dynamic_table[i - 1][j], dynamic_table[i][j - 1])

    # print("MATRIX ACCORDING TO LONGEST COMMON SUBSEQUENCE ALGORITHM: \n ")

    # for i in range(rows_length + 1):
    #     print(dynamic_table[i])

    #print("LENGTH OF LONGEST COMMON SUBSEQUENCE = ", dynamic_table[rows_length][columns_length])

    len_lcs = dynamic_table[rows_length][columns_length]

    i = len(comparing_string)
    j = len(main_string)

    lcs_string = str()

    # BACKTRACKING TO FIND THE LONGEST COMMON SUBSEQUENCE

    temp = True

    while temp is True:
        if dynamic_table[i][j] == 0:
            temp = False
        elif dynamic_table[i][j] == dynamic_table[i][j - 1]:
            j = j - 1

        else:
            lcs_string = main_string[j-1] + " " + lcs_string
            i = i - 1
            j = j - 1

    return lcs_string, len_lcs

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

def total_evaulate(article_sents, keywords_list, decoded_sents, ref_sents):
    # overlap = [len(set(article_sents[i].split(" ")) & set(ref_sents[i].split(" "))) for i in range(len(article_sents))]
    # too_overlap = [overlap[i] > len(set(ref_sents[i].split(" ")))-3 for i in range(len(article_sents))]
    
    # token_lcs, len_lcs = longest_common_subsequence(rev_tokens, summary_tokens)
    lcs_info = [longest_common_subsequence(article_sents[i].split(" "), ref_sents[i].split(" ")) for i in range(len(article_sents))]
    overlap = [info[1] for info in lcs_info]
    overlap_percent = [ (overlap[i] / len(ref_sents[i].split(" ")))*100  for i in range(len(article_sents))]
    gen_type = ["Ext" if p > 50 else "Abs" for p in overlap_percent]
    rouge = Rouge() 
    scores = rouge.get_scores(decoded_sents, ref_sents, avg = False)
    rouge_1 = [score['rouge-1']['f'] for score in scores]
    rouge_2 = [score['rouge-2']['f'] for score in scores]
    rouge_l = [score['rouge-l']['f'] for score in scores]
    '''
        累加的和单独的1元组BLEU使用相同的权重，也就是（1,0,0,0）。计算累加的2元组BLEU分数为1元组和2元组分别赋50％的权重，
        计算累加的3元组BLEU为1元组，2元组和3元组分别为赋33％的权重            
        在描述文本生成系统的性能时，通常会报告从BLEU-1到BLEU-4的累加分数
    '''
    # self-BLEU
    self_Bleu_1 = [
        sentence_bleu([ref_sents[i].split(" ")], decode.split(" "), weights=(1, 0, 0, 0)) \
        for i, decode in enumerate(decoded_sents)
    ]
    self_Bleu_2 = [
        sentence_bleu([ref_sents[i].split(" ")], decode.split(" "), weights=(0, 1, 0, 0)) \
        for i, decode in enumerate(decoded_sents)
    ]
    self_Bleu_3 = [
        sentence_bleu([ref_sents[i].split(" ")], decode.split(" "), weights=(0, 0, 1, 0)) \
        for i, decode in enumerate(decoded_sents)
    ]
    self_Bleu_4 = [
        sentence_bleu([ref_sents[i].split(" ")], decode.split(" "), weights=(0, 0, 0, 1)) \
        for i, decode in enumerate(decoded_sents)
    ]   
    # commulate-BLEU(BLEU)
    Bleu_1 = [
        sentence_bleu([ref_sents[i].split(" ")], decode.split(" "), weights=(1, 0, 0, 0)) \
        for i, decode in enumerate(decoded_sents)
    ]
    Bleu_2 = [
        sentence_bleu([ref_sents[i].split(" ")], decode.split(" "), weights=(0.5, 0.5, 0, 0)) \
        for i, decode in enumerate(decoded_sents)
    ]
    Bleu_3 = [
        sentence_bleu([ref_sents[i].split(" ")], decode.split(" "), weights=(0.33, 0.33, 0.33, 0)) \
        for i, decode in enumerate(decoded_sents)
    ]
    Bleu_4 = [
        sentence_bleu([ref_sents[i].split(" ")], decode.split(" "), weights=(0.25, 0.25, 0.25, 0.25)) \
        for i, decode in enumerate(decoded_sents)
    ]    
    Meteor = [
        single_meteor_score(ref_sents[i], decode) \
        for i, decode in enumerate(decoded_sents)
    ]
    
    batch_frame = {
        'article':article_sents,
        'keywords':keywords_list,
        'reference':ref_sents,
        'decoded':decoded_sents,            
        'rouge_1':rouge_1,
        'rouge_2':rouge_2,
        'rouge_l':rouge_l,            
        'self_Bleu_1':self_Bleu_1,
        'self_Bleu_2':self_Bleu_2,
        'self_Bleu_3':self_Bleu_3,
        'self_Bleu_4':self_Bleu_4,
        'Bleu_1':Bleu_1,
        'Bleu_2':Bleu_2,
        'Bleu_3':Bleu_3,
        'Bleu_4':Bleu_4,
        'Meteor':Meteor,
        'article_lens': [len(r.split(" ")) for r in article_sents],
        'ref_lens': [len(r.split(" ")) for r in ref_sents],
        'overlap': overlap,
        'overlap_percent': overlap_percent,
        'gen_type': gen_type
    }
    batch_frame = pd.DataFrame(batch_frame)
    return rouge_1, rouge_2, rouge_l, self_Bleu_1, self_Bleu_2, self_Bleu_3, self_Bleu_4, \
            Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, batch_frame 

def total_output(mode, writerPath, outFrame, avg_time, avg_rouge_1, avg_rouge_2, avg_rouge_l, \
    avg_self_bleu1, avg_self_bleu2, avg_self_bleu3, avg_self_bleu4, \
    avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4, avg_meteor
    ):
    #     print(avg_rouge_1)
    num = len(avg_rouge_1)
    avg_rouge_1 = sum(avg_rouge_1) / len(avg_rouge_1)
    avg_rouge_2 = sum(avg_rouge_2) / len(avg_rouge_2)
    avg_rouge_l = sum(avg_rouge_l) / len(avg_rouge_l)
    # --------------------------------------        
    # print(avg_bleu1)
    avg_self_bleu1 = sum(avg_self_bleu1)/len(avg_self_bleu1)
    avg_self_bleu2 = sum(avg_self_bleu2)/len(avg_self_bleu2)
    avg_self_bleu3 = sum(avg_self_bleu3)/len(avg_self_bleu3)
    avg_self_bleu4 = sum(avg_self_bleu4)/len(avg_self_bleu4)
    
    avg_bleu1 = sum(avg_bleu1)/len(avg_bleu1)
    avg_bleu2 = sum(avg_bleu2)/len(avg_bleu2)
    avg_bleu3 = sum(avg_bleu3)/len(avg_bleu3)
    avg_bleu4 = sum(avg_bleu4)/len(avg_bleu4)
    # --------------------------------------    
    avg_meteor = sum(avg_meteor)/len(avg_meteor)  
    
    # avg_time = avg_time / (num * config.batch_size) 
    with open(writerPath + '/%s_res.txt'% mode, 'w', encoding='utf-8') as f:
        f.write('Accuracy result:\n')
        f.write('##-- Rouge --##\n')
        f.write('%sing_avg_rouge_1: %s \n'%(mode, avg_rouge_1))
        f.write('%sing_avg_rouge_2: %s \n'%(mode, avg_rouge_2))
        f.write('%sing_avg_rouge_l: %s \n'%(mode, avg_rouge_l))

        f.write('##-- SELF-BLEU --##\n')
        f.write('%sing_avg_self_bleu1: %s \n'%(mode, avg_self_bleu1))
        f.write('%sing_avg_self_bleu2: %s \n'%(mode, avg_self_bleu2))
        f.write('%sing_avg_self_bleu3: %s \n'%(mode, avg_self_bleu3))
        f.write('%sing_avg_self_bleu4: %s \n'%(mode, avg_self_bleu4))
        
        f.write('##-- BLEU --##\n')
        f.write('%sing_avg_bleu1: %s \n'%(mode, avg_bleu1))
        f.write('%sing_avg_bleu2: %s \n'%(mode, avg_bleu2))
        f.write('%sing_avg_bleu3: %s \n'%(mode, avg_bleu3))
        f.write('%sing_avg_bleu4: %s \n'%(mode, avg_bleu4))
        
        f.write('##-- Meteor --##\n')
        f.write('%sing_avg_meteor: %s \n'%(mode, avg_meteor))

        f.write('Num : %s Execute Time: %s \n' % (num,avg_time))       
    # --------------------------------------     
    outFrame = outFrame.sort_values(by=['article_lens'], ascending = False)
    writeFrame = outFrame[:1000]
    writeFrame.to_excel(writerPath + '/%s_output.xls'% mode)
    
    read_info = open(writerPath + '/%s_res.txt'% mode, 'r', encoding='utf-8').readlines()
    print(mode,'\n',read_info)
    return avg_rouge_l, outFrame