from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from utils.data import output2words
import random

def prepare_result(vocab, batch, mode, pred_ids,rand = False, clean = False):
    decoded_sents = []
    ref_sents = []
    # ref_sents2 = []
    article_sents = []
    keywords_list = []
    
    summary_len = max_summary_len = long_seq_index = 0
    for i in range(len(pred_ids)):            
        decoded_words = output2words(pred_ids[i], vocab, batch.art_oovs[i])
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

def write_rouge(writer,step,mode,article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index):
    rouge = Rouge()
    # print(decoded_sents)    
    # print(ref_sents)    
    score = rouge.get_scores(decoded_sents, ref_sents, avg = True)    
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
    return score['rouge-l']['f']
    
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

def write_bleu(writer,step, mode, article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index):
        
    bleu_decode_sents = [decode.split(" ") for decode in decoded_sents]
    bleu_ref_sents = [[ref.split(" ")] for ref in ref_sents]

    writer.add_scalars('%s/BLEU' % mode,  
            {'BLEU-1': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(1, 0, 0, 0)),
            'BLEU-2': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0, 1, 0, 0)),
            'BLEU-3': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0, 0, 1, 0)),
            'BLEU-4': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0, 0, 0, 1))}
            , step)
    
#     writer.add_scalars('%s/Cumulative' % mode,  # 'rouge-2' , 'rouge-l'
#             {'BLEU-1': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(1, 0, 0, 0)),
#             'BLEU-2': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0.5, 0.5, 0, 0)),
#             'BLEU-3': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0.33, 0.33, 0.33, 0)),
#             'BLEU-4': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0.25, 0.25, 0.25, 0.25))}
#             , step)
def write_scalar(writer, logger, step, train_rouge_l_f, test_rouge_l_f):
    writer.add_scalars('scalar/Rouge-L',  
               {'train_rouge_l_f': train_rouge_l_f,
                'test_rouge_l_f': test_rouge_l_f
               }, step)
    logger.info('step: %s train_rouge_l_f: %.3f test_rouge_l_f: %.3f \n' \
                % (step, train_rouge_l_f, test_rouge_l_f))
    
def write_train_para(writer, exp_config):
    info_str = ''
    for a in dir(exp_config):
        if type(getattr(exp_config, a)) in [str,int,float,bool] \
        and 'path' not in str(a) \
        and '__' not in str(a) \
        and 'info' not in str(a):
            info_str += '## %s : %s\n'%(a,getattr(exp_config, a))

    writer.add_text('Parameters',info_str,0)