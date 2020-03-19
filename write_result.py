from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
import random

def prepare_result(data,vocab, batch, mode, pred_ids,rand = True, clean = True):
    decoded_sents = []
    ref_sents = []
    # ref_sents2 = []
    article_sents = []
    keywords_list = []
    
    summary_len = max_summary_len = long_seq_index = 0
    for i in range(len(pred_ids)):            
        decoded_words = data.outputids2words(pred_ids[i], vocab, batch.rev_oovs[i])
        if len(decoded_words) < 2:
            decoded_words = "xxx"
        else:
            decoded_words = " ".join(decoded_words)
        if clean : 
            decoded_words = decoded_words.replace("[UNK]","")
        decoded_sents.append(decoded_words)
        summary = batch.original_summarys[i]
        if clean :  
            summary = summary.replace("<s>","").replace("</s>","")
        review = batch.original_reviews[i]
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

def write_rouge(writer,iter,mode,article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index):
    rouge = Rouge()    
    score = rouge.get_scores(decoded_sents, ref_sents, avg = True)    
    writer.add_scalars('%s/rouge-1' % mode,  # 'rouge-2' , 'rouge-l'
            {'f': score['rouge-1']['f'],
            'p': score['rouge-1']['p'],
            'r': score['rouge-1']['r']}
            , iter)
    writer.add_scalars('%s/rouge-2' % mode,  # 'rouge-2' , 'rouge-l'
            {'f': score['rouge-2']['f'],
            'p': score['rouge-2']['p'],
            'r': score['rouge-2']['r']}
            , iter)
    writer.add_scalars('%s/rouge-l' % mode,  # 'rouge-2' , 'rouge-l'
            {'f': score['rouge-l']['f'],
            'p': score['rouge-l']['p'],
            'r': score['rouge-l']['r']}
            , iter)

    return score['rouge-l']['f']
    
def write_group(writer,iter,mode,article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index):
    writer.add_text('Group/%s/%s' % (iter,mode), 
                    "### decoded : &nbsp;&nbsp;&nbsp;\
                    %s" % (decoded_sents[long_seq_index]), iter)
    writer.add_text('Group/%s/%s' % (iter,mode), 
                    "### keywords : &nbsp;&nbsp;&nbsp;\
                    " + keywords_list[long_seq_index], iter)
    writer.add_text('Group/%s/%s' % (iter,mode), 
                    "### ref_summary : &nbsp;&nbsp;&nbsp;\
                    " + ref_sents[long_seq_index], iter)
    writer.add_text('Group/%s/%s' % (iter,mode), 
                    "### review : &nbsp;&nbsp;&nbsp;\
                    " + article_sents[long_seq_index], iter)        

def write_bleu(writer,iter, mode, article_sents, decoded_sents, keywords_list, ref_sents, long_seq_index):
        
    bleu_decode_sents = [decode.split(" ") for decode in decoded_sents]
    bleu_ref_sents = [[ref.split(" ")] for ref in ref_sents]

    writer.add_scalars('%s/Individual' % mode,  # 'rouge-2' , 'rouge-l'
            {'BLEU-1': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(1, 0, 0, 0)),
            'BLEU-2': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0, 1, 0, 0)),
            'BLEU-3': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0, 0, 1, 0)),
            'BLEU-4': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0, 0, 0, 1))}
            , iter)
    
    writer.add_scalars('%s/Cumulative' % mode,  # 'rouge-2' , 'rouge-l'
            {'BLEU-1': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(1, 0, 0, 0)),
            'BLEU-2': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0.5, 0.5, 0, 0)),
            'BLEU-3': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0.33, 0.33, 0.33, 0)),
            'BLEU-4': corpus_bleu(bleu_ref_sents,bleu_decode_sents, weights=(0.25, 0.25, 0.25, 0.25))}
            , iter)

