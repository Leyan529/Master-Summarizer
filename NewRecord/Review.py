
# coding: utf-8

# In[1]:

#%%
# NEW_PROD_DICT 
from data_util.product import *
from data_util.mainCat import *
from data_util.MongoDB import *
from data_util.stopwords import *
from data_util.preprocess import *
#%%
import spacy
# gpu = spacy.prefer_gpu()
# print('GPU:', gpu)
# pip install -U spacy[cuda100]
# python -m spacy validate
# python -m spacy download en_core_web_sm

from spacy import displacy
from collections import Counter
import en_core_web_sm
from pprint import pprint

nlp = en_core_web_sm.load()
import re
# from stopwords import *
import nltk
# from preprocess import *
# from feature import *

from textblob import TextBlob
import collections

from spacy.symbols import cop, acomp, amod, conj, neg, nn, nsubj, dobj
from spacy.symbols import VERB, NOUN, PROPN, ADJ, ADV, AUX, PART

pattern_counter = collections.Counter()

from nltk.corpus import stopwords
import networkx as nx
# from MongoDB import MongoDB
import os

import dateutil.parser as parser
from datetime import datetime
from tqdm import tqdm
import nltk

mode = 'prod'
# mode = 'main_cat'
# mode = 'mix-5'
if mode == 'prod':
    # Load Spec From Mongo
    # from product import *
    # mongoObj = DVD_Player()
    mongoObj = Cameras()
    # mongoObj = Cell_Phones()
    # mongoObj = GPS()
    # mongoObj = Keyboards()
    #--------------------------------#
    # mongoObj = Home_Kitchen()
    # mongoObj = Cloth_Shoes_Jewelry()
    # mongoObj = Grocery_Gourmet_Food()
    # mongoObj = Automotive()
    # mongoObj = Toys_Games()
    #--------------------------------#
    main_cat, category1, category2, cond_date = mongoObj.getAttr()
    print("make data dict from category1 : %s category2 : %s" % (category1, category2))
    folder = category2
    key = mongoObj.getReviewKey()

elif mode == 'main_cat':   
    # mongoObj = All_Electronics()
    # mongoObj = Pet_Supplies()
    # mongoObj = Sports_Outdoors()
    # mongoObj = Health_personal_Care()
    # -------------------------------------#
    mongoObj = CellPhones_Accessories()
    # mongoObj = Camera_Photo()
    # mongoObj = GPS_Navigation()
    # mongoObj = Music_Instrum()
    # mongoObj = Software()
    # mongoObj = Computers()
    # mongoObj = Video_Games()
    # -------------------------------------#
    main_cat = mongoObj.getAttr()
    print("make data dict from main_cat : %s " % (main_cat))
    folder = main_cat
    key = mongoObj.getReviewKey()

'''
cd D:\WorkSpace\JupyterWorkSpace\Text-Summarizer-BERT2\
D:
activate tensorflow
python Review.py
'''


# Total Opinion
opinion_lexicon = {}
for filename in os.listdir('opinion-lexicon-English/'):      
    if "txt" not in filename: continue
    print(filename)
    with open('opinion-lexicon-English/'+filename,'r') as f_input:
        lexion = []
        for line in f_input:
            if line.startswith(";"):
                continue
            word = line.replace("\n","")
            if word != "" : lexion.append(word)
        pos = filename.replace(".txt","")
        opinion_lexicon[pos] = lexion

opinion_lexicon["total-words"] = opinion_lexicon["negative-words"] + opinion_lexicon["positive-words"]
print("total-words 已取得")


# In[115]:

from transformers import BertModel, BertTokenizer 
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import re
from data_util.stopwords import *

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def remove_word5(text):
    text = str(text)
    text = text.lower()

    for k, v in contractions.items():
        if k in text:
            text = text.replace(k, v)

    for k in html_escape_table:
        if k in text:
            text = text.replace(k, "")

    text = text.replace("-", '')   
    text = text.replace("\"", '')
    text = text.replace("\n", '')  
    text = text.replace('" ', '')
    text = text.replace(' "', '')

    remove_chars = '["#$%&\'\"\()*+:<=>@★【】《》“”‘’[\\]^_`{|}~]+'
    text = re.sub(remove_chars, "", text)  # remove number and segment

    text = text.replace("\\", '')
    text = text.replace("/", '')
    return text

def process(text):
    if type(text) == list: 
        if len(text) == 1 : text = text[0]
        else: text = "\n".join(text)
    text = remove_tags(text)
    text = re.sub(r'http\S+', '', text)
    text = remove_word5(text)
    
#     dash_words = []
#     pattern = re.compile(r"([\d\w\.-]+[-'//.][\d\w\.-]+)")  # |  ([\(](\w)+[\)])
#     dash_words.extend(pattern.findall(text))

#     for sym in dash_words:    
#         text_sym = sym.replace('-','_').replace('.','_')
#         text = text.replace(sym,text_sym)    
    
    text = [line.strip() for line in text.split("\n") if line != '']
    text = "\n".join(text)
    
    text = " ".join(bert_tokenizer.tokenize(text))
    text = text.replace(" ##","")
    text = text.split(" . ")
    text = [line + " . " for line in text]
    text = [line for line in text if len(line) > 5]
    text = text[:-1] + [text[-1].replace(". "," . ").replace('.  .','.')]
    return text

def process2(text):
    if type(text) == list: 
        if len(text) == 1 : text = text[0]
        else: text = "\n".join(text)
    text = remove_tags(text)
    text = re.sub(r'http\S+', '', text)
    text = remove_word5(text)
    
#     dash_words = []
#     pattern = re.compile(r"([\d\w\.-]+[-'//.][\d\w\.-]+)")  # |  ([\(](\w)+[\)])
#     dash_words.extend(pattern.findall(text))

#     for sym in dash_words:    
#         text_sym = sym.replace('-','_').replace('.','_')
#         text = text.replace(sym,text_sym)    
    
    text = [line.strip() for line in text.split("\n") if line != '']
    text = "\n".join(text)
    
    text = " ".join(bert_tokenizer.tokenize(text))
    text = text.replace(" ##","")
    text = text.split(" ")
    text = [line for line in text]
    return text

rev = """
<a data-hook="product-link-linked" class="a-link-normal" href="/Crocheting-For-Dummies/dp/B001C4PKLW/ref=cm_cr_arp_d_rvw_txt?ie=UTF8">Crocheting For Dummies</a>Though the sample version for the Kindle included the entire table of contents, there were no images in the sample.  How can you tell if it's a good illustration if you don't get a sample of an illustration?  Thanks!
"""


# In[116]:

from data_util.stopwords import *
from data_util.preprocess import *
from data_util.eda import *

def lemm_keyword(text):
    text_keywords = []
    lemm_sents = []
    nt_chars = '[’!"#$%&\'\"\,()*+:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}.~]+'
    for line in text:
        lemm_text = lemm_sent_process4(line, remove_stopwords=False, summary=False, mode="spacy",withdot=False)
        lemm_text = lemm_text.replace("\n",' ')
        text_keywords2 = PF_rule_POS(lemm_text).run()
        lemm_sents.append(lemm_text)
        text_keywords.extend(text_keywords2)
    last_sent = lemm_sents[-1]
    s = re.findall(nt_chars,last_sent)
    if len(s) != 0:     
        last_sent = re.sub(nt_chars, " ", last_sent)  
        last_sent = last_sent + "."  
        lemm_sents[-1] = last_sent
    lemm_article = " ".join(lemm_sents)
    return lemm_sents, text_keywords, lemm_article

rev = process(rev)
rev
lemm_keyword(rev)[0]

summ = """
New hobby 4 a seventy-four year old man, Amazon.com
"""

def lemm_summ(text):
    text_keywords = []
    lemm_sents = []
    for line in text:
        lemm_text = lemm_sent_process4(line, remove_stopwords=False, summary=False, mode="spacy",withdot=False)
        lemm_text = lemm_text.replace("\n",'')
        lemm_sents.append(lemm_text)
    lemm_sents = " ".join(lemm_sents)
    return lemm_sents


def lemm_summ2(line):
    lemm_text = lemm_sent_process4(line, remove_stopwords=False, summary=False, mode="spacy",withdot=False)
    lemm_text = lemm_text.replace("\n",'')
    lemm_tokens = [token for token in lemm_text.split(" ") if token.isalpha()]
    lemm_text = " ".join(lemm_tokens)
    return lemm_text

summ = process(summ)
summ
lemm_summ(summ)


# In[82]:

def loadProdReviewData():
    global category1, category2, main_cat, cond_date

    # Connect MongoDB
    print("Connect to MongoDB")
    mongo = MongoDB()
    mongo.conn_db(db_name='Amazon')

    rev_db_col = 'new_reviews2'
    review_cursor = mongo.searchInDB(key, db_col=rev_db_col)
    docCount = review_cursor.count()
    print("make product reviews feature from %s reviews..." % (docCount))
    print("Search reviews finished...")  
    return review_cursor , docCount    

# review_cursor , docCount = loadProdReviewData()
nlp = en_core_web_sm.load()  


# In[ ]:
from collections import Counter, OrderedDict
import pandas as pd

def make_orign_rev_xlsx():
    asin_list, review_list, overall_list, vote_list, summary_list, review_ID_list  = [] , [] , [] , [] , [] , [] 
    review_cursor , docCount = loadProdReviewData()

    with tqdm(total=docCount) as pbar:
        for i1, rev in enumerate(review_cursor):
            asin, review, overall, vote, summary, review_ID = rev["asin"], rev["reviewText"], rev['overall'], rev['vote'], rev['summary'], str(rev['unixReviewTime'])       

            asin_list.append(asin)
            review_list.append(review)
            overall_list.append(overall)
            vote_list.append(vote)
            summary_list.append(summary)
            review_ID_list.append(review_ID)
            pbar.update(1)        
            
        df = pd.DataFrame({"asin":asin_list, "review": review_list, "overall": overall_list, "vote": vote_list,
                            "summary": summary_list , "review_ID": review_ID_list
                            })
        
    return df

if not os.path.exists(folder):
    os.makedirs(folder)
        

csv_path = '%s/orign_review.xlsx'%(folder)    
print("check ", os.path.exists(csv_path))
if os.path.exists(csv_path):
    df = pd.read_excel(csv_path)
    print("previous file %s ...."%(csv_path)) 
else:
    df = make_orign_rev_xlsx()
    df.to_excel(csv_path, encoding='utf8')
    print(csv_path + " Write finished") 

from collections import Counter, OrderedDict
import pandas as pd
import threading
import multiprocessing 
import time

corpus_path = '%s/corpus.txt'%(folder) 
corpus = open(corpus_path,'w',encoding='utf-8')

total_keywords = set()
feature_counter = Counter()
docCount = len(df)            
asin_list, review_list, overall_list, vote_list, summary_list, review_ID_list , cheat_num_list = [] , [] , [] , [] , [] , [] , []
lemm_review_len_list , lemm_summary_len_list = [] , [] 
cheat_list = []

def func(asin, review, overall, vote, summary, review_ID):
    global asin_list, review_list, overall_list, vote_list, summary_list, review_ID_list , cheat_num_list
    global lemm_review_len_list , lemm_summary_len_list , corpus
    lock.acquire()
    time.sleep(0.01)
    try:                       
        review = process(review)   
        # print(review)          
        sents , rev_keywords , lemm_review = lemm_keyword(review)
        # print(sents) 
        # print('-----------------------------')
        feature_counter.update(rev_keywords)
        lemm_review_len = len(lemm_review.split(" "))
        [corpus.write(sent + "\n") for sent in sents]
    except Exception as e :
        pass  
    # ---------------------------------------------------------------------------------------------
    try: 
        # --------------------------------------------------- single summary
        # # print(summary)           
        # summary = process2(summary)
        # # print(summary)
        # lemm_summary = lemm_summ(summary)
        # # print(lemm_summary) 
        # # print('-----------------------------')
        # corpus.write(lemm_summary + "\n") 
        # lemm_summary = '<s> ' + lemm_summary + " </s>"
        # lemm_summary_len = len(lemm_summary.split(" "))     
        # --------------------------------------------------- single summary    
        lemm_summary = ''
        # print(summary) 
        summarys = nltk.sent_tokenize(" ".join(process2(summary)))
        # print(summarys)
        for summary in summarys:
            lemm_s = lemm_summ2(summary)
            # corpus.write(lemm_s + "\n") 
            lemm_s = '<s> ' + lemm_s + " </s> "
            lemm_summary = lemm_summary + lemm_s
        lemm_summary_len = len(lemm_summary.split(" "))
        # print(lemm_summary)
        corpus.write(lemm_summary)
    except Exception as e :
        print(e)           
    # ----------------------------------------------------------------------------------------------
    rev_token_set = set(lemm_review.split(" "))
    summ_token_set = set(lemm_summary.split(" "))
    cheat_num = len(rev_token_set & summ_token_set) 
    # ----------------------------------------------------------------------------------------------
    pbar.set_description("%s training-pair " % (folder))
    pbar.update(1)
    
    asin_list.append(asin)
    review_list.append(lemm_review)
    overall_list.append(overall)
    vote_list.append(vote)
    summary_list.append(lemm_summary)
    review_ID_list.append(review_ID)
    cheat_num_list.append(cheat_num) 
    lemm_review_len_list.append(lemm_review_len)
    lemm_summary_len_list.append(lemm_summary_len)
    lock.release()

with tqdm(total=docCount) as pbar:
    # threads = []
    lock = threading.BoundedSemaphore(5) #最多允許10個執行緒同時執行
    # lock = multiprocessing.BoundedSemaphore(10)
    # lock = None
    for i in range(docCount):
        rev = df.iloc[i]
        pbar.update(1)

        asin, review, overall, vote, summary, review_ID = rev["asin"], rev["review"], rev['overall'], rev['vote'], rev['summary'], str(rev['review_ID'])       
        
        # t = threading.Thread(target=func,args=(asin, review, overall, vote, summary, review_ID,))
        # t = multiprocessing.Process(target=func,args=(asin, review, overall, vote, summary, review_ID,))
        # threads.append(t)
        # t.start()

        try:                       
            review = process(review)   
            # print(review)          
            sents , rev_keywords , lemm_review = lemm_keyword(review)
            # print(sents) 
            # print('-----------------------------')
            feature_counter.update(rev_keywords)
            lemm_review_len = len(lemm_review.split(" "))
            [corpus.write(sent + "\n") for sent in sents]
        except Exception as e :
            continue  
        # ---------------------------------------------------------------------------------------------
        try: 
            # --------------------------------------------------- single summary
            # # print(summary)           
            # summary = process2(summary)
            # # print(summary)
            # lemm_summary = lemm_summ(summary)
            # # print(lemm_summary) 
            # # print('-----------------------------')
            # corpus.write(lemm_summary + "\n") 
            # lemm_summary = '<s> ' + lemm_summary + " </s>"
            # lemm_summary_len = len(lemm_summary.split(" "))     
            # --------------------------------------------------- single summary    
            lemm_summary = ''
            # print(summary) 
            summarys = nltk.sent_tokenize(" ".join(process2(summary)))
            # print(summarys)
            for summary in summarys:
                if len(summary) < 2: continue
                lemm_s = lemm_summ2(summary)
                # corpus.write(lemm_s + "\n") 
                lemm_s = '<s> ' + lemm_s + " </s> "
                lemm_summary = lemm_summary + lemm_s
            lemm_summary_len = len(lemm_summary.split(" "))
            # print(lemm_summary)
            corpus.write(lemm_summary)           
        except Exception as e :
            continue           
        # ----------------------------------------------------------------------------------------------
        rev_token_set = set(lemm_review.split(" "))
        summ_token_set = set(lemm_summary.split(" "))
        # cheat = rev_token_set & summ_token_set & ( total_keywords | set(opinion_lexicon["total-words"]) )
        cheat = rev_token_set & summ_token_set & (set(opinion_lexicon["total-words"]) )

        # cheat = rev_token_set & summ_token_set
        cheat = set([c for c in cheat if c != ''])
        cheat_num = len(cheat) 
        # ----------------------------------------------------------------------------------------------
        pbar.set_description("%s training-pair " % (folder))
        
        asin_list.append(asin)
        review_list.append(lemm_review)
        overall_list.append(overall)
        vote_list.append(vote)
        summary_list.append(lemm_summary)
        review_ID_list.append(review_ID)
        cheat_num_list.append(cheat_num) 
        cheat_list.append(cheat)
        lemm_review_len_list.append(lemm_review_len)
        lemm_summary_len_list.append(lemm_summary_len)        
        
    
    # for t in threads:
    #     t.join()

    df = pd.DataFrame({"asin":asin_list, "review": review_list, "overall": overall_list, "vote": vote_list,
                        "summary": summary_list , "review_ID": review_ID_list, "cheat_num": cheat_num_list,
                        "cheat": cheat_list ,
                        "lemm_review_len": lemm_review_len_list , "lemm_summary_len": lemm_summary_len_list})
    corpus.close()


if not os.path.exists(folder):
    os.makedirs(folder)

csv_path = '%s/review.xlsx'%(folder)     
df.head()
df.to_excel(csv_path, encoding='utf8')
print(csv_path + " Write finished")          


# In[ ]:

df.head()


# In[ ]:
important_features = OrderedDict(sorted(feature_counter.items(), key=lambda pair: pair[1], reverse=True))
important_features = [(word, important_features[word]) for word in important_features if important_features[word] > 0]
print("Count : %s" % (len(important_features)))

fn3 = '%s/review_keywords.txt'%(folder)
with open(fn3, 'w', encoding="utf-8") as f:
    total_keywords = set()
    for word, v in important_features:
        f.write("%s:%s \n" % (word, v))
        total_keywords.add(word)

# In[ ]:
xlsx_path = '%s/review.xlsx'%(folder)  
orign_key_df = pd.read_excel(xlsx_path)
print(xlsx_path + " Read finished")
len(orign_key_df)