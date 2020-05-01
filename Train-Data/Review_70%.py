
# coding: utf-8
# use opinion-lexicon-English
# use data_util
# In[1]:

#%%
# NEW_PROD_DICT 

from data_util.product import *
from data_util.mainCat import *
from data_util.mixCat import *
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

# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))

import networkx as nx
# from MongoDB import MongoDB
import os

import dateutil.parser as parser
from datetime import datetime
from tqdm import tqdm
import nltk

# mode = 'prod'
# mode = 'main_cat'
mode = 'mixCat'

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
    # mongoObj = CellPhones_Accessories()
    mongoObj = Camera_Photo()
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
elif mode == 'mixCat':   
    mongoObj = Mix6()
    # mongoObj = Mix10()
    
    main_cat = mongoObj.getAttr()
    print("make data dict from Mix cat : %s " % (main_cat))
    folder = main_cat
    key = mongoObj.getReviewKey()

'''
cd D:\WorkSpace\JupyterWorkSpace\Text-Summarizer-BERT2\
D:
activate tensorflow
python Review3.py
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
# from data_util.stopwords import *

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
    return text



# In[116]:

from data_util.stopwords import *
from data_util.preprocess import *
from data_util.eda import *
from bs4 import BeautifulSoup

alphbet_stopword = ['b','c','d','e','f','g','h','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','#']

def text_cleaner(text):
    # print(text)
    text = squeeze3(text) # 優先過濾...

    # 移除符號特徵
    # ----------------------------------------------------------------------
    pattern = re.compile(r"([\d\w\.-]+[-'//.][\d\w\.-]+)")  # |  ([\(](\w)+[\)])
    keys = pattern.findall(text)           
    # 移除符號特徵    
    # keywords.extend(keys)
    for k in keys:    
        k = k.replace("(","").replace(")","")
        text = text.replace(k,"(" + k + ")")
        text = text.replace("((","(").replace("))",")")     
        text = text.replace("(" + k + ")","") 
        text = text.strip()
    # ----------------------------------------------------------------------

    text = remove_word5(text)
    newString = text.lower()
    # newString = remove_tags(newString)
    newString = re.sub(r'http\S+', '', newString)
    # newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    # newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    # newString = re.sub("[^a-zA-Z.]", " ", newString) # 不保留數字
    newString = re.sub("[^0-9a-zA-Z.]", " ", newString)   # 保留數字  
    tokens = [w for w in newString.split() if w not in alphbet_stopword]    

    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    newString = (" ".join(long_words)).strip()    
    
    newString_sents = []
    text_keywords = []
    for sent in nltk.sent_tokenize(newString):
        sent = " ".join([token for token in bert_tokenizer.tokenize(sent)])
        sent = sent.replace(" ##","")

        lemm_text = lemm_sent_process5(sent)
        if len(lemm_text.split(" "))<3:  continue
        lemm_text = squeeze2(lemm_text)
        if lemm_text[-1] != ".": lemm_text = lemm_text+ " ." # 強制加上dot
        newString_sents.append(lemm_text)
        text_keywords2 = PF_rule_POS(lemm_text).run()
        text_keywords.extend(text_keywords2)
    
    # newString_sents = [line for line in newString_sents if len(line) > 5]
    newString = " \n".join(newString_sents) # 句語句之間分隔
    newString = squeeze2(newString)
    newString = newString.replace(" . . "," . ")
    # print(newString)
    return newString,text_keywords,newString_sents

def summary_cleaner(text):
    # print(text)
    text = squeeze3(text)
    text = remove_word5(text)
    newString = re.sub('"','', text)
    # newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)

    newString = " ".join(bert_tokenizer.tokenize(newString))
    newString = newString.replace(" ##","")
    tokens=newString.split()
    newString=''
    for i in tokens:
        if len(i)>1:                                 
            newString=newString+i+' '     
    # print(newString)
    newString = lemm_sent_process5(newString,summary=True) 
    # print(newString)
    newString = squeeze2(newString)    
    # print('------------------------------------')
    return newString

def lemm_keyword(text):
    text_keywords = []
    lemm_sents = []
    # nt_chars = '[?!，。！.~]+'
    nt_chars = '[!;，。?、…？！.~]+'
    for i,line in enumerate(text):
        # print(line)       
        lemm_text = lemm_sent_process4(line, remove_stopwords=False, summary=False, mode="spacy",withdot=False)       
        # print(lemm_text)
        lemm_text = lemm_text.replace("\n",' ')
        text_keywords2 = PF_rule_POS(lemm_text).run()
        if i < len(text) - 1 : lemm_text = lemm_text + " ."
        lemm_sents.append(lemm_text)
        text_keywords.extend(text_keywords2)
    last_sent = lemm_sents[-1]
    s = re.findall(nt_chars,last_sent[-10:])
    if len(s) != 0:     
        last_sent_sp = re.sub(nt_chars, "", last_sent[-10:])  
        last_sent_sp = last_sent_sp + "."  
        last_sent = last_sent[:-10] + last_sent_sp
        # last_sent = squeeze2(last_sent)
        lemm_sents[-1] = last_sent        
    # lemm_sents = [sent for sent in lemm_sents if len(sent) > 10]
    lemm_sents = [squeeze2(sent) for sent in lemm_sents ]
    lemm_article = " ".join(lemm_sents)

    pattern = re.compile(r"(\d[\s][.][\s]\d)")  # |  ([\(](\w)+[\)])
    dot_numbers = pattern.findall(lemm_article)
    for dot_number in dot_numbers:
        lemm_article = lemm_article.replace(dot_number,dot_number.replace(" ",""))

    lemm_article = squeeze2(lemm_article)
    lemm_article = lemm_article.replace(" s ",' is ')
    return lemm_sents, text_keywords, lemm_article

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

# summ = process(summ)
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
            if mode == 'mixCat' :
                if len(summary.split(" ")) < 6: continue
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

def make_review(df):
    corpus_path = '%s/corpus.txt'%(folder) 
    corpus = open(corpus_path,'w',encoding='utf-8')

    # total_keywords = set()
    feature_counter = Counter()
    docCount = len(df)            
    asin_list, review_list, overall_list, vote_list, summary_list, review_ID_list , cheat_num_list = [] , [] , [] , [] , [] , [] , []
    lemm_review_len_list , lemm_summary_len_list = [] , [] 
    # bert_review_len_list , bert_summary_len_list = [] , [] 
    cheat_list = []


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
                # print('review1 : \n',review)
                lemm_review,rev_keywords,sents = text_cleaner(review)
                # print('review2 : \n',lemm_review)
                feature_counter.update(rev_keywords)
                rev_token_set = set(lemm_review.split(" "))
                lemm_review_len = len(lemm_review.split(" "))
                # bert_review_len = len(bert_tokenizer.tokenize(lemm_review))
                [corpus.write(sent + "\n") for sent in sents]
                # print(rev_keywords)
            except Exception as e :
                print(e)  
            # ---------------------------------------------------------------------------------------------
            try:              
                # print('summary1 : \n',summary)
                lemm_summary = summary_cleaner(summary)
                # print('summary2 : \n',lemm_summary) 
                lemm_summary_len = len(lemm_summary.split(" "))
                temp_summary = " ".join([t for t in lemm_summary.replace('<s> ','').replace(" </s>",'').split() if t != ''])
                bert_summary_len = len(bert_tokenizer.tokenize(temp_summary))
                corpus.write(temp_summary)    
                # print('summary3 : \n',temp_summary)        
                summ_token_set = set(lemm_summary.split(" "))
                # lemm_summary = '<s> ' + lemm_summary + " </s> "            
                # print('summary2 : \n',lemm_summary)                      
            except Exception as e :
                continue           
            # ----------------------------------------------------------------------------------------------        
            # # cheat = rev_token_set & summ_token_set & ( total_keywords | set(opinion_lexicon["total-words"]) )
            # cheat = rev_token_set & summ_token_set & (set(opinion_lexicon["total-words"]) )
            # # cheat = rev_token_set & summ_token_set
            # cheat = set([c for c in cheat if c != ''])
            # cheat_num = len(cheat) 
            # ----------------------------------------------------------------------------------------------
            pbar.set_description("%s training-pair " % (folder))
            
            asin_list.append(asin)
            review_list.append(lemm_review)
            overall_list.append(overall)
            vote_list.append(vote)
            summary_list.append(lemm_summary)
            review_ID_list.append(review_ID)
            # cheat_num_list.append(cheat_num) 
            # cheat_list.append(cheat)
            lemm_review_len_list.append(lemm_review_len)
            lemm_summary_len_list.append(lemm_summary_len)   
            # bert_review_len_list.append(bert_review_len)     
            # bert_summary_len_list.append(bert_summary_len)    
            
        
        # for t in threads:
        #     t.join()

        df = pd.DataFrame({"asin":asin_list, "review": review_list, "overall": overall_list, "vote": vote_list,
                            "summary": summary_list , "review_ID": review_ID_list, 
                            # "cheat_num": cheat_num_list, "cheat": cheat_list ,
                            "lemm_review_len": lemm_review_len_list , "lemm_summary_len": lemm_summary_len_list
                            # "bert_review_len": bert_review_len_list , "bert_summary_len": bert_summary_len_list
                            })
        corpus.close()



        important_features = OrderedDict(sorted(feature_counter.items(), key=lambda pair: pair[1], reverse=True))
        important_features = [(word, important_features[word]) for word in important_features if important_features[word] > 0]
        print("features Count : %s" % (len(important_features)))

        fn3 = '%s/review_keywords.txt'%(folder)
        with open(fn3, 'w', encoding="utf-8") as f:
            total_keywords = set()
            for word, v in important_features:
                f.write("%s:%s \n" % (word, v))
                total_keywords.add(word)


        return df


if not os.path.exists(folder):
    os.makedirs(folder)

csv_path = '%s/review.xlsx'%(folder)   
print("check ", os.path.exists(csv_path))
if not os.path.exists(csv_path): 
    orign_key_df = make_review(df)
    orign_key_df.to_excel(csv_path, encoding='utf8')
    print(csv_path + " Write finished")   
else:    
    orign_key_df = pd.read_excel(csv_path)
    print("previous file %s ...."%(csv_path))
print(orign_key_df.columns)


# In[ ]:
# xlsx_path = '%s/review.xlsx'%(folder)  
# orign_key_df = pd.read_excel(xlsx_path)
# print(xlsx_path + " Read finished")
# len(orign_key_df)



# --------------------------------------------- Fo-Bin --------------------------------------------- #
# --------------------------------------------- Fo-Bin --------------------------------------------- #
# --------------------------------------------- Fo-Bin --------------------------------------------- #
# --------------------------------------------- Fo-Bin --------------------------------------------- #
import shutil
# from product import *
# from data_util.product import *

import pandas as pd
import random
import time
import re

import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
import nltk
from summa import keywords as TextRank
from summa.summarizer import summarize

import sys
import shutil
# import tqdm
import random

from copy import deepcopy
# from product import *

VOCAB_SIZE = 50000
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data
# --------------------------------------------- Fo-Bin --------------------------------------------- #
fn = '%s/review_keywords.txt'%(folder)
print('load %s keywords...' % (fn))
review_keywords = set()
with open(fn, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        k, v = line.split(":")
        review_keywords.add(k) 

# csv_path = '%s/review.xlsx'%(folder)    
# print("check ", os.path.exists(csv_path))
# if os.path.exists(csv_path):
#     df = pd.read_excel(csv_path)
#     print("previous file %s ...."%(csv_path))

from matplotlib import pyplot as plt
statistic_path = '%s/statistic'%(folder) 
if not os.path.exists(statistic_path):
    os.makedirs(statistic_path)
df = orign_key_df

with open('%s/statistic/data_info.txt'%(folder),'w') as f:
    max_rev_len = df['lemm_review_len'].max()
    min_rev_len = df['lemm_review_len'].min()
    mean_rev_len = df['lemm_review_len'].mean()
    median_rev_len = df['lemm_review_len'].median()

    f.write('max_rev_len :%s \n'%(max_rev_len))
    f.write('min_rev_len :%s \n'%(min_rev_len))
    f.write('mean_rev_len :%s \n'%(mean_rev_len))
    f.write('median_rev_len :%s \n'%(median_rev_len))
    
    f.write('\n\n\n')
    max_summary_len = df['lemm_summary_len'].max()
    min_summary_len = df['lemm_summary_len'].min()
    mean_summary_len = df['lemm_summary_len'].mean()
    median_summary_len = df['lemm_summary_len'].median()

    f.write('max_summary_len :%s \n'%(max_summary_len))
    f.write('min_summary_len :%s \n'%(min_summary_len))
    f.write('mean_summary_len :%s \n'%(mean_summary_len))
    f.write('median_summary_len :%s \n'%(median_summary_len))   

    df['lemm_review_len'].value_counts().hist()
    plt.savefig('%s/statistic/review_len.png'%(folder))
    # plt.show()
    plt.close()

    df['lemm_summary_len'].value_counts().hist()
    plt.savefig('%s/statistic/summary_len.png'%(folder))
    # plt.show()
    plt.close()

# with open('%s/statistic/bert_data_info.txt'%(folder),'w') as f:
#     max_bert_rev_len = df['bert_review_len'].max()
#     min_bert_rev_len = df['bert_review_len'].min()
#     mean_bert_rev_len = df['bert_review_len'].mean()
#     median_bert_rev_len = df['bert_review_len'].median()

#     f.write('max_bert_rev_len :%s \n'%(max_bert_rev_len))
#     f.write('min_bert_rev_len :%s \n'%(min_bert_rev_len))
#     f.write('mean_bert_rev_len :%s \n'%(mean_bert_rev_len))
#     f.write('median_bert_rev_len :%s \n'%(median_bert_rev_len))
    
#     f.write('\n\n\n')
#     max_bert_summary_len = df['bert_summary_len'].max()
#     min_bert_summary_len = df['bert_summary_len'].min()
#     mean_bert_summary_len = df['bert_summary_len'].mean()
#     median_bert_summary_len = df['bert_summary_len'].median()

#     f.write('max_bert_summary_len :%s \n'%(max_bert_summary_len))
#     f.write('min_bert_summary_len :%s \n'%(min_bert_summary_len))
#     f.write('mean_bert_summary_len :%s \n'%(mean_bert_summary_len))
#     f.write('median_bert_summary_len :%s \n'%(median_bert_summary_len))   

#     df['bert_review_len'].value_counts().hist()
#     plt.savefig('%s/statistic/bert_review_len.png'%(folder))
#     # plt.show()
#     plt.close()

#     df['bert_summary_len'].value_counts().hist()
#     plt.savefig('%s/statistic/bert_summary_len.png'%(folder))
#     # plt.show()
#     plt.close()
print('make %s finished'%(csv_path))

df = df[(df.lemm_summary_len >= 4) ] # 過濾single word summary
df = df[(df.lemm_review_len <= 1000) ] # 過濾single word summary
df = df[(df.lemm_review_len >= 20) ] # 過濾single word summary

df = df.reset_index(drop=True)

csv_path = '%s/pro_review_70per.xlsx'%(folder)   
if not os.path.exists(csv_path):
    with tqdm(total=len(df)) as pbar:
        j = 0
        pro_df = {}
        for idx in range(len(df)):           
            series = df.iloc[idx]
            data_dict = series.to_dict()
            # review_ID , review , summary , cheat_num = \
            # data_dict['review_ID'], data_dict['review'], data_dict['summary'], data_dict['cheat_num'] 

            review_ID , review , summary = \
            data_dict['review_ID'], data_dict['review'], data_dict['summary']

            # lemm_review_len , lemm_summary_len , bert_review_len , bert_summary_len = \
            # data_dict['lemm_review_len'], data_dict['lemm_summary_len'], data_dict['bert_review_len'], data_dict['bert_summary_len'] 

            lemm_review_len , lemm_summary_len  = \
            data_dict['lemm_review_len'], data_dict['lemm_summary_len']


            review = squeeze2(review)
            summary = squeeze2(summary)            

            review = review.replace("\n","")
            # remove_chars = '[0-9]+'
            # review = re.sub(remove_chars, "", review)
            review = " ".join([t for t in review.split(" ") if t != ""]).strip()
            summary = summary.replace("\n","").strip()

            rev_token_set = set(review.split(" "))
            summ_token_set = set(summary.split(" "))
            cheat = rev_token_set & summ_token_set & ( review_keywords | set(opinion_lexicon["total-words"]) )
            if len(cheat) < 3 : continue # => 最佳
            cheat_num = len(cheat)

            overlap = len(rev_token_set & summ_token_set)
            if len(rev_token_set & summ_token_set) > 0.7*(len(summ_token_set)) : continue # => 避免過高重疊
            

            # FOP_keywords
            POS_FOP_keywords , DEP_FOP_keywords = [] , []
            for lemm_sent in review.split(" . "):
                lemm_sent = lemm_sent + " . "
                
                POS_fops = FO_rule_POS(lemm_sent).run()                    
                POS_fops = [(f, o) for f, o in POS_fops if f in review_keywords]
                POS_FOP_keywords = POS_FOP_keywords + POS_fops

                DEP_fops = FOP_rule_Depend(lemm_sent).run()
                DEP_fops = [(f, o) for f, o in DEP_fops if f in review_keywords]
                DEP_FOP_keywords = DEP_FOP_keywords + DEP_fops

            # FOP_keywords
            POS_FOP_keywords = ",".join(["%s %s"%(f, o) for f, o in POS_FOP_keywords])
            DEP_FOP_keywords = ",".join(["%s %s"%(f, o) for f, o in DEP_FOP_keywords])            
            if len(POS_FOP_keywords) == 0: continue
            if len(DEP_FOP_keywords) == 0: continue

            
            # TextRank
            TextRank_keywords , TextRank_summary = [] , []

            for words in TextRank.keywords(review).split('\n'):
                TextRank_keywords.extend(words.split(" "))
            TextRank_keywords = " ".join(TextRank_keywords)
            if len(TextRank_keywords) == 0: continue         

            save_dict = {
                "review": review.strip(),
                "summary": summary.strip(),
                "cheat": cheat,
                "cheat_num": cheat_num,
                'overlap':overlap,
                "lemm_review_len": lemm_review_len,
                "lemm_summary_len": lemm_summary_len,
                # "bert_review_len": bert_review_len,
                # "bert_summary_len": bert_summary_len,
                "POS_FOP_keywords": POS_FOP_keywords,
                "DEP_FOP_keywords": DEP_FOP_keywords,
                "TextRank_keywords": TextRank_keywords
            }
            pro_df[j] = save_dict
            j = j + 1

            pbar.update(1)
            pbar.set_description("%s pro review 70 percent " % (folder))

        pro_df = pd.DataFrame.from_dict(pro_df, orient='index')
        pro_df.to_excel(csv_path, encoding='utf8')
        print(csv_path + " Write finished")   

# # bin write info
# if not os.path.exists('%s/bin'%(folder)):
#     shutil.rmtree('%s/bin'%(folder), ignore_errors=True)

# if not os.path.exists('%s/bin/chunked'%(folder)):
#     os.makedirs('%s/bin/chunked'%(folder))

# makevocab = True
# if makevocab:
#     vocab_counter = collections.Counter()
# filt
csv_path = '%s/pro_review.xlsx'%(folder)   
df = pd.read_excel(csv_path)
print("previous file %s ...."%(csv_path))

df = df.dropna(
    axis=0,     # 0: 对行进行操作; 1: 对列进行操作
    how='any'   # 'any': 只要存在 NaN 就 drop 掉; 'all': 必须全部是 NaN 才 drop 
    )
# condition cheat
cond_statistic_path = '%s/cond_statistic'%(folder) 
if not os.path.exists(cond_statistic_path):
    os.makedirs(cond_statistic_path)

with open('%s/cond_statistic/data_info.txt'%(folder),'w') as f:
    max_rev_len = df['lemm_review_len'].max()
    min_rev_len = df['lemm_review_len'].min()
    mean_rev_len = df['lemm_review_len'].mean()
    median_rev_len = df['lemm_review_len'].median()

    f.write('max_rev_len :%s \n'%(max_rev_len))
    f.write('min_rev_len :%s \n'%(min_rev_len))
    f.write('mean_rev_len :%s \n'%(mean_rev_len))
    f.write('median_rev_len :%s \n'%(median_rev_len))
    
    f.write('\n\n\n')
    max_summary_len = df['lemm_summary_len'].max()
    min_summary_len = df['lemm_summary_len'].min()
    mean_summary_len = df['lemm_summary_len'].mean()
    median_summary_len = df['lemm_summary_len'].median()

    f.write('max_summary_len :%s \n'%(max_summary_len))
    f.write('min_summary_len :%s \n'%(min_summary_len))
    f.write('mean_summary_len :%s \n'%(mean_summary_len))
    f.write('median_summary_len :%s \n'%(median_summary_len))   

    df['lemm_review_len'].value_counts().hist()
    plt.savefig('%s/statistic/review_len.png'%(folder))
    # plt.show()
    plt.close()

    df['lemm_summary_len'].value_counts().hist()
    plt.savefig('%s/statistic/summary_len.png'%(folder))
    # plt.show()
    plt.close()

# with open('%s/cond_statistic/bert_data_info.txt'%(folder),'w') as f:
#     max_bert_rev_len = df['bert_review_len'].max()
#     min_bert_rev_len = df['bert_review_len'].min()
#     mean_bert_rev_len = df['bert_review_len'].mean()
#     median_bert_rev_len = df['bert_review_len'].median()

#     f.write('max_bert_rev_len :%s \n'%(max_bert_rev_len))
#     f.write('min_bert_rev_len :%s \n'%(min_bert_rev_len))
#     f.write('mean_bert_rev_len :%s \n'%(mean_bert_rev_len))
#     f.write('median_bert_rev_len :%s \n'%(median_bert_rev_len))
    
#     f.write('\n\n\n')
#     max_bert_summary_len = df['bert_summary_len'].max()
#     min_bert_summary_len = df['bert_summary_len'].min()
#     mean_bert_summary_len = df['bert_summary_len'].mean()
#     median_bert_summary_len = df['bert_summary_len'].median()

#     f.write('max_bert_summary_len :%s \n'%(max_bert_summary_len))
#     f.write('min_bert_summary_len :%s \n'%(min_bert_summary_len))
#     f.write('mean_bert_summary_len :%s \n'%(mean_bert_summary_len))
#     f.write('median_bert_summary_len :%s \n'%(median_bert_summary_len))   

#     df['bert_review_len'].value_counts().hist()
#     plt.savefig('%s/statistic/bert_review_len.png'%(folder))
#     # plt.show()
#     plt.close()

#     df['bert_summary_len'].value_counts().hist()
#     plt.savefig('%s/statistic/bert_summary_len.png'%(folder))
#     plt.show()
#     plt.close()    

amount = len(df)
print('Total data : %s'%(amount))    
    
# df = df.iloc[:35000]    
# train_file
flit_key_train_df = df.iloc[:int(amount*0.8)]

# test_file
flit_key_test_df = df.iloc[int(amount*0.8)+1:int(amount*0.9)]

# vald_file
flit_key_valid_df = df.iloc[int(amount*0.9)+1:]
sentence_start = "<s>"
sentence_end = "</s>"

import threading
import time

# write bin
def xlsx2bin(set_name,split_df):
    sents = []
    with open("%s/bin/%s.bin"%(folder,set_name), 'wb') as file:
        i = 0
        # threads = []
        # lock = threading.BoundedSemaphore(10) #最多允許10個執行緒同時執行
        with tqdm(total=len(split_df)) as pbar:
            for idx in range(len(split_df)):
                pbar.update(1)
                pbar.set_description("%s write bin file" % (folder))
                series = split_df.iloc[idx]
                data_dict = series.to_dict()
                review , summary , cheat_num = \
                data_dict['review'], data_dict['summary'], data_dict['cheat_num'] 

                POS_FOP_keywords, DEP_FOP_keywords, TextRank_keywords = \
                data_dict['POS_FOP_keywords'], data_dict['DEP_FOP_keywords'], data_dict['TextRank_keywords']


                # Write to tf.Example
                tf_example = example_pb2.Example()
                try:                
                    tf_example.features.feature['review'].bytes_list.value.extend(
                        [tf.compat.as_bytes(review, encoding='utf-8')])

                    tf_example.features.feature['summary'].bytes_list.value.extend(
                        [tf.compat.as_bytes(summary, encoding='utf-8')]) 
                                
                    tf_example.features.feature['POS_FOP_keywords'].bytes_list.value.extend(
                        [tf.compat.as_bytes(POS_FOP_keywords, encoding='utf-8')]) 

                    tf_example.features.feature['DEP_FOP_keywords'].bytes_list.value.extend(
                        [tf.compat.as_bytes(DEP_FOP_keywords, encoding='utf-8')])                

                    tf_example.features.feature['TextRank_keywords'].bytes_list.value.extend(
                        [tf.compat.as_bytes(TextRank_keywords, encoding='utf-8')])                    
                   
                    tf_example_str = tf_example.SerializeToString()
                    str_len = len(tf_example_str)  
                    file.write(struct.pack('q', str_len))
                    file.write(struct.pack('%ds' % str_len, tf_example_str))
                    i = i + 1
                except Exception as e:
                    pass
        # for t in threads:
        #     t.join()
    print("%s %s finished... "%(file.name,i))
    return i
def chunk_file(set_name, chunks_dir):
    # 分割record bin檔(1000為單位) 
    in_file = '%s/bin/%s.bin' % (folder,set_name)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        # chunk_fname = os.path.join('bin', '/%s/%s_%03d.bin' % (chunks_dir,set_name, chunk))  # new chunk
        chunk_fname = '%s/%s/%s_%03d.bin' % (chunks_dir,set_name,set_name, chunk)
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1
def chunk_all(chunks_dir = '%s/bin/chunked'%(folder)):
    # Make a dir to hold the chunks
    
    # Chunk the data
    for set_name in ['train', 'valid', 'test']:
        if not os.path.isdir(os.path.join(chunks_dir,set_name)):
            os.mkdir(os.path.join(chunks_dir,set_name))
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name, chunks_dir)
    print("Saved chunked data in %s" % chunks_dir)
def main_valid():
    #Performing rouge evaluation on 1.9 lakh sentences takes lot of time. So, create mini validation set & test set by borrowing 15k samples each from these 1.9 lakh sentences
    bin_valid_chuncks = os.listdir('%s/bin/chunked/valid/'%(folder))
    bin_valid_chuncks.sort()
    if not os.path.exists('%s/bin/chunked/main_valid'%(folder)):
        os.makedirs('%s/bin/chunked/main_valid'%(folder))
        
    samples = random.sample(set(bin_valid_chuncks[:-1]), 2)      #Exclude last bin file; contains only 9k sentences
    valid_chunk, test_chunk = samples[0], samples[1]
    shutil.copyfile(os.path.join('%s/bin/chunked/valid'%(folder), valid_chunk), os.path.join("%s/bin/chunked/main_valid"%(folder), "valid_00.bin"))
    shutil.copyfile(os.path.join('%s/bin/chunked/valid'%(folder), test_chunk), os.path.join("%s/bin/chunked/main_valid"%(folder), "test_00.bin"))

# make bin
# train_len = xlsx2bin('train',flit_key_train_df)
# test_len = xlsx2bin('test',flit_key_test_df)
# valid_len = xlsx2bin('valid',flit_key_valid_df)

with open("data-info.txt",'w',encoding='utf-8') as f :
    f.write("train : %s\n"%(len(flit_key_train_df)))
    f.write("test : %s\n"%(len(flit_key_test_df)))
    f.write("valid : %s\n"%(len(flit_key_valid_df)))

# chunk_all() 
# main_valid()


# Ready Embedding Corpus
from collections import Counter, OrderedDict
def make_corpus():
    corpus_path = '%s/corpus.txt'%(folder) 
    corpus = []
    embedding_corpus = []
    with open(corpus_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = squeeze2(line)
            tokens = line.replace("\n",'').replace("<s>",'').replace("</s>",'').split(" ")
            # tokens = [token for token in tokens if (token != '' and token.isalpha() and token not in alphbet_stopword)]
            tokens = [token for token in tokens if (token != '' )] # => 最佳
            corpus.append(tokens)
            embedding_corpus.append(tokens)
    print('make corpus finished...')
    return embedding_corpus
embedding_corpus = make_corpus()
# 引入 word2vec
from gensim.models import word2vec
from glob import glob
import sys

import gensim
import torch
import torch.nn as nn
import torchsnooper
import os
import numpy as np

# 引入日志配置
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
vocab_count = 50000

# write vocab to file
if not os.path.exists('%s/Embedding/word2Vec'%(folder)):
    os.makedirs('%s/Embedding/word2Vec'%(folder))

# Embedding/word2Vec
def train_word2Vec(vector_size):
    # sg=1表示採用skip-gram, 預設sg=0 表示採用cbow
    if not os.path.exists("%s/Embedding/word2Vec/word2Vec.%sd.txt"%(folder,vector_size)):    
        w2vec = word2vec.Word2Vec(embedding_corpus, size=vector_size, min_count=1,max_vocab_size=None,iter=100,
                                sorted_vocab=1,max_final_vocab=vocab_count)    

        w2vec.wv.save_word2vec_format('%s/Embedding/word2Vec/word2Vec.%sd.txt'%(folder, vector_size), binary=False)

    #模型讀取方式
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
        '%s/Embedding/word2Vec/word2Vec.%sd.txt'%(folder, vector_size), binary=False, encoding='utf-8')
    wvmodel.most_similar(u"player", topn=10)  
    return wvmodel
def bulid_word2Vec_vocab(wvmodel):
    vocab_file = "%s/Embedding/word2Vec/word.vocab"%(folder)

    if not os.path.exists(vocab_file):
        vocab_count = len(wvmodel.wv.index2entity)    

        print("Writing vocab file...")
        with open(vocab_file, 'w',encoding='utf-8') as writer:
            for word in wvmodel.wv.index2entity[:vocab_count]:
                # print(word, w2vec.wv.vocab[word].count)
                writer.write(word + ' ' + str(wvmodel.wv.vocab[word].count) + '\n') # Output vocab count
        print("Finished writing vocab file")
    else:
        print('Already get word2Vec vocab')
    # from data_util.data import Vocab
    # vocab_size = len(wvmodel.vocab) + 1
    # vocab = Vocab('%s/Embedding/word2Vec/word.vocab'%(folder), vocab_size)

train_word2Vec(vector_size = 300)
wvmodel = train_word2Vec(vector_size = 512)
bulid_word2Vec_vocab(wvmodel)


##------------------------------------------------------
from gensim.models.fasttext import FastText
# write vocab to file
if not os.path.exists('%s/Embedding/FastText'%(folder)):
    os.makedirs('%s/Embedding/FastText'%(folder))

# Embedding/FastText
def train_FastText(vector_size):
    if not os.path.exists("%s/Embedding/FastText/FastText.%sd.txt"%(folder, vector_size)):
        fasttext = FastText(embedding_corpus, size=vector_size, min_count=1,max_vocab_size=None,iter=100,
                                sorted_vocab=1)    

        fasttext.wv.save_word2vec_format('%s/Embedding/FastText/FastText.%sd.txt'%(folder, vector_size), binary=False)

    #模型讀取方式
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
        '%s/Embedding/FastText/FastText.%sd.txt'%(folder, vector_size), binary=False, encoding='utf-8')
    wvmodel.most_similar(u"player", topn=10)  
    return wvmodel
def bulid_FastText_vocab(wvmodel):
    vocab_file = "%s/Embedding/FastText/word.vocab"%(folder)

    if not os.path.exists(vocab_file):
        vocab_count = len(wvmodel.wv.index2entity)    

        print("Writing vocab file...")
        with open(vocab_file, 'w',encoding='utf-8') as writer:
            for word in wvmodel.wv.index2entity[:vocab_count]:
                # print(word, w2vec.wv.vocab[word].count)
                writer.write(word + ' ' + str(wvmodel.wv.vocab[word].count) + '\n') # Output vocab count
        print("Finished writing vocab file")
    else:
        print('Already get FastText vocab')

    # from data_util.data import Vocab
    # vocab_size = len(wvmodel.vocab) + 1
    # vocab = Vocab('%s/Embedding/FastText/word.vocab'%(folder), vocab_size)

train_FastText(vector_size = 300)
wvmodel = train_FastText(vector_size = 512)
bulid_FastText_vocab(wvmodel)


##-----------------------------------------------------

# Embedding/glove    
from glove import Glove
from glove import Corpus
from gensim import corpora

vocab_count = 50000
# write vocab to file
if not os.path.exists('%s/Embedding/glove'%(folder)):
    os.makedirs('%s/Embedding/glove'%(folder))

def create_glove_corpus():
    corpus_path = '%s/Embedding/glove/glove.corpus'%(folder)
    if not os.path.exists(corpus_path):
        corpus = Corpus() # 建立glove corpus物件，並設定matrix scan window大小
        corpus.fit(embedding_corpus, window=10) 

        corpus.fit(embedding_corpus, window=10)
        print('Dict size: %s' % len(corpus.dictionary))
        print('Collocations: %s' % corpus.matrix.nnz)
        corpus.save('%s/Embedding/glove/glove.corpus'%(folder)) # 存字典

    else:
        corpus = Corpus.load('%s/Embedding/glove/glove.corpus'%(folder))
        print('Already get glove corpus')
    return corpus
def train_glove(corpus, vector_size):
    if not os.path.exists("%s/Embedding/glove/glove%s.model"%(folder,vector_size)):
        '''Using Corpus to construct co-occurrence matrix'''        
        glove = Glove(no_components=vector_size, learning_rate=0.05)
        # 建立glove物件，並使用先前建立的co-occurrence matrix去建立embedding
        glove.fit(corpus.matrix, epochs=100,
                no_threads=10, verbose=True)
        # 將glove物件加入dictionary，使word vectors與其匹配
        glove.add_dictionary(corpus.dictionary)        
        glove.save('%s/Embedding/glove/glove%s.model'%(folder,vector_size)) # 存模型
    else:
        glove = Glove.load('%s/Embedding/glove/glove%s.model'%(folder,vector_size)) 
    return glove
def bulid_glove_vocab(glove_model):
    # write vocab to file
    vocab_file = "%s/Embedding/glove/word.vocab"%(folder)
    if not os.path.exists(vocab_file):
        vocab_count = len(glove_model.dictionary.keys()) 
        vocab_count = 0
        print("Writing vocab file...")
        with open(vocab_file, 'w',encoding='utf-8') as writer:
            for word, word_id in glove_model.dictionary.items():
                writer.write(word + ' ' + str(word_id) + '\n') # Output vocab count                    
        print("Finished writing vocab file %s" %(vocab_count))   
    else:
        print('Already get glove vocab')

corpus = create_glove_corpus()
train_glove(corpus,vector_size=300)
glove = train_glove(corpus,vector_size=512)
bulid_glove_vocab(glove)

#透過gensim以text_data建立字典
# dictionary = corpora.Dictionary(embedding_corpus)
# dictionary.save('%s/Embedding/glove/dictionary.gensim'%(folder))


# glove = Glove.load('%s/Embedding/glove/glove.model'%(folder))
# corpus = Corpus.load('%s/Embedding/glove/glove.corpus'%(folder))
# dictionary = gensim.corpora.Dictionary.load('%s/Embedding/glove/dictionary.gensim'%(folder))

