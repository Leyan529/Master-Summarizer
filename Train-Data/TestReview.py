
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
    # mongoObj = Mix6()
    # mongoObj = Mix12()
    mongoObj = OtherTest()
    
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
                if len(review.split(" ")) < 50: continue
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
print("check orign_review", os.path.exists(csv_path))
if os.path.exists(csv_path):
    df = pd.read_excel(csv_path)
    print("previous file %s ...."%(csv_path)) 
else:
    df = make_orign_rev_xlsx()
    df.to_excel(csv_path, encoding='utf8')
    print(csv_path + " Write finished") 

from collections import Counter, OrderedDict
import pandas as pd

def make_review(df):
    corpus_path = '%s/corpus.txt'%(folder) 
    corpus = open(corpus_path,'w',encoding='utf-8')

    # total_keywords = set()
    feature_counter = Counter()
    docCount = len(df)            
    asin_list, review_list, overall_list, vote_list, summary_list, review_ID_list , cheat_num_list = [] , [] , [] , [] , [] , [] , []
    lemm_review_len_list , lemm_summary_len_list = [] , [] 
    # bert_review_len_list , bert_summary_len_list = [] , [] 
    # cheat_list = []


    with tqdm(total=docCount) as pbar:
        # threads = []
        # lock = threading.BoundedSemaphore(5) #最多允許10個執行緒同時執行
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
                # rev_token_set = set(lemm_review.split(" "))
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
                # bert_summary_len = len(bert_tokenizer.tokenize(temp_summary))
                corpus.write(temp_summary)    
                # print('summary3 : \n',temp_summary)        
                # summ_token_set = set(lemm_summary.split(" "))
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
print("check review", os.path.exists(csv_path))
if not os.path.exists(csv_path): 
    orign_key_df = make_review(df)
    orign_key_df.to_excel(csv_path, encoding='utf8')
    print(csv_path + " Write finished")   
else:    
    orign_key_df = pd.read_excel(csv_path)
    print("previous file %s ...."%(csv_path))
print(orign_key_df.columns)


# --------------------------------------------- Fo-Bin --------------------------------------------- #
import shutil

import pandas as pd
import random
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

print('make %s finished'%(csv_path))

df = df[(df.lemm_summary_len >= 4) ] # 過濾single word summary
df = df[(df.lemm_review_len <= 1000) ] # 過濾single word summary
df = df[(df.lemm_review_len >= 50) ] # 過濾single word summary

df = df.reset_index(drop=True)

csv_path = '%s/pro_review.xlsx'%(folder)   
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

            # lemm_review_len , lemm_summary_len  = \
            # data_dict['lemm_review_len'], data_dict['lemm_summary_len']


            review = squeeze2(review)
            summary = squeeze2(summary)            

            review = review.replace("\n","")
            # remove_chars = '[0-9]+'
            # review = re.sub(remove_chars, "", review)
            review = " ".join([t for t in review.split(" ") if t != ""]).strip()
            summary = summary.replace("\n","").strip()

            rev_tokens, summ_tokens = review.split(" "), summary.split(" ")

            rev_token_set = set(rev_tokens)
            summ_token_set = set(summ_tokens)
            cheat = rev_token_set & summ_token_set & ( review_keywords | set(opinion_lexicon["total-words"]) )
            if len(cheat) < 3 : continue # => 最佳
            cheat_num = len(cheat)

            overlap = len(rev_token_set & summ_token_set)
            # if len(rev_token_set & summ_token_set) > 0.7*(len(summ_token_set)) : continue # => 避免過高重疊
            

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
                'review_len':len(rev_tokens),
                'summary_len':len(summ_tokens),
                # "lemm_review_len": lemm_review_len,
                # "lemm_summary_len": lemm_summary_len,
                # "bert_review_len": bert_review_len,
                # "bert_summary_len": bert_summary_len,
                "POS_FOP_keywords": POS_FOP_keywords,
                "DEP_FOP_keywords": DEP_FOP_keywords,
                "TextRank_keywords": TextRank_keywords
            }
            pro_df[j] = save_dict
            j = j + 1

            pbar.update(1)
            pbar.set_description("%s pro review" % (folder))

        pro_df = pd.DataFrame.from_dict(pro_df, orient='index')
        pro_df.to_excel(csv_path, encoding='utf8')
        print(csv_path + " Write finished")   

csv_path = '%s/pro_review.xlsx'%(folder)   
df = pd.read_excel(csv_path)
print("previous file %s ...."%(csv_path))