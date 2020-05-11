
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

from extract_key import noun_adj, extract_POS, extract_DEP

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
    # mongoObj = Mix12()
    # mongoObj = Mixbig_5()
    '''compare'''
    # mongoObj = Mix6()
    # mongoObj = Mixbig_Elect_30()
    # mongoObj = Mixbig_Books_3()
    mongoObj = Pure_kitchen()
    # mongoObj = Pure_Cloth()
    
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
# ----------------------------------------------------
csv_path = '%s/pro_review.xlsx'%(folder)   
print("check pro_review", os.path.exists(csv_path))
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
            # POS_FOP_keywords , DEP_FOP_keywords = [] , []
            # for lemm_sent in review.split(" . "):
            #     lemm_sent = lemm_sent + " . "
                
            #     POS_fops = FO_rule_POS(lemm_sent).run()                    
            #     POS_fops = [(f, o) for f, o in POS_fops if f in review_keywords]
            #     POS_FOP_keywords = POS_FOP_keywords + POS_fops

            #     DEP_fops = FOP_rule_Depend(lemm_sent).run()
            #     DEP_fops = [(f, o) for f, o in DEP_fops if f in review_keywords]
            #     DEP_FOP_keywords = DEP_FOP_keywords + DEP_fops

            # # FOP_keywords
            # POS_FOP_keywords = ",".join(["%s %s"%(f, o) for f, o in POS_FOP_keywords])
            # DEP_FOP_keywords = ",".join(["%s %s"%(f, o) for f, o in DEP_FOP_keywords])            
            # if len(POS_FOP_keywords) == 0: continue
            # if len(DEP_FOP_keywords) == 0: continue

            POS_keys , DEP_keys, Noun_adj_keys = [] , [] , []
            for sent in review.split(" . "):
                # pos
                POS_keys = POS_keys + extract_POS(sent).run()[0]
                # dep
                DEP_keys = DEP_keys + extract_DEP(sent).run()[0]
                # noun_adj
                Noun_adj_keys = Noun_adj_keys + noun_adj(sent)[0]

            
            # TextRank
            TextRank_keywords = []

            for words in TextRank.keywords(review).split('\n'):
                TextRank_keywords.extend(words.split(" "))

            # print(TextRank_keywords)
            # TextRank_keywords = " ".join(TextRank_keywords)
            if (len(POS_keys) == 0) or (len(DEP_keys) == 0) \
                or (len(Noun_adj_keys) == 0) or (len(TextRank_keywords) == 0): continue         

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
                "POS_keys": POS_keys,
                "DEP_keys": DEP_keys,
                "Noun_adj_keys": Noun_adj_keys,
                "TextRank_keys": TextRank_keywords
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
print("previous pro file %s ...."%(csv_path))

# ----------------------------------------------------

# ----------------------------------------------------
df = df.dropna(
    axis=0,     # 0: 对行进行操作; 1: 对列进行操作
    how='any'   # 'any': 只要存在 NaN 就 drop 掉; 'all': 必须全部是 NaN 才 drop 
    )
# condition cheat
cond_statistic_path = '%s/cond_statistic'%(folder) 
if not os.path.exists(cond_statistic_path):
    os.makedirs(cond_statistic_path)

with open('%s/cond_statistic/data_info.txt'%(folder),'w') as f:
    max_rev_len = df['review_len'].max()
    min_rev_len = df['review_len'].min()
    mean_rev_len = df['review_len'].mean()
    median_rev_len = df['review_len'].median()

    f.write('max_rev_len :%s \n'%(max_rev_len))
    f.write('min_rev_len :%s \n'%(min_rev_len))
    f.write('mean_rev_len :%s \n'%(mean_rev_len))
    f.write('median_rev_len :%s \n'%(median_rev_len))
    
    f.write('\n\n\n')
    max_summary_len = df['summary_len'].max()
    min_summary_len = df['summary_len'].min()
    mean_summary_len = df['summary_len'].mean()
    median_summary_len = df['summary_len'].median()

    f.write('max_summary_len :%s \n'%(max_summary_len))
    f.write('min_summary_len :%s \n'%(min_summary_len))
    f.write('mean_summary_len :%s \n'%(mean_summary_len))
    f.write('median_summary_len :%s \n'%(median_summary_len))   

    df['review_len'].value_counts().hist()
    plt.savefig('%s/statistic/review_len.png'%(folder))
    # plt.show()
    plt.close()

    df['summary_len'].value_counts().hist()
    plt.savefig('%s/statistic/summary_len.png'%(folder))
    # plt.show()
    plt.close() 

amount = len(df)
print('Total data : %s'%(amount))  
    

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

wvmodel = train_word2Vec(vector_size = 300)
bulid_word2Vec_vocab(wvmodel)
train_word2Vec(vector_size = 512)
# train_word2Vec(vector_size = 768)



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

wvmodel = train_FastText(vector_size = 300)
bulid_FastText_vocab(wvmodel)
train_FastText(vector_size = 512)
# wvmodel = train_FastText(vector_size = 768)



##-----------------------------------------------------

# Embedding/glove    
from glove import Glove
from glove import Corpus
from gensim import corpora

vocab_count = 80000
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
glove = train_glove(corpus,vector_size=300)
bulid_glove_vocab(glove)
train_glove(corpus,vector_size=512)
# glove = train_glove(corpus,vector_size=768)


#透過gensim以text_data建立字典
# dictionary = corpora.Dictionary(embedding_corpus)
# dictionary.save('%s/Embedding/glove/dictionary.gensim'%(folder))


# glove = Glove.load('%s/Embedding/glove/glove.model'%(folder))
# corpus = Corpus.load('%s/Embedding/glove/glove.corpus'%(folder))
# dictionary = gensim.corpora.Dictionary.load('%s/Embedding/glove/dictionary.gensim'%(folder))

