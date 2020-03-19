#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
# from product import *
from data_util.product import *

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

import sys
import shutil
import tqdm
import random

from copy import deepcopy
# from product import *

VOCAB_SIZE = 50000
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data


# # Key word Attention DataSet 讀取

# In[2]:


# from data_util.config import *
# _, category1, category2, _ = DVD_Player().getAttr()
_,category1,category2,_ = Cameras().getAttr()
# _,category1,category2,_ = Cell_Phones().getAttr()
# _,category1,category2,_ = GPS().getAttr()
# _,category1,category2,_ = Keyboards().getAttr()
# category1,category2 = config.category1 , config.category2

xlsx_path = "XLSX/category/%s_%s_key.xlsx"%(category1,category2)
# df.to_csv(csv_path) #默认dt是DataFrame的一个实例，参数解释如下
# key_train_df.to_excel(csv_path, encoding='utf8')
orign_key_df = pd.read_excel(xlsx_path)
print(xlsx_path + " Read finished")
len(orign_key_df)

orign_key_df['bert_review'] = '' 
orign_key_df['bert_summary'] = ''
orign_key_df['bert_review_len'] = 0
orign_key_df['bert_summary_len'] = 0

orign_key_df.head()


# # Key word load

# In[3]:


fn = 'FOP-View/%s_%s_keywords2.txt' % (category1, category2)
print('load %s keywords...' % (fn))
total_keywords = set()
with open(fn, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        k, v = line.split(":")
        total_keywords.add(k)


# # Total Opinion

# In[4]:


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


# In[5]:


# tokenizer 裡頭的字典資訊


# In[6]:


from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
vocab = tokenizer.vocab # word_to_id
print("字典大小：", len(vocab))


# # Bert-Summary 資料清理

# In[7]:


import spacy
from collections import Counter
import en_core_web_sm
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

nlp = en_core_web_sm.load()

def isnumber(aString):
    try:
        float(aString)
        return True
    except:
        return False
    
def create_custom_tokenizer(nlp):
    prefix_re = re.compile(r'[0-9]\.')
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search)

nlp.tokenizer = create_custom_tokenizer(nlp)

alphbet_stopword = ['b','c','d','e','f','g','h','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


# In[8]:


def compose_summary(x):
    x = x.replace("\n"," ")
    #x = x.replace("\n","").replace("</s>","").replace("<s>","")
    #x = "<s>" + x + "</s>"  
    tokens = [str(token) for token in x.split(" ") if (" " not in str(token)) and                    #(str(token).isalpha()) and \
                  (len(str(token)) > 1)   ]
    # print(tokens)
    #return " ".join(tokens),tokens
    
    newtokens = []
    for token in tokens:
        if (isnumber(token) or len(token) == 1 or token == ".") and (token not in alphbet_stopword):
            newtokens.append(token)
        else:
            token = token.replace("."," . ")
            sub_tokens = token.split(" ")
            sub_tokens = [t for t in sub_tokens if t != "" and t not in alphbet_stopword]
            newtokens.extend(sub_tokens)
    
#     dot_tokens = [token for token in newtokens if (token[0] == "." or token[-1] == ".") and (len(token)>1)]
#     if len(dot_tokens) > 0 :print(dot_tokens)
        
    return " ".join(newtokens).replace(' . . ',' . '),newtokens
    

def bert_compose_summary(newtokens):
#     x = x.replace("\n","").replace("</s>","").replace("<s>","")
#     x = "<s>" + x + "</s>"  
#     tokens = [str(token) for token in nlp(x) if (" " not in str(token)) and \
#                   (str(token).isalpha()) and \
#                   (len(str(token)) > 1)   ]
    
#     newtokens = []
#     for token in tokens:
#         if (isnumber(token) or len(token) == 1 or token == ".") and (token not in alphbet_stopword):
#             newtokens.append(token)
#         else:
#             token = token.replace("."," . ")
#             sub_tokens = token.split(" ")
#             sub_tokens = [t for t in sub_tokens if t != "" and t not in alphbet_stopword]
#             newtokens.extend(sub_tokens)

#     dot_tokens = [t for t in newtokens if ("." in t) and (len(t) > 1) ]
#     if len(dot_tokens) > 0 :print(dot_tokens)
    newtokens = [t for t in newtokens if t not in ["<s>","</s>"]]
    newtokens = ['[CLS]'] + tokenizer.tokenize(" ".join(newtokens)) + ['[SEP]']
    return " ".join(newtokens)

'''
def calc_summary_len(x):
#     tokens = [token for token in nlp(x)]
#     print(tokens)
#     print([len(t) for t in tokens])
#     return len(tokens)
    return len(x.split(" "))

nlp.tokenizer = create_custom_tokenizer(nlp)

orign_key_df['lemm_summary'] = orign_key_df['lemm_summary'].apply(compose_summary)
orign_key_df['lemm_summary_len'] = orign_key_df['lemm_summary'].apply(calc_summary_len)


amount = len(orign_key_df)
print('Total data : %s'%(amount))

orign_key_df.head()
'''
orign_key_df.head()


# In[9]:


from tqdm import tqdm
# 非符號alpha word重疊數
with tqdm(total=len(orign_key_df)) as pbar:
    for i ,row in orign_key_df.iterrows():        
        pbar.update(1)
        pbar.set_description("row %s " % (i))

        lemm_summary,newtokens = compose_summary(row['lemm_summary'])
        # bert_summary = bert_compose_summary(newtokens)
        
        orign_key_df.loc[i,'lemm_summary'] = lemm_summary
        # orign_key_df.loc[i,'bert_summary'] = bert_summary
        
        orign_key_df.loc[i,'lemm_summary_len'] = len(lemm_summary.split(" "))       
        # orign_key_df.loc[i,'bert_summary_len'] = len(bert_summary.split(" "))

        
        
amount = len(orign_key_df)
print('Total data : %s'%(amount))
orign_key_df.head()


# # Bert-review 多句合併

# In[10]:


from copy import deepcopy
def compose_review(x):
    x = eval(x)
    x = "\n".join(x)
    x = x.replace(".\n"," . ").replace("\n."," . ").replace("\n"," ")
#     x = x.replace("\n"," ")    
    tokens = [str(token) for token in x.split(" ") if (" " not in str(token))and (str(token) == '.' or str(token).isalpha())]
#    tokens = [str(token) for token in x.split(" ") if (" " not in str(token)) ]

    #return " ".join(tokens)
    
    newtokens = []
    for token in tokens:
#         if (len(token) == 1 or token == "."):
        if (isnumber(token) or len(token) == 1 or token == ".")and (token not in alphbet_stopword):
            newtokens.append(token)
#             if (token not in ['a','i']) and (token != "."): print(token)
        else:
#             token = token.replace("."," . ")
            token = token.replace("."," . ")
            sub_tokens = token.split(" ")
            sub_tokens = [t for t in sub_tokens if t != "" and t not in alphbet_stopword]
            if len(sub_tokens) == 0: continue
            newtokens.extend(sub_tokens)

    dot_tokens = [token for token in newtokens if (token[0] == "." or token[-1] == ".") and (len(token)>1)]
    # if len(dot_tokens) > 0 :print(dot_tokens)
    
    return " ".join(newtokens).replace(' . . ',' . ')



'''
def calc_review_len(x):
#     tokens = [token for token in nlp(x)]
#     print(tokens)
#     print([len(t) for t in tokens])
#     return len(tokens)
    return len(x.split(" "))
key_df = deepcopy(orign_key_df)
key_df['lemm_review'] = key_df['lemm_review'].apply(compose_review)
key_df['lemm_review_len'] = key_df['lemm_review'].apply(calc_review_len)
key_df.head()

'''
def bert_compose_review(x):
    x = eval(x)
    review_sents = deepcopy(x)
    total_tokens = []
    for sent in review_sents:
        sent = sent.replace('\n','[SEP]')
        tokens = [str(token) for token in sent.split(" ") if (" " not in str(token)) ]
        newtokens = []
        for token in tokens:
            if (isnumber(token) or len(token) == 1 or token == ".")and (token not in alphbet_stopword):
                newtokens.append(token)
            else:
                token = token.replace("."," . ")
                sub_tokens = token.split(" ")
                sub_tokens = [t for t in sub_tokens if t != "" and t not in alphbet_stopword]
                newtokens.extend(sub_tokens)
        newtokens = newtokens + ['[SEP]']        
        total_tokens.extend(newtokens)
    total_tokens = ['[CLS]'] + tokenizer.tokenize(" ".join(total_tokens))    
    return " ".join(total_tokens)

'''
# def compose_review(bert_review):
#     return bert_review.replace('[CLS] ','').replace('[SEP] ','')
'''
''''''


# In[11]:


orign_key_df.head()


# In[12]:


key_df = deepcopy(orign_key_df)

from tqdm import tqdm
# 非符號alpha word重疊數
with tqdm(total=len(key_df)) as pbar:
    for i ,row in key_df.iterrows():  
        try:
            lemm_review = compose_review(row['lemm_review'])
            # bert_review = bert_compose_review(row['lemm_review'])

            # key_df.loc[i,'bert_review'] = bert_review
            key_df.loc[i,'lemm_review'] = lemm_review
            
            # key_df.loc[i,'bert_review_len'] = len(bert_review.split(" "))
            key_df.loc[i,'lemm_review_len'] = len(lemm_review.split(" "))
        except Exception as e:
            pass
            # key_df.loc[i,'bert_review_len'] = 0
            # key_df.loc[i,'lemm_review_len'] = 0     
        
        pbar.set_description("row %s " % (i))
        pbar.update(1)

key_df = key_df[(key_df.lemm_review_len > 0) ] # 過濾 lemm_review_len = 0

amount = len(key_df)
print('Total data : %s'%(amount))

key_df.head()


# In[13]:


# key_df.loc[66152]['lemm_review']


# # 過濾不合適的訓練資料

# In[14]:


def to_words(text):
    keywords = set()
    for words in text.split(","):
        for word in words.split(" "):
            keywords.add(word)
    keywords = " ".join(keywords)
    return keywords

def calc_keyword_num(x):
    return len(x.split(" "))

# and(key_df.lemm_review_len>20)
flit_key_df = key_df[(key_df.lemm_summary_len>=4) ] # 過濾single word summary
flit_key_df = flit_key_df[(flit_key_df.lemm_review_len <= 1000) ] # 過濾single word summary
flit_key_df = flit_key_df[(flit_key_df.lemm_review_len >= 50) ] # 過濾single word summary

flit_key_df = flit_key_df.dropna(
    axis=0,     # 0: 对行进行操作; 1: 对列进行操作
    how='any'   # 'any': 只要存在 NaN 就 drop 掉; 'all': 必须全部是 NaN 才 drop 
    )


# # FOP_keywords 資料整理

# In[15]:


flit_key_df['FOP_keywords'] = flit_key_df['total_keyword']
flit_key_df['FOP_keywords'] = flit_key_df['FOP_keywords'].apply(to_words)
flit_key_df['FOP_keywords_num'] = flit_key_df['FOP_keywords'].apply(calc_keyword_num)
flit_key_df = flit_key_df[(flit_key_df.FOP_keywords_num>=2) ] # 過濾single word summary
flit_key_df.head()


# # Cheat Processing

# In[16]:


flit_key_df['Cheat'] = False 

# flit_key_df.head()
from tqdm import tqdm
# 非符號alpha word重疊數
with tqdm(total=len(flit_key_df)) as pbar:
    for i ,row in flit_key_df.iterrows():
        rev_tokens = set(row['lemm_review'].split(" "))
        summ_tokens = set(row['lemm_summary'].split(" "))
        key_words = rev_tokens & summ_tokens & (total_keywords| set(opinion_lexicon["total-words"]))
        if len(key_words) > 2: 
            flit_key_df.loc[i,'Cheat'] = True
        pbar.update(1)
    
flit_key_df = flit_key_df[(flit_key_df.Cheat == True) ] # 過濾single word summary
amount = len(flit_key_df)
print('Total data : %s'%(amount))
flit_key_df.head()


# # TextRank_keywords 資料整理

# In[17]:


from summa import keywords as TextRank
from summa.summarizer import summarize
def textrank_keys(text):
    keywords1 = list()
    for words in TextRank.keywords(text).split('\n'):
        keywords1.extend(words.split(" "))
    keywords1 = set(keywords1)    
    
    return " ".join(list(keywords1))

def textrank_summ_keys(text): 
    keywords2 = list()
    for words in summarize(text, words=8).split('\n'):
        keywords2.extend(words.split(" "))
    keywords2 = set(keywords2)
    
    return " ".join(list(keywords2))

flit_key_df.loc[:,'TextRank_keywords'] = ''
flit_key_df.loc[:,'TextRank_summary'] = ''
# flit_key_df.head()


# In[18]:


from tqdm import tqdm
with tqdm(total=len(flit_key_df)) as pbar:
    for i ,row in flit_key_df.iterrows():
        TextRank_keywords = textrank_keys(row['lemm_review'])
        TextRank_summary = textrank_summ_keys(row['lemm_review'])  
#         num = calc_keyword_num(TextRank_keywords)
        flit_key_df.loc[i,'TextRank_keywords'] = TextRank_keywords
        flit_key_df.loc[i,'TextRank_summary'] = TextRank_summary
        # print(TextRank_keywords);print('****')
        # print(TextRank_summary);print('----------------------')
#         flit_key_df.loc[i,'TextRank_keywords_num'] = num
        pbar.update(1)
        
flit_key_df.head()


# In[19]:


from matplotlib import pyplot as plt
if not os.path.exists('XLSX/statistic'):
    os.makedirs('XLSX/statistic')
with open('XLSX/statistic/%s_%s_info.txt'%(category1,category2),'w') as f:
    max_rev_len = flit_key_df['lemm_review_len'].max()
    min_rev_len = flit_key_df['lemm_review_len'].min()
    mean_rev_len = flit_key_df['lemm_review_len'].mean()
    median_rev_len = flit_key_df['lemm_review_len'].median()

    f.write('max_rev_len :%s \n'%(max_rev_len))
    f.write('min_rev_len :%s \n'%(min_rev_len))
    f.write('mean_rev_len :%s \n'%(mean_rev_len))
    f.write('median_rev_len :%s \n'%(median_rev_len))
    
    f.write('\n\n\n')
    max_summary_len = flit_key_df['lemm_summary_len'].max()
    min_summary_len = flit_key_df['lemm_summary_len'].min()
    mean_summary_len = flit_key_df['lemm_summary_len'].mean()
    median_summary_len = flit_key_df['lemm_summary_len'].median()

    f.write('max_summary_len :%s \n'%(max_summary_len))
    f.write('min_summary_len :%s \n'%(min_summary_len))
    f.write('mean_summary_len :%s \n'%(mean_summary_len))
    f.write('median_summary_len :%s \n'%(median_summary_len))
    
    f.write('\n\n\n')
    max_FOP_keywords_num = flit_key_df['FOP_keywords_num'].max()
    min_FOP_keywords_num = flit_key_df['FOP_keywords_num'].min()
    mean_FOP_keywords_num = flit_key_df['FOP_keywords_num'].mean()
    median_FOP_keywords_num = flit_key_df['FOP_keywords_num'].median()

    f.write('max_FOP_keywords_num :%s \n'%(max_FOP_keywords_num))
    f.write('min_FOP_keywords_num :%s \n'%(min_FOP_keywords_num))
    f.write('mean_FOP_keywords_num :%s \n'%(mean_FOP_keywords_num))
    f.write('median_FOP_keywords_num :%s \n'%(median_FOP_keywords_num))
    
    f.write('\n\n\n')
#     max_TextRank_keywords_num = flit_key_df['TextRank_keywords_num'].max()
#     min_TextRank_keywords_num = flit_key_df['TextRank_keywords_num'].min()
#     mean_TextRank_keywords_num = flit_key_df['TextRank_keywords_num'].mean()
#     median_TextRank_keywords_num = flit_key_df['TextRank_keywords_num'].median()

#     f.write('max_TextRank_keywords_num :%s \n'%(max_TextRank_keywords_num))
#     f.write('min_TextRank_keywords_num :%s \n'%(min_TextRank_keywords_num))
#     f.write('mean_TextRank_keywords_num :%s \n'%(mean_TextRank_keywords_num))
#     f.write('median_TextRank_keywords_num :%s \n'%(median_TextRank_keywords_num))

    '''
    f.write('\n\n\n')    
    max_bert_rev_len = flit_key_df['bert_review_len'].max()
    min_bert_rev_len = flit_key_df['bert_review_len'].min()
    mean_bert_rev_len = flit_key_df['bert_review_len'].mean()
    median_bert_rev_len = flit_key_df['bert_review_len'].median()

    f.write('max_bert_rev_len :%s \n'%(max_bert_rev_len))
    f.write('min_bert_rev_len :%s \n'%(min_bert_rev_len))
    f.write('mean_bert_rev_len :%s \n'%(mean_bert_rev_len))
    f.write('median_bert_rev_len :%s \n'%(median_bert_rev_len))
    
    f.write('\n\n\n')
    max_bert_summary_len = flit_key_df['bert_summary_len'].max()
    min_bert_summary_len = flit_key_df['bert_summary_len'].min()
    mean_bert_summary_len = flit_key_df['bert_summary_len'].mean()
    median_bert_summary_len = flit_key_df['bert_summary_len'].median()

    f.write('max_bert_summary_len :%s \n'%(max_bert_summary_len))
    f.write('min_bert_summary_len :%s \n'%(min_bert_summary_len))
    f.write('mean_bert_summary_len :%s \n'%(mean_bert_summary_len))
    f.write('median_bert_summary_len :%s \n'%(median_bert_summary_len))
    '''

    

# plt.xlim(xmax = mean_rev_len)
# plt.ylim(ymax = flit_key_df['lemm_review_len'].value_counts().max())

flit_key_df['lemm_review_len'].value_counts().hist()
plt.savefig('XLSX/statistic/review_len_%s_%s.png'%(category1,category2))
# plt.show()
plt.close()

# plt.xlim(xmax = max_summary_len)
# plt.ylim(ymax = flit_key_df['lemm_summary_len'].value_counts().max())
flit_key_df['lemm_summary_len'].value_counts().hist()
plt.savefig('XLSX/statistic/summary_len_%s_%s.png'%(category1,category2))
# plt.show()
plt.close()

# plt.xlim(xmax = mean_keyword_num)
flit_key_df['FOP_keywords_num'].value_counts().hist()
plt.savefig('XLSX/statistic/FOP_keywords_num_%s_%s.png'%(category1,category2))
# plt.show()
plt.close()

# flit_key_df['TextRank_keywords_num'].value_counts().hist()
# plt.savefig('XLSX/statistic/TextRank_keywords_num_%s_%s.png'%(category1,category2))
# plt.show()
# plt.close()


# # 製作record bin檔

# In[60]:


import shutil
if not os.path.exists('bin'):
    shutil.rmtree('/bin', ignore_errors=True)

if not os.path.exists('bin/category/chunked'):
    os.makedirs('bin/category/chunked')

makevocab = True
if makevocab:
    vocab_counter = collections.Counter()
    
# train_file
flit_key_train_df = flit_key_df.iloc[:int(amount*0.8)]

# test_file
flit_key_test_df = flit_key_df.iloc[int(amount*0.8)+1:int(amount*0.9)]

# vald_file
flit_key_valid_df = flit_key_df.iloc[int(amount*0.9)+1:]
sentence_start = "<s>"
sentence_end = "</s>"


def xlsx2bin(set_name,df):
    sents = []
    with open("bin/category/%s.bin"%(set_name), 'wb') as file:
        i = 0
        for idx in tqdm(range(len(df))):
            series = df.iloc[idx]
            data_dict = series.to_dict()
            review_ID , big_categories , small_categories ,             orign_review , lemm_review , orign_summary , lemm_summary ,             FOP_keywords ,TextRank_keywords , TextRank_summary =             data_dict['review_ID'],data_dict['big_categories'],data_dict['small_categories'],             data_dict['review'],data_dict['lemm_review'], data_dict['summary'],data_dict['lemm_summary'],             data_dict['FOP_keywords'] , data_dict['TextRank_keywords'] , data_dict['TextRank_summary']

            '''
            review_ID , big_categories , small_categories , \
            orign_review , lemm_review , orign_summary , lemm_summary , \
            bert_review , bert_summary ,FOP_keywords ,TextRank_keywords = \
            data_dict['review_ID'],data_dict['big_categories'],data_dict['small_categories'], \
            data_dict['review'],data_dict['lemm_review'], data_dict['summary'],data_dict['lemm_summary'], \
            data_dict['bert_review'], data_dict['bert_summary'] , data_dict['FOP_keywords'] ,data_dict['TextRank_keywords']
            '''
            
#             print(FOP_keywords)

            # save Embedding/word2Vec calculate sents
#             for sent in nltk.sent_tokenize(lemm_review):
#                 sent = sent.replace("." ,"")
#                 sents.append(str(sent).split()) # 切分词汇 

#             for sent in nltk.sent_tokenize(lemm_summary):
#                 sent = sent.replace(sentence_start ,"").replace(sentence_end ,"")
#                 sents.append(str(sent).split()) # 切分词汇 

            lemm_review = lemm_review.replace("\n","")
            lemm_summary = lemm_summary.replace("\n","").replace("."," ")
            # lemm_summary = sentence_start + ' '+ lemm_summary + ' ' + sentence_end
#             print(lemm_summary)
            # Write to tf.Example
            tf_example = example_pb2.Example()
            try:
                tf_example.features.feature['orign_review'].bytes_list.value.extend(
                    [tf.compat.as_bytes(orign_review, encoding='utf-8')])

                tf_example.features.feature['orign_summary'].bytes_list.value.extend(
                    [tf.compat.as_bytes(orign_summary, encoding='utf-8')])
                
                tf_example.features.feature['review'].bytes_list.value.extend(
                    [tf.compat.as_bytes(lemm_review, encoding='utf-8')])

                tf_example.features.feature['summary'].bytes_list.value.extend(
                    [tf.compat.as_bytes(lemm_summary, encoding='utf-8')]) 
                '''    
                tf_example.features.feature['bert_review'].bytes_list.value.extend(
                    [tf.compat.as_bytes(bert_review, encoding='utf-8')])

                tf_example.features.feature['bert_summary'].bytes_list.value.extend(
                    [tf.compat.as_bytes(bert_summary, encoding='utf-8')])
                '''
                tf_example.features.feature['FOP_keywords'].bytes_list.value.extend(
                    [tf.compat.as_bytes(FOP_keywords, encoding='utf-8')]) 
            
                tf_example.features.feature['TextRank_keywords'].bytes_list.value.extend(
                    [tf.compat.as_bytes(TextRank_keywords, encoding='utf-8')]) 
                
                tf_example.features.feature['TextRank_summary'].bytes_list.value.extend(
                    [tf.compat.as_bytes(TextRank_summary, encoding='utf-8')])

                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)  
                file.write(struct.pack('q', str_len))
                file.write(struct.pack('%ds' % str_len, tf_example_str))
            except Exception as e:
                # print(e)
                pass
    print(" %s finished... "%(file.name))
    return sents
    
    
sents1 = xlsx2bin('train',flit_key_train_df)
sents2 = xlsx2bin('test',flit_key_test_df)
sents3 = xlsx2bin('valid',flit_key_valid_df)


# In[61]:


with open("bin/category/bin-info.txt",'w',encoding='utf-8') as f :
    f.write("train : %s\n"%(len(flit_key_train_df)))
    f.write("test : %s\n"%(len(flit_key_test_df)))
    f.write("valid : %s\n"%(len(flit_key_valid_df)))


# # 分割record bin檔(1000為單位)

# In[62]:


def chunk_file(set_name, chunks_dir):
    in_file = 'bin/category/%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
#         chunk_fname = os.path.join('bin', '/%s/%s_%03d.bin' % (chunks_dir,set_name, chunk))  # new chunk
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


def chunk_all(chunks_dir = 'bin/category/chunked'):
    # Make a dir to hold the chunks
    
    # Chunk the data
    for set_name in ['train', 'valid', 'test']:
        if not os.path.isdir(os.path.join(chunks_dir,set_name)):
            os.mkdir(os.path.join(chunks_dir,set_name))
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name, chunks_dir)
    print("Saved chunked data in %s" % chunks_dir)
    
chunk_all()


# In[63]:


def main_valid():
    #Performing rouge evaluation on 1.9 lakh sentences takes lot of time. So, create mini validation set & test set by borrowing 15k samples each from these 1.9 lakh sentences
    bin_valid_chuncks = os.listdir('bin/category/chunked/valid/')
    bin_valid_chuncks.sort()
    if not os.path.exists('bin/category/chunked/main_valid'):
        os.makedirs('bin/category/chunked/main_valid')
        
    samples = random.sample(set(bin_valid_chuncks[:-1]), 2)      #Exclude last bin file; contains only 9k sentences
    valid_chunk, test_chunk = samples[0], samples[1]
    shutil.copyfile(os.path.join('bin/category/chunked/valid', valid_chunk), os.path.join("bin/category/chunked/main_valid", "valid_00.bin"))
    shutil.copyfile(os.path.join('bin/category/chunked/valid', test_chunk), os.path.join("bin/category/chunked/main_valid", "test_00.bin"))
main_valid()


# # Embedding/word2Vec

# In[45]:


sentences = [] # total sentence
for idx in tqdm(range(len(orign_key_df))):
    series = orign_key_df.iloc[idx]
    data_dict = series.to_dict()
    lemm_review_sents , lemm_summary  = data_dict['lemm_review'],data_dict['lemm_summary'] 
    try:
        lemm_review_sents = eval(lemm_review_sents)
        for sent in lemm_review_sents:
            sent_tokens = sent.split(" ")
            tokens = [str(token) for token in sent.split() if (" " not in str(token))and (str(token) == '.' or str(token).isalpha())]
    #         tokens = [str(token) for token in sent.split() if (" " not in str(token))]
            sentences.append(tokens)   
        
            dot_tokens = [token for token in tokens if (token[0] == "." or token[-1] == ".") and (len(token)>1)]
            # if len(dot_tokens) > 0 :print(dot_tokens)
            
        sentences.append([t for t in lemm_summary.split(" ") if t not in ["<s>" , "</s>"]])
        
        dot_tokens = [token for token in lemm_summary.split(" ") if (token[0] == "." or token[-1] == ".") and (len(token)>1)]
        # if len(dot_tokens) > 0 :print(dot_tokens)
    except Exception as e:
        continue
        
    
print('word2Vec training sentence finished...')


# In[46]:


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


# In[47]:


# write vocab to file
if not os.path.exists('Embedding/category/word2Vec'):
    os.makedirs('Embedding/category/word2Vec')
    
if not os.path.exists("Embedding/category/word2Vec/word2Vec.300d.txt"):

    w2vec = word2vec.Word2Vec(sentences, size=300, min_count=2,max_vocab_size=None,iter=100,
                              sorted_vocab=1,max_final_vocab=vocab_count)

    

    w2vec.wv.save_word2vec_format('Embedding/category/word2Vec/word2Vec.300d.txt', binary=False)

    #保存模型，供日後使用
    # w2vec.save("Embedding/word2Vec/word2vec.model")   


# In[48]:


# sentences = sents1 + sents2 + sents3


# In[49]:


#模型讀取方式
# model = word2vec.Word2Vec.load("Embedding/word2Vec/word2vec.model")

wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
    'Embedding/category/word2Vec/word2Vec.300d.txt', binary=False, encoding='utf-8')

wvmodel.most_similar(u"player", topn=10)
# wvmodel.most_similar(['dvd','player','changer','machine','video'], topn=20)


# In[50]:


vocab_file = "Embedding/category/word2Vec/word.vocab"

if not os.path.exists(vocab_file):
    vocab_count = len(wvmodel.wv.index2entity)    

    print("Writing vocab file...")
    with open(vocab_file, 'w',encoding='utf-8') as writer:
        for word in wvmodel.wv.index2entity[:vocab_count]:
            # print(word, w2vec.wv.vocab[word].count)
            writer.write(word + ' ' + str(wvmodel.wv.vocab[word].count) + '\n') # Output vocab count
    print("Finished writing vocab file")


# In[51]:


word = wvmodel.wv.index2entity[25]
vector = wvmodel.wv.vectors[25]
print(word)
# print(vector)


# In[52]:


import torch
from data_util.data import Vocab
vocab_size = len(wvmodel.vocab) + 1


vocab = Vocab('Embedding/category/word2Vec/word.vocab', vocab_size)

embed_size = 300
weight = torch.zeros(vocab_size, embed_size)

for i in range(len(vocab._id_to_word.keys())):
    try:
        vocab_word = vocab._id_to_word[i+4]
        w2vec_word = w2vec.wv.index2entity[i]
    except Exception as e :
        continue
    if i + 4 > vocab_size: break
#     print(vocab_word,w2vec_word)
    weight[i+4, :] = torch.from_numpy(w2vec.wv.vectors[i])
        
embedding = torch.nn.Embedding.from_pretrained(weight)
# requires_grad指定是否在训练过程中对词向量的权重进行微调
embedding.weight.requires_grad = True
embedding


# In[53]:


vocab.word2id('the')


# # Embedding/glove

# In[54]:


from glove import Glove
from glove import Corpus

vocab_count = 50000
# write vocab to file
if not os.path.exists('Embedding/category/glove'):
    os.makedirs('Embedding/category/glove')


# In[55]:


if not os.path.exists("Embedding/category/glove/glove.model"):

    corpus_model = Corpus()
    corpus_model.fit(sentences, window=10)
    #corpus_model.save('corpus.model')
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)
    
    glove = Glove(no_components=300, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=100,
              no_threads=10, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)
    
    glove.save('Embedding/category/glove/glove.model') # 存模型
    corpus_model.save('Embedding/category/glove/corpus.model') # 存字典


glove = Glove.load('Embedding/category/glove/glove.model')
corpus_model = Corpus.load('Embedding/category/glove/corpus.model')


# In[56]:


vocab_file = "Embedding/category/glove/word.vocab"

if not os.path.exists(vocab_file):
#     vocab_count = len(glove.dictionary)    
    vocab_count = 0
    print("Writing vocab file...")
    with open(vocab_file, 'w',encoding='utf-8') as writer:
        for word,idx in glove.dictionary.items():
            if word in vocab._word_to_id.keys():
                vocab_count += 1
                writer.write(word + ' ' + str(idx) + '\n') # Output vocab count
    print("Finished writing vocab file %s" %(vocab_count))


# In[57]:


glove.word_vectors[glove.dictionary['.']].shape
# vocab._word_to_id.keys()
# len(glove.dictionary)


# In[58]:


vocab_size = len(open('Embedding/category/glove/word.vocab').readlines())
print(vocab_size)

vocab = Vocab('Embedding/category/glove/word.vocab', vocab_size)
embed_size = 300
weight = torch.zeros(vocab_size, embed_size)

for word,idx in glove.dictionary.items():
    if word in vocab._word_to_id.keys():
        wid = vocab.word2id(word) 
        vector = np.asarray(glove.word_vectors[glove.dictionary[word]], "float32")
        weight[wid, :] = torch.from_numpy(vector)

embedding = torch.nn.Embedding.from_pretrained(weight)
# requires_grad指定是否在训练过程中对词向量的权重进行微调
embedding.weight.requires_grad = True
embedding 


# # Embedding/Bert

# In[59]:


from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
# model.eval()
model.embeddings.word_embeddings


# vocab = Vocab('Embedding/word2Vec/word2Vec.vocab', vocab_size)

# embed_size = 300
# weight = torch.zeros(vocab_size, embed_size)


# embedding = torch.nn.Embedding.from_pretrained(weight)
# # requires_grad指定是否在训练过程中对词向量的权重进行微调
# embedding.weight.requires_grad = True
# embedding        


# In[1]:


# !jupyter nbconvert --to script makeRecord.ipynb

