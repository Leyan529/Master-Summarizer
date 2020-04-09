import shutil
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

# coding: utf-8

# In[1]:

#%%
# NEW_PROD_DICT 
from data_util.product import *
from data_util.mainCat import *
from data_util.MongoDB import *
from data_util.stopwords import *
from data_util.preprocess import *
from data_util.eda import *
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

from summa import keywords as TextRank
from summa.summarizer import summarize

# 斷詞辭典
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
stpwords_list3 = [f.replace("\n","") for f in open("data_util/stopwords.txt","r",encoding = "utf-8").readlines()]
stpwords_list3.remove("not")
stopwords = list(html_escape_table + stpwords_list2) + list(list(stops) + list(stpwords_list1) + list(stpwords_list3))
print("斷詞辭典 已取得")

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

mode = 'prod'
# mode = 'main_cat'
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

'''
cd D:\WorkSpace\JupyterWorkSpace\Text-Summarizer-BERT2\
D:
activate tensorflow
python FOP_bin.py
'''

# In[115]:

from transformers import BertModel, BertTokenizer 
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import pandas as pd


# corpus
# from collections import Counter, OrderedDict
# corpus_path = '%s/corpus.txt'%(folder) 
# corpus = []
# with open(corpus_path,'r',encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         tokens = line.replace(" \n",'').replace("<s>",'').replace("</s>",'').split(" ")
#         tokens = [token for token in tokens if token != '']
#         corpus.append(tokens)

# prod_keywords
# fn = '%s/prod_keywords.txt'%(folder)
# print('load %s keywords...' % (fn))
# prod_keywords = set()
# with open(fn, 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         k, v = line.split(":")
#         prod_keywords.add(k)

# review_keywords
fn = '%s/review_keywords.txt'%(folder)
print('load %s keywords...' % (fn))
review_keywords = set()
with open(fn, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        k, v = line.split(":")
        review_keywords.add(k)        

# read process review xlsv
csv_path = '%s/review.xlsx'%(folder)    
print("check ", os.path.exists(csv_path))
if os.path.exists(csv_path):
    df = pd.read_excel(csv_path)
    print("previous file %s ...."%(csv_path)) 

# FOP、TextRank (keyword、summary)改為batcher內置

# statistic
from matplotlib import pyplot as plt
statistic_path = '%s/statistic'%(folder) 
if not os.path.exists(statistic_path):
    os.makedirs(statistic_path)

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

# bin write info

if not os.path.exists('%s/bin'%(folder)):
    shutil.rmtree('%s/bin'%(folder), ignore_errors=True)

if not os.path.exists('%s/bin/chunked'%(folder)):
    os.makedirs('%s/bin/chunked'%(folder))

makevocab = True
if makevocab:
    vocab_counter = collections.Counter()
# filt
df = df[(df.lemm_summary_len >= 8) ] # 過濾single word summary
df = df[(df.lemm_review_len <= 1000) ] # 過濾single word summary
df = df[(df.lemm_review_len >= 50) ] # 過濾single word summary
# df = df[(df.cheat_num >= 2) ] # 過濾single word summary

df = df.dropna(
    axis=0,     # 0: 对行进行操作; 1: 对列进行操作
    how='any'   # 'any': 只要存在 NaN 就 drop 掉; 'all': 必须全部是 NaN 才 drop 
    )

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

def func(review_ID , review , summary , cheat_num, file, lock):
    lock.acquire()
    time.sleep(0.01)
    review = review.replace("\n","")
    summary = summary.replace("\n","")

    # Write to tf.Example
    tf_example = example_pb2.Example()
    try:                
        tf_example.features.feature['review'].bytes_list.value.extend(
            [tf.compat.as_bytes(review, encoding='utf-8')])

        tf_example.features.feature['summary'].bytes_list.value.extend(
            [tf.compat.as_bytes(summary, encoding='utf-8')]) 

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

                    
        tf_example.features.feature['POS_FOP_keywords'].bytes_list.value.extend(
            [tf.compat.as_bytes(POS_FOP_keywords, encoding='utf-8')]) 

        tf_example.features.feature['DEP_FOP_keywords'].bytes_list.value.extend(
            [tf.compat.as_bytes(DEP_FOP_keywords, encoding='utf-8')])                 
    
        # TextRank
        TextRank_keywords , TextRank_summary = [] , []

        for words in TextRank.keywords(review).split('\n'):
            TextRank_keywords.extend(words.split(" "))
        TextRank_keywords = " ".join(TextRank_keywords)

        for words in summarize(review, words=2).split('\n'):
            TextRank_summary.extend(words.split(" "))
        TextRank_summary = " ".join(TextRank_summary) # 有問題

        tf_example.features.feature['TextRank_keywords'].bytes_list.value.extend(
            [tf.compat.as_bytes(TextRank_keywords, encoding='utf-8')]) 
        
        # tf_example.features.feature['TextRank_summary'].bytes_list.value.extend(
        #     [tf.compat.as_bytes(TextRank_summary, encoding='utf-8')])

        # print(POS_FOP_keywords)
        # print(DEP_FOP_keywords)
        # print(TextRank_keywords);print('-------------')
        # print(TextRank_summary)
        

        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)  
        file.write(struct.pack('q', str_len))
        file.write(struct.pack('%ds' % str_len, tf_example_str))

    except Exception as e:
        pass        
    lock.release()

# ../Train-Data/
# write bin
def xlsx2bin(set_name,split_df):
    sents = []
    with open("%s/bin/%s.bin"%(folder,set_name), 'wb') as file:
        i = 0
        # threads = []
        # lock = threading.BoundedSemaphore(10) #最多允許10個執行緒同時執行
        for idx in tqdm(range(len(split_df))):
            series = split_df.iloc[idx]
            data_dict = series.to_dict()
            review_ID , review , summary , cheat_num = \
            data_dict['review_ID'], data_dict['review'], data_dict['summary'], data_dict['cheat_num'] 
            # cheat = eval(cheat)
            review = squeeze2(review)
            summary = squeeze2(summary)
            rev_token_set = set(review.split(" "))
            summ_token_set = set(summary.split(" "))
            cheat = rev_token_set & summ_token_set & ( review_keywords | set(opinion_lexicon["total-words"]) )
            if len(cheat) < 4 : continue
            # print(type(cheat),cheat)            
            # t = threading.Thread(target=func,args=(review_ID , review , summary , cheat_num, file, lock))
            # threads.append(t)
            # t.start()

            review = review.replace("\n","")
            remove_chars = '[0-9]+'
            review = re.sub(remove_chars, "", review)
            review = " ".join([t for t in review.split(" ") if t != ""])
            summary = summary.replace("\n","")

            # Write to tf.Example
            tf_example = example_pb2.Example()
            try:                
                tf_example.features.feature['review'].bytes_list.value.extend(
                    [tf.compat.as_bytes(review, encoding='utf-8')])

                tf_example.features.feature['summary'].bytes_list.value.extend(
                    [tf.compat.as_bytes(summary, encoding='utf-8')]) 

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

                            
                tf_example.features.feature['POS_FOP_keywords'].bytes_list.value.extend(
                    [tf.compat.as_bytes(POS_FOP_keywords, encoding='utf-8')]) 

                tf_example.features.feature['DEP_FOP_keywords'].bytes_list.value.extend(
                    [tf.compat.as_bytes(DEP_FOP_keywords, encoding='utf-8')])                 
            
                # TextRank
                TextRank_keywords , TextRank_summary = [] , []

                for words in TextRank.keywords(review).split('\n'):
                    TextRank_keywords.extend(words.split(" "))
                TextRank_keywords = " ".join(TextRank_keywords)

                for words in summarize(review, words=2).split('\n'):
                    TextRank_summary.extend(words.split(" "))
                TextRank_summary = " ".join(TextRank_summary) # 有問題

                tf_example.features.feature['TextRank_keywords'].bytes_list.value.extend(
                    [tf.compat.as_bytes(TextRank_keywords, encoding='utf-8')]) 
                
                # tf_example.features.feature['TextRank_summary'].bytes_list.value.extend(
                #     [tf.compat.as_bytes(TextRank_summary, encoding='utf-8')])

                # print(POS_FOP_keywords)
                # print(DEP_FOP_keywords)
                # print(TextRank_keywords);print('-------------')
                # print(TextRank_summary)
                

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
    
    



# 分割record bin檔(1000為單位)   
def chunk_file(set_name, chunks_dir):
    in_file = '%s/bin/%s.bin' % (folder,set_name)
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
train_len = xlsx2bin('train',flit_key_train_df)
test_len = xlsx2bin('test',flit_key_test_df)
valid_len = xlsx2bin('valid',flit_key_valid_df)

with open("%s/bin/bin-info.txt"%(folder),'w',encoding='utf-8') as f :
    f.write("train : %s\n"%(train_len))
    f.write("test : %s\n"%(test_len))
    f.write("valid : %s\n"%(valid_len))

chunk_all() 
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
            tokens = line.replace(" \n",'').replace("<s>",'').replace("</s>",'').split(" ")
            tokens = [token for token in tokens if (token != '' and token not in stopwords and token.isalpha())]
            corpus.append(tokens)
            embedding_corpus.append(tokens)
    return embedding_corpus

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
if not os.path.exists("%s/Embedding/word2Vec/word2Vec.300d.txt"%(folder)):
    embedding_corpus = make_corpus()
    w2vec = word2vec.Word2Vec(embedding_corpus, size=300, min_count=2,max_vocab_size=None,iter=100,
                              sorted_vocab=1,max_final_vocab=vocab_count)    

    w2vec.wv.save_word2vec_format('%s/Embedding/word2Vec/word2Vec.300d.txt'%(folder), binary=False)

#模型讀取方式
wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
    '%s/Embedding/word2Vec/word2Vec.300d.txt'%(folder), binary=False, encoding='utf-8')
wvmodel.most_similar(u"player", topn=10)  

vocab_file = "%s/Embedding/word2Vec/word.vocab"%(folder)

if not os.path.exists(vocab_file):
    vocab_count = len(wvmodel.wv.index2entity)    

    print("Writing vocab file...")
    with open(vocab_file, 'w',encoding='utf-8') as writer:
        for word in wvmodel.wv.index2entity[:vocab_count]:
            # print(word, w2vec.wv.vocab[word].count)
            writer.write(word + ' ' + str(wvmodel.wv.vocab[word].count) + '\n') # Output vocab count
    print("Finished writing vocab file")

from data_util.data import Vocab
vocab_size = len(wvmodel.vocab) + 1
vocab = Vocab('%s/Embedding/word2Vec/word.vocab'%(folder), vocab_size)

# Embedding/glove    
from glove import Glove
from glove import Corpus
from gensim import corpora

vocab_count = 50000
# write vocab to file
if not os.path.exists('%s/Embedding/glove'%(folder)):
    os.makedirs('%s/Embedding/glove'%(folder))

if not os.path.exists("%s/Embedding/glove/glove.model"%(folder)):
    corpus_model = Corpus()
    corpus_model.fit(embedding_corpus, window=10)
    #corpus_model.save('corpus.model')
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)
    
    glove = Glove(no_components=300, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=100,
              no_threads=10, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)
    
    glove.save('%s/Embedding/glove/glove.model'%(folder)) # 存模型
    corpus_model.save('%s/Embedding/glove/corpus.model'%(folder)) # 存字典

    #透過gensim以text_data建立字典
    dictionary = corpora.Dictionary(embedding_corpus)
    dictionary.save('%s/Embedding/glove/dictionary.gensim'%(folder))


glove = Glove.load('%s/Embedding/glove/glove.model'%(folder))
corpus_model = Corpus.load('%s/Embedding/glove/corpus.model'%(folder))
dictionary = gensim.corpora.Dictionary.load('%s/Embedding/glove/dictionary.gensim'%(folder))


# write vocab to file
vocab_file = "%s/Embedding/glove/word.vocab"%(folder)
if not os.path.exists(vocab_file):
#     vocab_count = len(glove.dictionary)    
    vocab_count = 0
    print("Writing vocab file...")
    with open(vocab_file, 'w',encoding='utf-8') as writer:
        for word,idx in glove.dictionary.items():
            try:
                word_id = dictionary.token2id[word]
                word_freq = dictionary.dfs[word_id]
                if word_freq < 2 : continue
                writer.write(word + ' ' + str(word_freq) + '\n') # Output vocab count
            except Exception as e :
            # if word in vocab._word_to_id.keys():
                writer.write(word + ' ' + str(0) + '\n') # Output vocab count
            vocab_count += 1                
    print("Finished writing vocab file %s" %(vocab_count))

# write vocab to file
# vocab_file = "%s/Embedding/glove/word.vocab"%(folder)
# if not os.path.exists(vocab_file):
# #     vocab_count = len(glove.dictionary)    
#     vocab_count = 0
#     print("Writing vocab file...")
#     with open(vocab_file, 'w',encoding='utf-8') as writer:
#         for word in list(dictionary.token2id):
#             word_id = dictionary.token2id[word]
#             word_freq = dictionary.dfs[word_id]
#             writer.write(word + ' ' + str(word_freq) + '\n') # Output vocab count
#     print("Finished writing vocab file %s" %(len(list(dictionary.token2id))))