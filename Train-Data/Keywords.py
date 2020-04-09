
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

# Load Spec From Mongo
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
    key = mongoObj.getProductKey()

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
    key = mongoObj.getProductKey()
'''
cd D:\WorkSpace\JupyterWorkSpace\Text-Summarizer-BERT2\
D:
activate tensorflow
python keywords.py
'''

# In[2]:

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

    remove_chars = '["#$%&\'\"\()*+:<=>?@★【】《》“”‘’[\\]^_`{|}~]+'
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
    text = [line.strip() for line in text.split("\n") if line != '']
    text = "\n".join(text)
    
    text = " ".join(bert_tokenizer.tokenize(text))
    text = text.replace(" ##","")
    text = text.split(" . ")
    text = [line.replace(" .","") for line in text]
    text = [line for line in text if len(line) > 0]

    return text

descr = """
"Amerock BP5322BJ Allison Value Hardware 1-1/2\" Round Knob, BlackAmerock is introducing the launch of 36 new skus, 12 size extensions and applying its best selling finishes to the Allison Value Hardware collection. The new product assortment will add additional depth and breadth to this collection.Amerock BP5322BJ Allison Value Hardware 1-1/2\" Round Knob, Black Features: From rustic to modern-day casual to sophisticated beauty, the Allison Value Hardware collection offers a variety of designs, making on-trend, quality hardware affordable 1 x #8 32 x 1''Amerock BP5322BJ Allison Value Hardware 1-1/2\" Round Knob, Black Specifications: Collection: Allison Value Hardware Diameter (Inches): 1.5 Finish: Black Projection (Inches): 1.125 Type: Knob Model Variations: BP5322-BJ, BP5322, BP5322BJ Previously known as: BP5322-BJ", 
        "The Amerock Ceramic Knob comes in black and is 1-1/2 inches in diameter. This collection is a favorite accent for wood cabinetry and a popular match for porcelain fixtures. Choose from a wide variety of styles and finishes to create the look you want. These high quality ceramics are skillfully crafted to ensure durability and lasting beauty. For more than 70 years, Amerock has manufactured quality cabinet hardware and provided dependable service nationwide."
"""


# In[3]:

from data_util.stopwords import *
from data_util.preprocess import *
from data_util.eda import *

# descr = process(descr)
# lemm_keyword(descr)

def lemm_keyword(text):
    text_keywords = []
    lemm_sents = []
    for line in text:
        lemm_text = lemm_sent_process4(line, remove_stopwords=False, summary=False, mode="spacy",withdot=False)
        lemm_text = lemm_text.replace("\n",'')
        text_keywords2 = PF_rule_POS(lemm_text).run()
        lemm_sents.append(lemm_text)
        text_keywords.extend(text_keywords2)

    return lemm_sents, text_keywords

# descr = process(descr)


# In[4]:

def loadProdSpecData():
    global big_categories, small_categories, main_cat, cond_date

    # Connect MongoDB
    print("Connect to MongoDB")
    mongo = MongoDB()
    mongo.conn_db(db_name='Amazon')

    db_col = 'new_Product2'
    PROD_DICT = {}
    SPEC_LIST, ASIN_LIST, TITLE_LIST = [], [], []
    prod_cursor = mongo.searchInDB(key, db_col=db_col)
    docCount = prod_cursor.count()
    print("make product spec feature from %s products..." % (docCount))
    return prod_cursor , docCount    
# prod_cursor , docCount = loadProdSpecData()


# In[ ]:

from collections import Counter, OrderedDict
import pandas as pd


def make_prod_xlsx():
    prod_cursor , docCount = loadProdSpecData()
    with tqdm(total=docCount) as pbar:
        asin_list, title_list, description_list, big_categories_list, small_categories_list, salesRank_list, feature_list = [] , [] , [] , [] , [] , [] , []

        for i1, prod in enumerate(prod_cursor):
            asin, title, description, big_categories, small_categories, salesRank, feature = prod["asin"], prod["title"],         prod["description"], prod["category1"], prod["category2"], prod["salesRank"], prod["feature"]
            asin_list.append(asin)
            title_list.append(title)
            description_list.append(description)
            big_categories_list.append(big_categories)
            small_categories_list.append(small_categories)
            salesRank_list.append(salesRank)
            feature_list.append(feature)       
            pbar.update(1)
            
        df = pd.DataFrame({"asin":asin_list, "title": title_list, "description": description_list, "big_categories": big_categories_list,
                        "small_categories": small_categories_list , "salesRank": salesRank_list,"feature": feature_list})
    return df

if not os.path.exists(folder):
    os.makedirs(folder)
        
csv_path = '%s/prod.xlsx'%(folder)    
print("check ", os.path.exists(csv_path))
if os.path.exists(csv_path):
    df = pd.read_excel(csv_path)  
    print("previous file %s ...."%(csv_path))  
else:
    df = make_prod_xlsx()
    df.to_excel(csv_path, encoding='utf8')
    print(csv_path + " Write finished")    


# In[ ]:
import threading
import time
feature_counter = Counter()
docCount = len(df)

def func(asin, title, description, big_categories, small_categories, salesRank, feature):
    global feature_counter
    lock.acquire()
    time.sleep(0.01)
    if type(description) == str:          
            description = process(description)
            _, keywords1 = lemm_keyword(description)
            feature_counter.update(keywords1)

    if type(feature) == str:
        feature = process(feature)
        _, keywords2 = lemm_keyword(feature)
        feature_counter.update(keywords2)

    pbar.update(1)
    pbar.set_description("%s %s keyword " % (folder,len(feature_counter.items())))
    lock.release()

with tqdm(total=docCount) as pbar:
    threads = []
    lock = threading.BoundedSemaphore(10) #最多允許10個執行緒同時執行
    for i in range(docCount):      
        try:
            prod = df.iloc[i]
            asin, title, description, big_categories, small_categories, salesRank, feature = prod["asin"], prod["title"], prod["description"], prod["big_categories"], prod["small_categories"], prod["salesRank"], prod["feature"]
        except Exception as e :
            continue

        # t = threading.Thread(target=func,args=(asin, title, description, big_categories, small_categories, salesRank, feature,))
        # threads.append(t)
        # t.start() # 2:40
        if type(description) == str:          
            description = process(description)
            _, keywords1 = lemm_keyword(description)
            feature_counter.update(keywords1)

        if type(feature) == str:
            feature = process(feature)
            _, keywords2 = lemm_keyword(feature)
            feature_counter.update(keywords2)

        pbar.set_description("%s %s keyword " % (folder,len(feature_counter.items())))
        pbar.update(1)
#         if i1 == 100: break
    # for t in threads:
    #     t.join()


# In[ ]:

important_features = OrderedDict(sorted(feature_counter.items(), key=lambda pair: pair[1], reverse=True))
important_features = [(word, important_features[word]) for word in important_features]
print("Count : %s" % (len(important_features)))

fn3 = '%s/prod_keywords.txt'%(folder)
with open(fn3, 'w', encoding="utf-8") as f:
    total_keywords = set()
    for word, v in important_features:
        f.write("%s:%s \n" % (word, v))
        total_keywords.add(word)


# In[ ]:

total_keywords


# In[ ]:


