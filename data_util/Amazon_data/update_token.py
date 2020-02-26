import json
import gzip
import pandas as pd
from MongoDB import MongoDB
# import metaToCsv
import threading
import time
from preprocess import *
import os
from glob import glob
from pymongo import MongoClient
import pymongo

# categroy_list = [
#     'Electronics', 'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry', 'Office_Products', 'Automotive',
#     'Movies_and_TV', 'Tools_and_Home_Improvement', 'Home_and_Kitchen', 'Video_Games', 'Patio_Lawn_and_Garden',
#     'Sports_and_Outdoors',
#     # 'Apps_for_Android', # not have require column
#     'Health_and_Personal_Care'
# ]  # 0~11
#
# categroy_list = [c + ".json.gz" for c in categroy_list]

import nltk
from nltk import word_tokenize, pos_tag


def token_process(d):
    review,summary = d["reviewText"],d["summary"]

    # review = lemm_sent_process(review, remove_stopwords=True, mode="spacy")
    # summary = lemm_sent_process(summary, remove_stopwords=False, mode="spacy")
    keywords, review = removeSpecWord(review)
    review = lemm_sent_process(review,remove_stopwords=False,summary=False,mode = "nltk",withdot =True)
    summary = lemm_sent_process(summary,remove_stopwords=False,summary=False,mode = "nltk",withdot =True)
    d["lemm_review"] = review
    d["lemm_summary"] = summary
    d["review_keywords"] = keywords

    d["review_token_len"] = calc_len(review)
    d["summary_token_len"] = calc_len(summary)

    return d


if __name__ == "__main__":
    # metaToCsv.makeCsv()
    print("Connect to MongoDB")
    mongo = MongoDB()
    mongo_db = mongo.conn_db(db_name='Amazon')
    mongo_coll = mongo_db['new_reviews3']
    mongo_coll.create_index([('asin', pymongo.TEXT)])
    del mongo_db,mongo_coll

    key = {}
    cursor = mongo.searchInDB(key, db_col='new_reviews2')
    amount = cursor.count()

    i = 0

    st_time = time.time()
    for entry in cursor:
        keys = entry.keys()
        asin = entry["asin"]

        if i > 0 and i % 1000 == 0:
            en_time = time.time()
            print("process reviews:",i, en_time - st_time) #已證實跟記憶體爆炸無關
            st_time = en_time

        i = i + 1

        entry = token_process(entry)
        mongo.insert(entry, db_col="new_reviews3")

