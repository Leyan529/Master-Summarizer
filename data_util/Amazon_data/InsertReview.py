import json
import gzip
import pandas as pd

from bs4 import BeautifulSoup
import requests
import nltk
import re
from MongoDB import MongoDB
import threading
import random
from glob import glob
from datetime import datetime
import dateutil.parser as parser
import pymongo
import os

import threading
import time
from preprocess import *
from pymongo import MongoClient
import pymongo

i = 0

def token_process(d):
    review,summary = d["reviewText"],d["summary"]

    # review = lemm_sent_process(review, remove_stopwords=True, mode="spacy")
    # summary = lemm_sent_process(summary, remove_stopwords=False, mode="spacy")
    keywords, review = removeSpecWord(review)
    review = lemm_sent_process(review,remove_stopwords=True,summary=False,mode = "nltk",withdot =True)
    summary = lemm_sent_process(review,remove_stopwords=True,summary=False,mode = "nltk",withdot =True)
    d["lemm_review"] = review
    d["lemm_summary"] = summary
    d["review_keywords"] = keywords

    d["review_token_len"] = calc_len(review)
    d["summary_token_len"] = calc_len(summary)

    # if (type(description) == str):
    #     description = squeeze(get_Lemmatizer(description, remove_stopwords=True))
    #     d["lemm_desc"] = description
    #     d["desc_token_len"] = calc_len(description)
    return d

def getProd():
    if not os.path.exists("csv/All_Amazon_Meta.csv"):
        prod = None
        idx = 0
        for csv_path in glob("csv/*.csv"):
            # print(csv_path)
            try:
                newprod = pd.read_csv(csv_path, delimiter=',',
                                      usecols=['asin', 'category1', 'category2', 'salesRank',
                                               "date", "main_cat", "tech1", "brand", "description", "feature",
                                               "title"], index_col=False)
                newprod = newprod.fillna("")

                if idx == 0:
                    prod = newprod
                    # print(prod.head())
                else:
                    prod = pd.concat([prod, newprod], axis=0, ignore_index=False)
                    prod = pd.DataFrame(prod, columns=list(prod.columns))
                    del newprod
                idx += 1
                print(csv_path + ": success")
                prod.to_csv("csv/meta_ALL.csv")
            except Exception as e:
                # if csv_path in ["csv/Luxury_Beauty.csv",
                #                 "csv/Kindle_Store.csv",
                #                 "csv/AMAZON_FASHION.csv",
                #                 "csv/All_Beauty.csv",
                #                 "csv/Magazine_Subscriptions.csv",
                #                 "csv/Prime_Pantry.csv"
                #                 ]:
                #     print(csv_path + ": error")
                #     print(newprod.columns)
                pass
                # print(csv_path + ": error" )
                # print(newprod.columns())
                # csv / Luxury_Beauty.csv: error
                # csv / Kindle_Store.csv: error
                # csv / AMAZON_FASHION.csv: error
                # csv / All_Beauty.csv: error
                # csv / Magazine_Subscriptions.csv: error
                # csv / Prime_Pantry.csv: error
                # print(e)
                # break
    else:
        print("load past csv...")
        prod = pd.read_csv("csv/All_Amazon_Meta.csv", delimiter=',',
                           usecols=['asin', 'category1', 'category2', 'salesRank',"main_cat"],
                           index_col=False)
        prod = prod.fillna("")

    asins = list(prod["asin"])
    products = len(asins)

    category1s = list(prod["category1"])
    category2s = list(prod["category2"])
    salesRanks = list(prod["salesRank"])
    main_cats = list(prod["main_cat"])

    print("Item list ...")
    items_lists = []
    for i in range(products):
        try:
            items_lists.append([category1s[i], category2s[i], int(salesRanks[i]), main_cats[i]])
            # print(category1s[i], category2s[i])
        except Exception as e:
            items_lists.append([category1s[i], category2s[i], 0, main_cats[i]])
            # print(category1s[i], category2s[i])
        if (i % 1000000 == 0) and (i > 0 ):
            print(i)

    mydict = dict(zip(asins, items_lists))

    print("Load finished")
    return asins,mydict


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
      # l = str(l)
      yield json.loads(l)


def getDF(path):
  global mongo,i
  asins, mydict = getProd()

  print("finished ...")
  # print(len(asins))

  idx = 0
  for d in parse(path):
    if isinstance(d, (bytes, bytearray)):
        d = d.decode()
        d = json.loads(d) # 輸出dict型別

    asin = d["asin"]
    keys = d.keys()

    idx += 1
    if (idx % 1000 == 0) and (idx > 0):
        print(idx)
    try :
        category1, category2, salesRank, main_cat = mydict[asin]
        if "vote" not in keys: continue
        if "overall" not in keys: continue
        # if "reviewerID" not in keys: continue
        if "reviewTime" not in keys: continue
        if "reviewText" not in keys: continue
        if "summary" not in keys: continue
        if "image" in keys: del d["image"]
        # if "unixReviewTime" not in keys: continue
        # del d["unixReviewTime"]

        date = parser.parse(d["reviewTime"])
        d["reviewTime"] = date


        #----------------------------------------------------
        # print(ategory1, category2, salesRank, main_cat)
        if category1 == "":continue
        if category2 == "":continue

        d["big_categories"] = category1
        d["small_categories"] = category2
        d["main_cat"] = main_cat
        try:
            d["vote"] = int(d["vote"].replace(",", ""))
        except Exception as e:
            d["vote"] = 0
        if d["vote"] == 0: continue
        if "image" in keys: del d["image"]
        d["salesRank"] = int(salesRank)  # salesRank 在最大主類別裡的銷售排行
        # d = token_process(d)
        #----------------------------------------------------
        mongo.insert(d, db_col="new_reviews2")

        # ------------------------------------------------------------------------------------
    except Exception as e:
        pass

print("Write finished")

def makeReviewAll(): # only review
      # for review_path in glob('raw/k-score-review/*.*'):
      review_path = "all-meta-review/All_Amazon_Review.json.gz"
      print(review_path)
      getDF(review_path)

if __name__ == "__main__":
    # metaToCsv.makeCsv()
    print("Connect to MongoDB")
    mongo = MongoDB()
    mongo_db = mongo.conn_db(db_name='Amazon')

    mongo_coll = mongo_db['new_reviews2']
    mongo_coll.create_index([('asin', pymongo.TEXT)])
    # mongo_coll.create_index([('category1', pymongo.TEXT)])
    # mongo_coll.create_index([('category2', pymongo.TEXT)])

    # mongo_coll.create_index([('asin', pymongo.TEXT),
    #                          ('date', pymongo.TEXT),
    #                          ('category1', pymongo.TEXT),
    #                          ('category2', pymongo.TEXT)])  # 对名为field的项建立文档索引



    makeReviewAll()
    # makeCsv()
