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

i = 0

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
      # l = str(l)
      yield json.loads(l)

def getProdDF(path): # description[list] ,main_cat[str]
  global mongo
  df = {}
  i = 0
  for d in parse(path):
    if isinstance(d, (bytes, bytearray)):
        d = d.decode()
        d = json.loads(d) # 輸出dict型別

    keys = d.keys()


    if "tech1" not in keys: d["tech1"] = ""
    if "salesRank" not in keys: d["tech1"] = ""
    if "feature" not in keys: d["feature"] = ""
    if "similar_item" in keys: del d["similar_item"]

    if "category" in keys:
        d["category1"] = list(d["category"])[0]
        d["category2"] = list(d["category"])[-1]
        # print([d["category1"],d["category2"]])
    else:
        continue
    # if d["category1"] == "": continue
    # if d["category2"] == "": continue

    if "related" in keys: del d["related"]
    # if "salesRank" in keys:
    #   if len(d["salesRank"]) > 0:
    #       for k, v in d["salesRank"].items():
    #           d["salesRank"] = int(v)
    #           break
    if "image" in keys: del d["image"]
    if "also_buy" in keys: del d["also_buy"]
    if "also_view" in keys: del d["also_view"]
    if "title" in keys:
        if len(d["title"]) == 0: d["title"] = ""
        if d["title"] == "nan": d["title"] = ""
    else:
        d["title"] = ""

    # if "tech1" not in keys: continue
    # if "tech2" not in keys: continue

    if "date" in keys:
        try:
            date = parser.parse(d["date"])
            d["date"] = date
        except Exception as e:
            d["date"] = ""
    else: d["date"] = ""

    if "rank"  in keys: # salesRank in main_cat
        d["salesRank"] = ""
        rank = d["rank"]
        # print(rank)
        if type(rank) == list:
            rank = rank[0]
        rank = rank.replace(",", "")
        rank = rank.split("in")[0]
        rank = rank.replace(">#", " ")
        for num in rank.split(" "):
            if num.isnumeric():
                d["salesRank"] = num
                break                  
    else:
        d["salesRank"] = ""

    if "description" in keys:
        description = d["description"]
        if type(description) == list:
            description = [desc for desc in description if desc != ""]
            # description = "\n".join(description)
            d["description"] = description
    else: continue

    if "main_cat" not in keys:
        d["main_cat"] = ""
    # mongo.insert(d, db_col="new_Product2")
    # print(d)

    save_dict = {
        "asin": d["asin"],
        "salesRank": d["salesRank"],
        "category1": d["category1"],
        "category2": d["category2"],
        "main_cat" : d["main_cat"]
    }
    df[i] = save_dict
    i = i + 1
    if (i % 100000 == 0) and (i>0): print(i);
  print("Write finished")
  return pd.DataFrame.from_dict(df, orient='index')

def makeCsv(): # only product
    # for meta_path in glob('raw/small_meta/*.*'):
    meta_path = "all-meta-review/All_Amazon_Meta.json.gz"
    print(meta_path)
    df = getProdDF(meta_path)
    print(df.head())
    csv_path = meta_path.replace('all-meta-review/', "csv/").replace('.json.gz', ".csv")
    df.to_csv(csv_path) #默认dt是DataFrame的一个实例，参数解释如下
    print(csv_path + " Write finished")
    print(" Write finished")

def makeCsv2(): # only product
    # for meta_path in glob('raw/small_meta/*.*'):
    idx = 0
    meta_path = "raw/*.json.gz"
    prod = None
    for meta_path in glob(meta_path):
        print(meta_path)
        df = getProdDF(meta_path)
        csv_path = meta_path.replace('raw', "csv").replace('.json.gz', ".csv")
        df.to_csv(csv_path) #默认dt是DataFrame的一个实例，参数解释如下
        print(csv_path + " Write finished")

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
        except Exception as e:
            pass

    prod.to_csv("csv/meta_ALL.csv")

if __name__ == "__main__":
    # metaToCsv.makeCsv()
    print("Connect to MongoDB")
    mongo = MongoDB()
    mongo_db = mongo.conn_db(db_name='Amazon')

    # mongo_coll = mongo_db['new_Product2']
    # mongo_coll.create_index([('asin', pymongo.TEXT)])
    # mongo_coll.create_index([('category1', pymongo.TEXT)])
    # mongo_coll.create_index([('category2', pymongo.TEXT)])

    # mongo_coll.create_index([('asin', pymongo.TEXT),
    #                          ('date', pymongo.TEXT),
    #                          ('category1', pymongo.TEXT),
    #                          ('category2', pymongo.TEXT)])  # 对名为field的项建立文档索引



    # make5coreCsv()
    makeCsv()
    # makeCsv2()
