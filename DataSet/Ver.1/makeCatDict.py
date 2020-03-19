# NEW_PROD_DICT
from data_util.mainCat import *
from data_util.MongoDB import *
from data_util.stopwords import *
from data_util.preprocess import *

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
# from mainCat import *

# mongoObj = All_Electronics()
# mongoObj = Pet_Supplies()
# mongoObj = Sports_Outdoors()
# mongoObj = Health_personal_Care()
# -------------------------------------#
# mongoObj = CellPhones_Accessories()
# mongoObj = Camera_Photo()
# mongoObj = GPS_Navigation()
# mongoObj = Music_Instrum()
# mongoObj = Software()
mongoObj = Computers()
# mongoObj = Video_Games()
# --------------------------------#
'''
cd D:\WorkSpace\JupyterWorkSpace\Text-Summarizer-BERT2\
D:	
activate tensorflow
python makeCatDict.py
'''

# cd D:\WorkSpace\JupyterWorkSpace\Text-Summarizer-FOP\
# d:
# activate tensorflow



main_cat = mongoObj.getAttr()
print("make data dict from main_cat : %s " % (main_cat))

# Connect MongoDB
print("Connect to MongoDB")
mongo = MongoDB()
mongo.conn_db(db_name='Amazon')


def loadProdSpecData():
	global main_cat, cond_date
	db_col = 'new_Product2'
	PROD_DICT = {}
	SPEC_LIST, ASIN_LIST, TITLE_LIST = [], [], []
	prod_cursor = mongo.searchInDB(mongoObj.getProductKey(), db_col=db_col)
	docCount = prod_cursor.count()
	print("make mainCat spec feature from %s products..." % (docCount))
	with tqdm(total=docCount) as pbar:
		for i1, prod in enumerate(prod_cursor):
			asin, title, description, big_categories, small_categories, salesRank, feature = prod["asin"], prod[
				"title"], prod["description"], prod["category1"], prod["category2"], prod["salesRank"], prod["feature"]
			if title == "": continue
			# spec = lemm_sent_process(spec, remove_stopwords=True, summary=False, mode="spacy", withdot=True)
			if "(new Date()).getTime();" in title: continue
			DATA_DICT = {}
			DATA_DICT["description"] = description
			DATA_DICT["title"] = title
			DATA_DICT["category1"] = big_categories
			DATA_DICT["category2"] = small_categories
			if salesRank == "": salesRank = '0'
			DATA_DICT["salesRank"] = int(salesRank)
			DATA_DICT["feature"] = feature
			PROD_DICT[asin] = DATA_DICT
			pbar.set_description("%s PROD_DICT" % (main_cat))
			pbar.update(1)

		# Default DICT
		DATA_DICT = {}
		DATA_DICT["description"] = description
		DATA_DICT["title"] = title
		DATA_DICT["category1"] = big_categories
		DATA_DICT["category2"] = small_categories
		if salesRank == "": salesRank = '0'
		DATA_DICT["salesRank"] = int(salesRank)
		DATA_DICT["feature"] = feature
		PROD_DICT['UNK'] = DATA_DICT

	return PROD_DICT


PROD_DICT = loadProdSpecData()

# Load Review From Mongo
import re
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

nlp = en_core_web_sm.load()


def create_custom_tokenizer(nlp):
	prefix_re = re.compile(r'[0-9]\.')
	return Tokenizer(nlp.vocab, prefix_search=prefix_re.search)


nlp.tokenizer = create_custom_tokenizer(nlp)


def loadProdReviewData(PROD_DICT):
	global main_cat, cond_date
	rev_db_col = 'new_reviews2'
	review_cursor = mongo.searchInDB(mongoObj.getReviewKey(), db_col=rev_db_col)
	reviewCount = review_cursor.count()
	print("make mainCat reviews feature from %s reviews..." % (reviewCount))
	print("Search reviews finished...")

	with tqdm(total=reviewCount) as pbar:
		for i2, rev in enumerate(review_cursor):
			asin, review, overall, vote, summary, review_ID = \
				rev["asin"], rev["reviewText"], rev['overall'], rev['vote'], rev['summary'], str(rev['unixReviewTime'])
			if asin not in PROD_DICT.keys(): asin = "UNK"

			if "REVIEW_ITEM_LIST" not in PROD_DICT[asin].keys():
				DATA_DICT = PROD_DICT[asin]
				DATA_DICT["REVIEW_ITEM_LIST"] = []

			DATA_DICT = PROD_DICT[asin]
			REVIEW_ITEM_LIST = DATA_DICT["REVIEW_ITEM_LIST"]

			# ----------------------------------------------------------------------------------------------
			# print("\n\n")
			# print(review)
			# print("\n\n\n\n")
			review = re.sub(r'http\S+', '', review)
			review = remove_word4(review)
			lemm_review = lemm_sent_process4(review, remove_stopwords=False, summary=False, mode="spacy",
											 withdot=True)
			lemm_review_len = len(lemm_review.split(" "))

			lemm_review = lemm_review.split(" .")
			lemm_review = [line + " . " for line in lemm_review if len(line) > 2]

			# print("\n\n\n\n")
			# [print(line)  for line in lemm_review]
			# break
			# ---------------------------------------------------------------------------------------------
			# print("\n\n")
			# summary = '''
			# Light weight and smaller size = Fantastic!
			# '''
			# print(summary)
			# print("\n\n\n\n")
			summary = re.sub(r'http\S+', '', summary)
			summary = remove_word4(summary)

			lemm_summary = lemm_sent_process4(summary, remove_stopwords=False, summary=True, mode="spacy",
											  withdot=False)
			lemm_summary_len = len(lemm_summary.split(" "))

			# print("\n\n\n\n")
			# print(lemm_summary)
			# break
			# ----------------------------------------------------------------------------------------------

			item_dict = {'review_ID': review_ID, "review": review, "overall": overall, "vote": vote, 'summary': summary,
						 'lemm_review': lemm_review, 'lemm_review_len': lemm_review_len,
						 'lemm_summary': lemm_summary, 'lemm_summary_len': lemm_summary_len}

			REVIEW_ITEM_LIST.append(item_dict)
			DATA_DICT["REVIEW_ITEM_LIST"] = REVIEW_ITEM_LIST
			PROD_DICT[asin] = DATA_DICT
			pbar.set_description("%s REVIEW_ITEM_LIST " % (main_cat))
			pbar.update(1)

		DEL_ASIN = []
		for asin, DATA_DICT in PROD_DICT.items():
			if "REVIEW_ITEM_LIST" not in DATA_DICT: DEL_ASIN.append(asin)

		for asin in DEL_ASIN:
			del PROD_DICT[asin]

		return PROD_DICT


PROD_DICT = loadProdReviewData(PROD_DICT)

# pickle 保存
import pickle

if not os.path.exists("pickle/main_cat_data"): os.makedirs('pickle/main_cat_data')
# pickle a variable to a file
fn = 'pickle/main_cat_data/%s.pickle' % (main_cat)
# fn = fn.replace(' ','-')
file = open(fn, 'wb')
pickle.dump(PROD_DICT, file)
file.close()






