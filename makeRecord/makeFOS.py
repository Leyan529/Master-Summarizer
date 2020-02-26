# import warnings
# warnings.simplefilter('ignore')
# warnings.filterwarnings('ignore')
# warnings.filterwarnings(action='once')

from data_util.product import *
from data_util.MongoDB import *
from data_util.stopwords import *
from data_util.preprocess import *
from data_util.eda import *

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

# from spacy.symbols import cop, acomp, amod, conj, neg, nn, nsubj, dobj,prep,advmod
# from spacy.symbols import VERB, NOUN, PROPN, ADJ, ADV, AUX, PART

pattern_counter = collections.Counter()

# from nltk.corpus import stopwords
# import networkx as nx
# from MongoDB import MongoDB
import os
import pickle
# from eda import *
from glob import glob

# from product import *

# pickle 提取
# _, category1, category2, _ = DVD_Player().getAttr()
_, category1, category2, _ = Cameras().getAttr()
# _,category1,category2,_ = Cell_Phones().getAttr()
# _,category1,category2,_ = GPS().getAttr()
# _, category1, category2, _ = Keyboards().getAttr()

glob('pickle/data/new_%s_%s.pickle' % (category1, category2))

# reload a file to a variable
with open(glob('pickle/data/new_%s_%s.pickle' % (category1, category2))[0], 'rb') as file:
	print(file.name)
	PROD_DICT = pickle.load(file)
	category1, category2 = file.name.replace(".pickle", "").split("_")[1:]

# 提取所有規格關鍵字
import warnings
from tqdm import tqdm
from collections import Counter, OrderedDict

feature_counter1 = Counter()
feature_counter2 = Counter()
feature_counter3 = Counter()
warnings.filterwarnings('ignore')

import collections

freq_words = collections.Counter()
feat_words = collections.Counter()
opinion_words = collections.Counter()
cooccurs_words = collections.Counter()

_Sents = []
keywords_list = []
# 其中 freq_words 是單字出現的頻率 , _Sents 是文章中所有的句子

i = 0

# total_keywords = []
Products = len(PROD_DICT.items())
print("%s Products\n" % (Products))

if not os.path.exists('FOP-View'):
	os.makedirs('FOP-View')

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

nlp = en_core_web_sm.load()


def create_custom_tokenizer(nlp):
	prefix_re = re.compile(r'[0-9]\.')
	return Tokenizer(nlp.vocab, prefix_search=prefix_re.search)


nlp.tokenizer = create_custom_tokenizer(nlp)

Reviews = 0
fn = 'FOP-View/%s_%s_keywords.txt' % (category1, category2)
if not os.path.exists(fn):
	print("Start get Product keywords...")
	with open('FOP-View/Prod_%s_%s.txt' % (category1, category2), "w", encoding="utf-8") as f:
		with tqdm(total=Products) as pbar:
			for asin, DATA_DICT in PROD_DICT.items():
				REVIEW_ITEM_LIST = DATA_DICT['REVIEW_ITEM_LIST']
				title = DATA_DICT['title']
				description = DATA_DICT['description']
				feature = DATA_DICT['feature']
				dash_keywords, noun_keywords, newkeywords, newdescription = getKeywordFeat(description, feature)
				#                 print(noun_keywords)
				f.write("* asin:" + asin + "\n")
				if "(new Date()).getTime();" in title:
					f.write("title:" + asin + "\n")
				else:
					f.write("title:" + title + "\n")
				f.write('dash_keywords:\n' + str(dash_keywords) + "\n");
				f.write('noun_keywords2:\n' + str(noun_keywords) + "\n")

				f.write('* keywords:\n' + str(newkeywords) + "\n")

				f.write('************************************************************************' + "\n")
				f.write('************************************************************************' + "\n")
				for pair in REVIEW_ITEM_LIST:
					review_ID = pair["review_ID"]
					review = pair["review"]
					lemm_review = pair["lemm_review"]  # 已斷句抽取
					cc_lemm_review = "".join(lemm_review)
					vote = pair["vote"]
					summary = pair["summary"]
					pbar.set_description(
						"%s, %s Product : %s , keywords Processing ID %s" % (category1, category2, asin, review_ID))

					dash_keywords2, noun_keywords2, newkeywords2, _ = getKeywordFeat_2(cc_lemm_review)
					dash_keywords3, noun_keywords3, newkeywords3, _ = getKeywordFeat_2(summary)
					feature_counter2.update(newkeywords2)
					feature_counter3.update(newkeywords3)
					Reviews += 1

				feature_counter1.update(newkeywords)
				pbar.update(1)
			pbar.close()

	important_features = OrderedDict(sorted(feature_counter1.items(), key=lambda pair: pair[1], reverse=True))
	important_features = [(word, important_features[word]) for word in important_features if
						  important_features[word] > 5]
	print("Count : %s" % (len(important_features)))

	important_features2 = OrderedDict(sorted(feature_counter2.items(), key=lambda pair: pair[1], reverse=True))
	important_features2 = [(word, important_features2[word]) for word in important_features2 if
						   important_features2[word] > 5]
	print("Count : %s" % (len(important_features2)))

	important_features3 = OrderedDict(sorted(feature_counter3.items(), key=lambda pair: pair[1], reverse=True))
	important_features3 = [(word, important_features3[word]) for word in important_features3 if
						   important_features3[word] > 5]
	print("Count : %s" % (len(important_features3)))

	fn = 'FOP-View/%s_%s_keywords.txt' % (category1, category2)
	with open(fn, "w", encoding="utf-8") as f:
		total_keywords = set()
		for word, v in important_features:
			f.write("%s:%s \n" % (word, v))
			if v > 20: total_keywords.add(word)

	fn2 = 'FOP-View/%s_%s_keywords2.txt' % (category1, category2)
	with open(fn2, 'w', encoding="utf-8") as f:
		total_keywords2 = set()
		for word, v in important_features2:
			f.write("%s:%s \n" % (word, v))
			total_keywords2.add(word)

	fn3 = 'FOP-View/%s_%s_keywords3.txt' % (category1, category2)
	with open(fn3, 'w', encoding="utf-8") as f:
		total_keywords3 = set()
		for word, v in important_features3:
			f.write("%s:%s \n" % (word, v))
			total_keywords3.add(word)
	total_keywords = total_keywords2
else:
	fn = 'FOP-View/%s_%s_keywords2.txt' % (category1, category2)
	print('load %s keywords...' % (fn))
	total_keywords = set()
	with open(fn, 'r', encoding='utf-8') as f:
		lines = f.readlines()
		for line in lines:
			k, v = line.split(":")
			total_keywords.add(k)

# 計算FO-PAIR以及句子抽取
import pandas as pd

i = 0

if not os.path.exists('XLSX'):
	os.makedirs('XLSX')

if not os.path.exists('FOP-View/result'):
	os.makedirs('FOP-View/result')

key_FOP_Sent_df = {}
key_FOP_df = {}
i = 0
j = 0
if Reviews == 0: Reviews = len(PROD_DICT.items())
with tqdm(total=Reviews) as pbar:
	with open("FOP-View/result/%s_%s_result.txt" % (category1, category2), 'w', encoding="utf-8") as result:
		for asin, DATA_DICT in PROD_DICT.items():
			REVIEW_ITEM_LIST = DATA_DICT['REVIEW_ITEM_LIST']
			title = DATA_DICT['title']
			big_categories, small_categories = DATA_DICT["category1"], DATA_DICT["category2"]
			#             if "(new Date()).getTime();" in title : continue
			description = DATA_DICT['description']
			feature = DATA_DICT['feature']
			if "(new Date()).getTime();" not in title:
				result.write("title:" + title + "\n")
			else:
				result.write("title:" + asin + "\n")
			if Reviews == len(PROD_DICT.items()):
				pbar.update(1)
				pbar.set_description(
					"%s %s Product : %s " % (category1, category2, asin))
			result.write('************************************************************************' + "\n")
			for pair in REVIEW_ITEM_LIST:
				review = pair["review"]
				rev_dash_keywords, newreview = cleanReview(review)

				vote = pair["vote"]
				overall = pair["overall"]
				summary = pair["summary"]
				lemm_review = pair["lemm_review"]  # 已斷句抽取
				lemm_summary = pair["lemm_summary"]
				lemm_review_len = pair['lemm_review_len']
				lemm_summary_len = pair['lemm_summary_len']

				review_ID = pair["review_ID"]
				if Reviews != len(PROD_DICT.items()):
					pbar.update(1)
					pbar.set_description(
						"%s %s Product : %s , FOP Processing ID %s  " % (category1, category2, asin, review_ID))

				cc_lemm_review = "".join(lemm_review)
				rev_token_set = set(nltk.word_tokenize(cc_lemm_review))
				summ_token_set = set([str(token) for token in nlp(lemm_summary)])
				# if (rev_token_set & summ_token_set) < 10: continue # for cheat data set

				if len(summ_token_set & total_keywords) == 0: continue
				result.write('overall:' + str(overall) + "\n")
				result.write('review:\n' + review + "\n")
				result.write('summary:\n' + summary + "\n")
				result.write("************************************************\n")

				keyword_list = []
				key_fop_pair_row = {}
				# extract_sents = sentence_extract_blob(lemm_review)
				# extract_sents = nltk.sent_tokenize(lemm_review)

				FOP_sents = []  # 儲存帶有FOP的句子
				total_mention_features = []
				for lemm_sent in lemm_review:
					fop_sent_row = {}  # fop train row
					sentiment = TextBlob(lemm_sent).sentences[0].sentiment

					# review token_set
					token_set = set(nltk.word_tokenize(lemm_sent))
					mention_features = list(token_set & total_keywords)
					total_mention_features.extend(mention_features)

					POS_fops = FO_rule_POS(lemm_sent).run()
					DEP_fops = FOP_rule_Depend(lemm_sent).run()
					POS_fops = [(f, o) for f, o in POS_fops if f in total_keywords]
					DEP_fops = [(f, o) for f, o in DEP_fops if f in total_keywords]
					#  PMI_fops,cand_pairs_score = pmi_fopair(lemm_sent) # 皆為空有問題

					result.write("************************************************\n")
					result.write('lemm_sent:\n' + lemm_sent);
					fop_sent_row['review_ID'] = review_ID
					fop_sent_row['overall'] = overall
					fop_sent_row['lemm_sent'] = lemm_sent

					result.write('mention features:' + str(mention_features) + "\n")
					result.write("polarity: %s subjectivity: %s \n" % (sentiment.polarity, sentiment.subjectivity))
					fop_sent_row['polarity'] = sentiment.polarity;
					fop_sent_row['subjectivity'] = sentiment.subjectivity
					fop_sent_row['mention_features'] = ",".join(mention_features)

					result.write("POS_fops:" + str(POS_fops) + "\n");
					# pattern_text.write("POS_fops:" + str(POS_fops) + "\n")
					result.write("DEP_fops:" + str(DEP_fops) + "\n");
					# pattern_text.write("DEP_fops:" + str(DEP_fops) + "\n")
					#                     f.write("PMI_fops:"+str(cand_pairs_score)+"\n")
					#                     print(cand_pairs_score)
					result.write("************************************************\n")
					if "not" in token_set: continue
					if overall > 3 and sentiment.polarity < 0: continue
					if overall < 3 and sentiment.polarity > 0: continue
					if (len(POS_fops) == 0) and (len(DEP_fops) == 0): continue
					FOP_sents.append(lemm_sent)
					#                     fop_list = set(PMI_fops + POS_fops + DEP_fops)
					fop_list = set(POS_fops + DEP_fops)
					fop_prob_list = []

					for fop in fop_list:
						p1, p2, p3 = 0, 0, 0;
						y1, y2, y3 = 0.6, 0.2, 0.2;
						feat, opinion = fop
						#                         if fop in PMI_fops:
						#                             p1 = pmi(feat,opinion)
						if fop in POS_fops: p2 = 1
						if fop in DEP_fops: p3 = 1
						#                         p = y1*p1 + y2*p2 + y3*p3
						p = y2 * p2 + y3 * p3
						fop_prob_list.append((fop, p))
					result.write("fop_prob_list:" + str(fop_prob_list) + "\n")
					i += 1
					result.write("\n");
					# ------------------------------------------------------
					fop_str = ""
					for idx, fop_score in enumerate(fop_prob_list):
						fop, score = fop_score
						feat, o = fop
						if idx == len(fop_prob_list) - 1:
							fop_str += "%s %s" % (feat, o)
						else:
							fop_str += "%s %s," % (feat, o)
					fop_sent_row["fops"] = fop_str
					keyword_list.append(fop_str)
					fop_sent_row['review_ID'] = review_ID
					fop_sent_row['summary'] = lemm_summary

					if abs(sentiment.polarity) <= 0.5: continue
					if abs(sentiment.subjectivity) < 0.4: continue
					if len(mention_features) == 0: continue

					key_FOP_Sent_df[i] = fop_sent_row;
					i += 1

				total_keyword = " ".join(keyword_list)
				key_fop_pair_row['review_ID'] = review_ID
				key_fop_pair_row['review'] = review
				key_fop_pair_row['summary'] = summary
				key_fop_pair_row['big_categories'] = big_categories
				key_fop_pair_row['small_categories'] = small_categories
				key_fop_pair_row['lemm_review'] = cc_lemm_review
				key_fop_pair_row['lemm_summary'] = lemm_summary
				key_fop_pair_row['lemm_review_len'] = lemm_review_len
				key_fop_pair_row['lemm_summary_len'] = lemm_summary_len
				key_fop_pair_row['overall'] = overall
				key_fop_pair_row['vote'] = vote
				key_fop_pair_row['total_keyword'] = total_keyword
				key_fop_pair_row['FOP_sents'] = "\n".join(FOP_sents)
				key_fop_pair_row['total_mention_features'] = " ".join(total_mention_features)
				key_FOP_df[j] = key_fop_pair_row;
				j += 1
			# if j > 20: break
			#                 f.write("---------------------------------------------------\n")
	pbar.close()

if not os.path.exists('XLSX/category'):
	os.makedirs('XLSX/category')

print("finished...")
# training_data.close()
key_FOP_Sent_df = pd.DataFrame.from_dict(key_FOP_Sent_df, orient='index')
key_FOP_df = pd.DataFrame.from_dict(key_FOP_df, orient='index')

# topic-to-eaasy DataSet 建立
csv_path = "XLSX/category/%s_%s_key_FOP_Sent.xlsx" % (category1, category2)
# df.to_csv(csv_path) #默认dt是DataFrame的一个实例，参数解释如下
key_FOP_Sent_df.to_excel(csv_path, encoding='utf8')
print(csv_path + " Write finished")
print("%s Write finished" % (len(key_FOP_Sent_df)))
key_FOP_Sent_df.head()

# Key word Attention DataSet 建立
csv_path = "XLSX/category/%s_%s_key.xlsx" % (category1, category2)
# df.to_csv(csv_path) #默认dt是DataFrame的一个实例，参数解释如下
key_FOP_df.to_excel(csv_path, encoding='utf8')
print(csv_path + " Write finished")
print("%s Write finished" % (len(key_FOP_df)))
