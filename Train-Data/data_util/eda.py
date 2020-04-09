# import warnings
# warnings.simplefilter('ignore')
# warnings.filterwarnings('ignore')
# warnings.filterwarnings(action='once')


import spacy

from data_util.product import *
from data_util.stopwords import *
from data_util.preprocess import *

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

from spacy.symbols import cop, acomp, amod, conj, neg, nn, nsubj, dobj,prep,advmod
from spacy.symbols import VERB, NOUN, PROPN, ADJ, ADV, AUX, PART

pattern_counter = collections.Counter()

from nltk.corpus import stopwords
# import networkx as nx
import os
import pickle

# 斷詞辭典
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
stpwords_list3 = [f.replace("\n","") for f in open("data_util/stopwords.txt","r",encoding = "utf-8").readlines()]
stpwords_list3.remove("not")
stopwords = list(html_escape_table + stpwords_list2) + list(list(stops) + list(stpwords_list1) + list(stpwords_list3))
print("斷詞辭典 已取得")

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

# 句子篩選/切割
import textacy
from summa import keywords,summarizer

def sentence_extract_blob(text):    
    extract_sents = []
    text = text.replace(",","").replace("i.e.","")
    text = SentProcess(text)
    
    for sent in text.split("<end>"):        
#         print(sent)
        if sent == "": continue
#         sentiment = TextBlob(sent).sentences[0].sentiment
#         if abs(sentiment.polarity) <= 0.5: continue
#         if abs(sentiment.subjectivity) <= 0.5: continue
#         print(TextBlob(sent).sentences)
#         print("polarity : ",sentiment.polarity)
#         print("subjectivity : ",sentiment.subjectivity)
# #         print(keywords.keywords(sent))
#         print("***")
        extract_sents.append(sent)
    return extract_sents

def extract_cand_pharse(text): 
    sent_pattern = r"""( 
    (<NOUN><NOUN> | <ADJ><NOUN> | <NOUN><NOUN><NOUN> | <NOUN><ADP><DET><NOUN> | 
    <NOUN> | <ADJ><NOUN><NOUN> | <ADJ><ADJ><NOUN> | <NOUN><PREP><NOUN> | <NOUN><PREP><DET><NOUN> | 
    <VERB><PUNCT><ADP><NOUN><NOUN> | <DET><NOUN> )


    (<AUX><ADV><ADJ>|<AUX><ADJ>|<ADV><ADJ>|<ADJ><NOUN>) 
    )
    """
#     pattern = r'<PROPN>+ (<PUNCT|CCONJ> <PUNCT|CCONJ>? <PROPN>+)*'
    extract_pharse = []
    doc = textacy.make_spacy_doc(text,lang='en_core_web_sm')
    phrases = textacy.extract.pos_regex_matches(doc, sent_pattern)
    # Print all Verb Phrase

    for phrase in phrases:
#         print("verb_phrases : " , phrase.text)
    #     print([(token.text,token.pos_) for token in nlp(chunk.text)])
        extract_pharse.append(phrase.text+"\n")           
    return extract_pharse 

# 規格關鍵字提取
def getKeywordFeat(description,feature,get_new_description = False):  # 搜索關鍵保留字，並重新清理句子       
    description_list = []
    for sent in description:
        if len(TextBlob(sent)) > 1:
            sent_list = sent.split(".")
            description_list.extend(sent_list) 
        else:
            description_list.extend(sent) 
    description_list = [sent.strip() +"." for sent in description_list]
    if type(feature) == str:
        total_desc_sents = description_list + [feature]
    else:
        total_desc_sents = description_list + feature    
 
    # Stage 1 取dash feature
    keywords = []
    newdescription = ""
    
    for sent in total_desc_sents:
        keys = []
        for token in nlp(sent):            
            sent = sent.lower()
            for k, v in contractions.items():
                if k in sent:
                    sent = sent.replace(k, v)

            for k in html_escape_table:
                if k in sent:
                    sent = sent.replace(k, "")            
            
        pattern = re.compile(r"([\d\w\.-]+[-'//.][\d\w\.-]+)")  # |  ([\(](\w)+[\)])
        keys = pattern.findall(sent)            
            
        keywords.extend(keys)
        cleansent = sent
        for k in keys:    
            k = k.replace("(","").replace(")","")
            cleansent = cleansent.replace(k,"(" + k + ")")
            cleansent = cleansent.replace("((","(").replace("))",")")     
            cleansent = cleansent.replace("(" + k + ")","") # 移除符號特徵
            cleansent = cleansent.strip()
        
#         cleansent = cleansent + ".\n"
        cleansent = remove_word3(cleansent)
        char = " "
        while char * 2 in cleansent:
            cleansent = cleansent.replace(char * 2, char)  
        char = "."
        while char * 2 in cleansent:
            cleansent = cleansent.replace(char * 2, char)            
         
        if len(cleansent) < 5 : continue
        newdescription = newdescription + cleansent + ".\n"
        
    newdescription = newdescription.replace("..",".")   
    newdescription = re.sub(r"\([\w]+\)","",newdescription)
    newdescription = re.sub(r"\(\)","",newdescription)   
    newdescription = re.sub(r'\"',"",newdescription)
    
    # get Noun pharse keyword for newdescription
#     for pharse in nlp(newdescription).noun_chunks:
#         print(pharse.text)
    keywords2 = PF_rule_POS(newdescription).run()
#     print('cand_pf:',cand_pf)
#     keywords2 = set()
#     for pf in cand_pf:
#         chunk_pfs = nltk.word_tokenize(pf)
#         for tok in chunk_pfs:
#             if tok in stopwords: continue
#             if tok in opinion_lexicon["total-words"]: continue
#             keywords2.add(tok)
#     print('keywords2:',keywords2)

#     print(len(keywords2),keywords2)
    newkeywords = list()
    for key in list(keywords) + list(keywords2):
        clean_key = remove_word2(key)
        clean_key = clean_key.split(" ")            
        newkeywords.extend(clean_key) 
    newkeywords = list(set([key for key in newkeywords if len(key) >=2 and key not in stopwords and key not in opinion_lexicon["total-words"]]))
    
    return keywords,keywords2,newkeywords,newdescription


def getKeywordFeat_2(article):  # 搜索關鍵保留字，並重新清理句子       
   
    total_desc_sents = sentence_extract_blob(article)
    # Stage 1 取dash feature
    keywords = []
    newarticle = ""
    
    for sent in total_desc_sents:
        keys = []
        for token in nlp(sent):            
            sent = sent.lower()
            for k, v in contractions.items():
                if k in sent:
                    sent = sent.replace(k, v)

            for k in html_escape_table:
                if k in sent:
                    sent = sent.replace(k, "")            
            
        pattern = re.compile(r"([\d\w\.-]+[-'//.][\d\w\.-]+)")  # |  ([\(](\w)+[\)])
        keys = pattern.findall(sent)            
            
        keywords.extend(keys)
        cleansent = sent
        for k in keys:    
            k = k.replace("(","").replace(")","")
            cleansent = cleansent.replace(k,"(" + k + ")")
            cleansent = cleansent.replace("((","(").replace("))",")")     
            cleansent = cleansent.replace("(" + k + ")","") # 移除符號特徵
            cleansent = cleansent.strip()
        
#         cleansent = cleansent + ".\n"
        cleansent = remove_word3(cleansent)
        char = " "
        while char * 2 in cleansent:
            cleansent = cleansent.replace(char * 2, char)  
        char = "."
        while char * 2 in cleansent:
            cleansent = cleansent.replace(char * 2, char)            
         
        if len(cleansent) < 5 : continue
        newarticle = newarticle + cleansent + ".\n"
        
    newarticle = newarticle.replace("..",".")   
    newarticle = re.sub(r"\([\w]+\)","",newarticle)
    newarticle = re.sub(r"\(\)","",newarticle)   
    newarticle = re.sub(r'\"',"",newarticle)
    
    # get Noun pharse keyword for newarticle
#     for pharse in nlp(newarticle).noun_chunks:
#         print(pharse.text)
    keywords2 = PF_rule_POS(newarticle).run()
#     print(cand_pf)
    
#     keywords2 = set()
#     for pf in cand_pf:
#         chunk_pfs = nltk.word_tokenize(pf)
#         for tok in chunk_pfs:
#             if tok in stopwords: continue
#             if tok in opinion_lexicon["total-words"]: continue
#             keywords2.add(tok)
            
    newkeywords = list()
    for key in list(keywords) + list(keywords2):
        clean_key = remove_word2(key)
        clean_key = clean_key.split(" ")            
        newkeywords.extend(clean_key) 
    newkeywords = list(set([key for key in newkeywords if len(key) >=2 and key not in stopwords and key not in opinion_lexicon["total-words"]]))
	
#     newkeywords = list(set([key for key in newkeywords if len(key) >=2]))

#     print(len(keywords2),keywords2)
    return keywords,keywords2,newkeywords,newarticle


# 評論濾除符號-英文特徵
def cleanReview(review):  # 搜索關鍵保留字，並重新清理句子     
    # Stage 1 取dash feature
    keywords = []
    newReview = ""
    for sent in nltk.sent_tokenize(review):
        for token in nlp(sent):
            sent = sent.lower()
            for k, v in contractions.items():
                if k in sent:
                    sent = sent.replace(k, v)

            for k in html_escape_table:
                if k in sent:
                    sent = sent.replace(k, "")
            
            
            pattern = re.compile(r"([\d\w\.-]+[-'//.][\d\w\.-]+)")  # |  ([\(](\w)+[\)])
            keys = pattern.findall(sent)
            
            
        keywords.extend(keys)
        cleansent = sent
        for k in keys:    
            k = k.replace("(","").replace(")","")
            cleansent = cleansent.replace(k,"(" + k + ")")
            cleansent = cleansent.replace("((","(").replace("))",")")     
            cleansent = cleansent.replace("(" + k + ")","") # 移除符號特徵
            cleansent = cleansent.strip()
        
#         cleansent = cleansent + ".\n"
        cleansent = remove_word3(cleansent)
        char = " "
        while char * 2 in cleansent:
            cleansent = cleansent.replace(char * 2, char)  
        char = "."
        while char * 2 in cleansent:
            cleansent = cleansent.replace(char * 2, char)            
         
        if len(cleansent) < 5 : continue
        newReview = newReview + cleansent + ".\n"
        
    newReview = newReview.replace("..",".")   
    newReview = re.sub(r"\([\w]+\)","",newReview)
    newReview = re.sub(r"\(\)","",newReview)   
    newReview = re.sub(r'\"',"",newReview)
    
    # get Noun pharse keyword for newdescription
#     cand_pf = PF_rule_POS(newReview).run()
    
#     keywords2 = set()
#     for pf in cand_pf:
#         chunk_pfs = nltk.word_tokenize(pf)
#         for tok in chunk_pfs:
#             if tok in stopwords: continue
#             if tok in opinion_lexicon["total-words"]: continue
#             keywords2.add(tok)

#     print(len(keywords2),keywords2)
    return keywords, newReview

# PF-Extraction-Rule(POS)
# -NN, 
# -NN NN, 
# JJ NN 
# -NN NN NN, 
# JJ NN NN, 
# JJ JJ NN, 
# NN IN NN 
# -NN IN DT NN

class PF_rule_POS():
    def __init__(self, article):
        self.article = article
        self.matched_sents = []  # Collect data of matched sentences to be visualized

    def collect_sents(self, matcher, doc, i, matches):
        match_id, start, end = matches[i]
        span = doc[start:end]  # Matched span
        sent = span.sent  # Sentence containing matched span
        # Append mock entity for match in displaCy style to matched_sents
        # get the match span by ofsetting the start and end of the span with the
        # start and end of the sentence in the doc
        match_ents = [{
            "start": span.start_char - sent.start_char,
            "end": span.end_char - sent.start_char,
            "label": "MATCH",
        }]
        self.matched_sents.append({"text": sent.text, "ents": match_ents})

    def match_pattern(self, sent, flit_keyword=None):
        res = []
        #         ('Dolby', 'PROPN'), ('Digital', 'PROPN')
        matcher = Matcher(nlp.vocab)
        #         matcher.add("pf1", self.collect_sents, [{'POS': {"IN": ['NOUN', 'PROPN']}}, {'IS_PUNCT': True},
        #                                                    {'POS': {"IN": ['NOUN', 'PROPN']}}])  # add pattern
        #         matcher.add("pf2", self.collect_sents,
        #                     [{'POS': {"IN": ['NOUN', 'PROPN']}}, {'IS_PUNCT': True}, {'POS': {"IN": ['NOUN', 'PROPN']}},
        #                      {'POS': 'SYM'}, {'POS': {"IN": ['NOUN', 'PROPN']}}])  # add pattern
        #         #         matcher.add("specn3", self.collect_sents, [{'POS': 'ADJ'}, {'POS': 'NOUN'}])  # add pattern
        #         matcher.add("pf3", self.collect_sents,
        #                     [{'POS': {"IN": ['NOUN', 'PROPN']}}, {'POS': {"IN": ['NOUN', 'PROPN']}}])  # add pattern
        #         matcher.add("pf4", self.collect_sents,
        #                     [{'POS': {"IN": ['NOUN', 'PROPN']}}, {'POS': {"IN": ['NOUN', 'PROPN']}},
        #                      {'POS': {"IN": ['NOUN', 'PROPN']}}])  # add pattern
        #         matcher.add("pf5", self.collect_sents, [{'POS': 'ADJ'}, {'POS': {"IN": ['NOUN', 'PROPN']}},
        #                                                    {'POS': {"IN": ['NOUN', 'PROPN']}}])  # add pattern
        #         matcher.add("pf6", self.collect_sents,
        #                     [{'POS': 'ADJ'}, {'POS': 'ADJ'}, {'POS': {"IN": ['NOUN', 'PROPN']}}])  # add pattern
        #         #         ('inches', 'NOUN'), ('(', 'PUNCT'), ('W', 'NOUN'), ('x', 'SYM'), ('H', 'NOUN'), ('x', 'SYM'), ('D', 'NOUN'), (')', 'PUNCT')
        #         matcher.add("pf6.1", self.collect_sents,
        #                     [{'POS': {"IN": ['NOUN', 'PROPN']}}, {'IS_PUNCT': True}, {'POS': {"IN": ['NOUN', 'PROPN']}},
        #                      {'POS': 'SYM'}, {'POS': {"IN": ['NOUN', 'PROPN']}}, {'POS': 'SYM'},
        #                      {'POS': {"IN": ['NOUN', 'PROPN']}}, {'IS_PUNCT': True}])  # inches (W x H x D)
        #         matcher.add("pf6.2", self.collect_sents,
        #                     [{'POS': {"IN": ['NOUN', 'PROPN']}}, {'IS_PUNCT': True}, {'POS': {"IN": ['NOUN', 'PROPN']}},
        #                      {'POS': 'SYM'}, {'POS': {"IN": ['NOUN', 'PROPN']}}, {'IS_PUNCT': True}])  # inches (W x H)
        #         matcher.add("pf7", self.collect_sents, [{'POS': {"IN": ['NOUN', 'PROPN']}}, {'POS': 'PREP'},
        #                                                    {'POS': {"IN": ['NOUN', 'PROPN']}}])  # add pattern
        #         matcher.add("pf8", self.collect_sents, [{'POS': {"IN": ['NOUN', 'PROPN']}}, {'POS': 'PREP'}, {'POS': 'DET'},
        #                                                    {'POS': {"IN": ['NOUN', 'PROPN']}}])  # add pattern
        #         matcher.add("pf9", self.collect_sents,
        #                     [{'POS': 'VERB'}, {'POS': 'PUNCT'}, {'POS': 'ADP'}, {'POS': {"IN": ['NOUN', 'PROPN']}},
        #                      {'POS': {"IN": ['NOUN', 'PROPN']}}])  # add pattern

        matcher.add("pf1", self.collect_sents, [{'TAG': {"IN": ['NN']}}])  # add pattern
        matcher.add("pf2", self.collect_sents, [{'TAG': {"IN": ['NN']}}, {'TAG': {"IN": ['NN']}}])  # add pattern
        matcher.add("pf3", self.collect_sents, [{'TAG': {"IN": ['JJ']}}, {'TAG': {"IN": ['NN']}}])  # add pattern    
        matcher.add("pf4", self.collect_sents,
                    [{'TAG': {"IN": ['NN']}}, {'TAG': {"IN": ['NN']}}, {'TAG': {"IN": ['NN']}}])  # add pattern
        matcher.add("pf5", self.collect_sents,
                    [{'TAG': {"IN": ['JJ']}}, {'TAG': {"IN": ['NN']}}, {'TAG': {"IN": ['NN']}}])
        matcher.add("pf6", self.collect_sents,
                    [{'TAG': {"IN": ['JJ']}}, {'TAG': {"IN": ['JJ']}}, {'TAG': {"IN": ['NN']}}])
        matcher.add("pf7", self.collect_sents,
                    [{'TAG': {"IN": ['NN']}}, {'TAG': {"IN": ['IN']}}, {'TAG': {"IN": ['NN']}}])
        matcher.add("pf8", self.collect_sents, [{'TAG': {"IN": ['NN']}}, {'TAG': {"IN": ['IN']}},
                                                {'TAG': {"IN": ['DT']}}, {'TAG': {"IN": ['NN']}}])

        doc = nlp(sent)
        matches = matcher(doc)
        # Serve visualization of sentences containing match with displaCy
        # set manual=True to make displaCy render straight from a dictionary
        # (if you're not running the code within a Jupyer environment, you can
        # use displacy.serve instead)
        #         displacy.render(self.matched_sents, style="ent", manual=True)
        for match_id, start, end in matches:
            # Get the string representation
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]  # The matched span
            span_N = [w.text for w in span if w.tag_.startswith('N') and (w.text not in stopwords) and (w.text not in opinion_lexicon["total-words"])]

            #             print(match_id, string_id, start, end, span.text)
#             if flit_keyword:
#                 found = False
#                 spanStr = span.text
#                 for word in spanStr.split(" "):
#                     if word in flit_keyword: found = True; break
#                 if found: res.append(span.text)
#             else:
#                 res.append(span.text)
#             res.append(span.text)
#             print(span.text)
#             print(span_N)
            res.extend(span_N)
        return res

    def run(self):
        total_res = []
        for sent in nltk.sent_tokenize(self.article):
            sent = sent.lower()
            sent = sent.replace(",", " ").replace(":", " ").replace("(", " ").replace(")", " ").replace("multi-",
                                                                                                        "multiple ")
            ress = self.match_pattern(sent)
            total_res.extend(ress)
#         total_res = list(set(total_res))
        #         print("final res",total_res)
        return set(total_res)

# FO-Rule(POS)
class FO_rule_POS():
    def __init__(self, article):
        self.article = article
        self.matched_sents = []  # Collect data of matched sentences to be visualized

    def collect_sents(self, matcher, doc, i, matches):
        match_id, start, end = matches[i]
        span = doc[start:end]  # Matched span
        sent = span.sent  # Sentence containing matched span
        # Append mock entity for match in displaCy style to matched_sents
        # get the match span by ofsetting the start and end of the span with the
        # start and end of the sentence in the doc
        match_ents = [{
            "start": span.start_char - sent.start_char,
            "end": span.end_char - sent.start_char,
            "label": "MATCH",
        }]
        self.matched_sents.append({"text": sent.text, "ents": match_ents})

    def match_pattern(self, sent, flit_keyword=None):
        res = []
        #         ('Dolby', 'PROPN'), ('Digital', 'PROPN')
        matcher = Matcher(nlp.vocab)
        matcher.add("fos1", self.collect_sents, [{'TAG': {"IN": ['JJ']}},
                                                 {'TAG': {"IN": ['NN']}}])  # add pattern JJ[O] NN[F] 
        matcher.add("fos1.1", self.collect_sents, [{'TAG': {"IN": ['NN']}}, # picture/NN quality/NN great/JJ
                                                 {'TAG': {"IN": ['NN']}},
                                                  {'TAG': {"IN": ['JJ']}}])  # add pattern JJ[O] NN[F] 
               
        matcher.add("fos2", self.collect_sents,
                    [{'TAG': {"IN": ['VB']}},
                     {'TAG': {"IN": ['IN']}},
                     {'TAG': 'DT'},
                     {'TAG': {"IN": ['NN']}}])  # add pattern VB[O] NN[F]

        matcher.add("fos3", self.collect_sents,
                    [{'TAG': {"IN": ['JJ']}},
                     {'TAG': {"IN": ['CC']}},
                     {'TAG': {"IN": ['JJ']}},
                     {'TAG': {"IN": ['NN']}}])  # add pattern JJ[O] JJ[O] NN[F]

        matcher.add("fos4", self.collect_sents,
                    [{'TAG': {"IN": ['NN']}},
                     {'TAG': {"IN": ['VB']}},
                     {'TAG': {"IN": ['JJ']}}])  # add pattern NN[F] JJ[O]

        matcher.add("fos5", self.collect_sents,
                    [{'TAG': {"IN": ['NN']}},
                     {'TAG': {"IN": ['VB']}},
                     {'TAG': {"IN": ['RB']}}])  # add pattern NN[F] RB[O] 

        matcher.add("fos6", self.collect_sents,
                    [{'TAG': {"IN": ['NN']}},
                     {'TAG': {"IN": ['VB']}},
                     {'TAG': {"IN": ['RB']}},
                     {'TAG': {"IN": ['RB']}}])  # add pattern NN[F] RB[O]

        matcher.add("fos7", self.collect_sents,
                    [{'TAG': {"IN": ['NN']}},
                     {'TAG': {"IN": ['MD']}},
                     {'TAG': {"IN": ['RB']}},
                     {'TAG': {"IN": ['VB']}}])  # add pattern NN[F] VB[O]        

        matcher.add("fos8", self.collect_sents,
                    [{'TAG': {"IN": ['VB']}},
                     {'TAG': {"IN": ['DT']}},
                     {'TAG': {"IN": ['NN']}}])  # add pattern NN[F] VB[O]

        matcher.add("fos9", self.collect_sents,
                    [{'TAG': {"IN": ['NN']}},
                     {'TAG': {"IN": ['VBZ']}},
                     {'TAG': {"IN": ['JJR']}},
                     {'TAG': {"IN": ['IN']}},
                     {'TAG': {"IN": ['NN']}}])  # add pattern NN[F] JJR[O]

        matcher.add("fos10", self.collect_sents,
                    [{'TAG': {"IN": ['VB']}},
                     {'TAG': {"IN": ['NN']}}])  # add pattern NN[F] VB[O]    
        # -----------------------------------------------------------------------------
        matcher.add("p1", self.collect_sents, [{'TAG': {"IN": ['JJ']}},
                                               {'TAG': {"IN": ['NN',"NNS"]}}
                                               ])  
        matcher.add("p2", self.collect_sents, [{'TAG': {"IN": ['JJ']}},
                                               {'TAG': {"IN": ['NN',"NNS"]}},
                                               {'TAG': {"IN": ['NN',"NNS"]}}
                                               ])  
        matcher.add("p3", self.collect_sents, [{'TAG': {"IN": ['RB','RBR','RBS']}},
                                               {'TAG': {"IN": ['JJ']}},
                                               {'TAG': {"IN": ['NN',"NNS"]}}
                                               ]) 
#         matcher.add("p4", self.collect_sents, [{'TAG': {"IN": ['RB','RBR','RBS']}},
#                                                {'TAG': {"IN": ['RB','RBR','RBS']}},
#                                                {'TAG': {"IN": ['NN',"NNS"]}}
#                                                ])
        matcher.add("p5", self.collect_sents, [{'TAG': {"IN": ['RB','RBR','RBS']}},
                                               {'TAG': {"IN": ['RB','RBR','RBS']}},
                                               {'TAG': {"IN": ['JJ']}},
                                               {'TAG': {"IN": ['NN',"NNS"]}}
                                               ])
#         matcher.add("p6", self.collect_sents, [{'TAG': {"IN": ['RB','RBR','RBS']}},
#                                                {'TAG': {"IN": ['VBN','VBD','VB']}}                                               
#                                                ])
        matcher.add("p7", self.collect_sents, [{'TAG': {"IN": ['VBN','VBD','VB']}},
                                               {'TAG': {"IN": ['JJ']}}                                               
                                               ])    
#         matcher.add("p8", self.collect_sents, [{'TAG': {"IN": ['RB','RBR','RBS']}},
#                                                {'TAG': {"IN": ['RB','RBR','RBS']}},
#                                                {'TAG': {"IN": ['JJ']}}                                               
#                                                ])  
        matcher.add("p9", self.collect_sents, [{'TAG': {"IN": ['VBN','VBD','VB']}},
                                               {'TAG': {"IN": ['RB','RBR','RBS']}},
                                               {'TAG': {"IN": ['RB','RBR','RBS']}}
                                              ])

        doc = nlp(sent)
        matches = matcher(doc)

        for match_id, start, end in matches:
            # Get the string representation
            pattern_id = nlp.vocab.strings[match_id]
            span = doc[start:end]  # The matched span
            pos = [(token.text,token.tag_) for token in doc]
            span_pos = pos[start:end]
#             print(match_id, pattern_id, start, end, span.text)
#             print(span_pos)
            fos = self.match_fos(pattern_id,span_pos) 
#             print(fos)
            for fo in fos:
                if fo not in res: 
                    res.append(fo)
        return res
    
    def match_fos(self,pattern_id,span_pos):
        if pattern_id == "fos1": # add pattern JJ[O] NN[F]
            o = span_pos[0][0]
            f = span_pos[1][0]
        if pattern_id == "fos1.1": # add pattern JJ[O] NN[F]
            o = span_pos[2][0]
            f = "%s %s"%(span_pos[0][0] ,span_pos[1][0])
        elif pattern_id == "fos2":
            o = span_pos[0][0]
            f = span_pos[3][0]
        elif pattern_id == "fos3":
            o1 = span_pos[0][0] 
            o2 = span_pos[2][0]
            f = span_pos[3][0]
            return [(f,o1),(f,o2)]
        elif pattern_id == "fos4":
            o = span_pos[2][0]
            f = span_pos[0][0]
        elif pattern_id == "fos5":
            o = span_pos[2][0]
            f = span_pos[0][0]
        elif pattern_id == "fos6":
            o = span_pos[3][0]
            f = span_pos[0][0]
        elif pattern_id == "fos7":
            o = span_pos[0][0]
            f = span_pos[2][0]
        elif pattern_id == "fos8":
            o = span_pos[0][0]
            f = span_pos[2][0]
        elif pattern_id == "fos9":
            o = span_pos[1][0]
            f = span_pos[0][0]
        #-----------------------------------------------
        elif pattern_id == "p1":
            o = span_pos[0][0]
            f = span_pos[1][0]
        elif pattern_id == "p2":
            o = span_pos[0][0]
            f = "%s %s"%(span_pos[1][0] ,span_pos[2][0])
        elif pattern_id == "p3":
            o = "%s %s"%(span_pos[0][0] ,span_pos[1][0])
            f = span_pos[2][0]
        elif pattern_id == "p5":
            o = "%s %s"%(span_pos[1][0] ,span_pos[2][0])
            f = span_pos[3][0]
        elif pattern_id == "p6":
            o = "%s %s"%(span_pos[1][0] ,span_pos[2][0])
            f = span_pos[3][0]
        elif pattern_id == "p7":
            o = span_pos[1][0]
            f = span_pos[0][0]
#         elif pattern_id == "p8":
#             o = span_pos[1][0]
#             f = span_pos[0][0]
        elif pattern_id == "p9":
            o = span_pos[2][0]
            f = span_pos[0][0]
        else:
            return []
        
        if o in stopwords: return ()
        else: return [(f,o)]

    def run(self):
#         for sent in nltk.sent_tokenize(self.article):
#             sent = sent.lower()
#             sent = sent.replace(",", " ").replace(":", " ").replace("(", " ").replace(")", " ").replace("multi-",                                                                                                        "multiple ")
        self.article = self.article.replace(" is "," ")
        self.article = self.article.replace("  "," ")
        self.article = re.sub(r'(?:^| )\w(?:$| )', " ", self.article).strip() # remove single alphbet
        ress = self.match_pattern(self.article)
        return ress
	
# FOP-Rule(Dependency)
class FOP_rule_Depend():
    def __init__(self, review):
        self.review = review
        self.possible_verb = ['VB','MD','VBG','VBN']
        self.possible_adj = ["JJ","JJR"]
        self.possible_noun = ["NN","NNP","NNPS","NNS"]
        self.possible_adv = ['RB','RBR','RBS']

    def rule1(self, doc):
        #         amod(N, A) →< N, A >
        res = []
        for token_noun in doc:
            if token_noun.tag_ in self.possible_noun:
                for possible_opinion in token_noun.children:
                    if possible_opinion.dep == amod and possible_opinion.tag_ in self.possible_adj:
#                         print(token_noun.text,token_noun.tag_, possible_opinion.text,possible_opinion.tag_)
                        f,o = token_noun.text,possible_opinion.text
                        if o not in stopwords and f != o:
                            res.append((f,o))
                    #         print("rule1",res)
        return res

    def rule2(self, doc):
        #         acomp(V, A) + nsubj(V, N) →< N, A >
        res = []
        term1s, term2s = [], []
        for token_verb in doc:
            if token_verb.tag_ in self.possible_verb:
                for possible_opinion in token_verb.children:
                    if possible_opinion.dep == acomp and possible_opinion.tag_ in self.possible_adj:
#                         print(token_verb.text,token_verb.tag_, possible_opinion.text,possible_opinion.tag_)
                        term1s.append([token_verb.text, possible_opinion.text])  # acomp(V, A)
                    if possible_opinion.dep == nsubj and possible_opinion.tag_ in self.possible_noun:
                        term2s.append([token_verb.text, possible_opinion.text])  # nsubj(V, N)

        for t1 in term1s:
            V1, A1 = t1
            for t2 in term2s:
                V2, N2 = t2
                if V1 == V2: 
                    f,o = N2, A1
                    if o not in stopwords and f != o:
                        res.append((f,o))  # →< N, A >
            #         print("rule2",res)
        return res
    
    def rule3(self, doc):
        res = []
        #         nsubj(aux,noun) + acomp(aux,adj) →< N, A >

        term1s, term2s = [], []
        for token_AUX in doc:
            if token_AUX.pos == AUX:
                for possible_head in token_AUX.children:
                    if possible_head.dep == nsubj and possible_head.tag_ in self.possible_noun:
                        term1s.append([token_AUX.text, possible_head.text])  # nsubj(aux,noun)
                    if possible_head.dep == acomp and possible_head.tag_ in self.possible_adj:
                        term2s.append([token_AUX.text, possible_head.text])  # acomp(aux,adj)

        for t1 in term1s:
            Aux1, N1 = t1
            for t2 in term2s:
                Aux2, A2 = t2
                if Aux1 == Aux2 and N1 != Aux1 and Aux1 != "is": 
                    f,o = N1, Aux1
                    if o not in stopwords and f != o:
                        res.append((f,o))  # →< N, A >
            #         print("rule3",res)
        v = ""
#         if len(res) > 0: 
#             v = [get_pos_sequence(" ".join(r))[1] for r in res]
        return res


#     def rule4(self, doc):
#         res = []
#         #     dobj(V, N) + nsubj(V, N0) →< N, V >
#         term1s, term2s = [], []
#         for token_verb in doc:
#             if token_verb.tag_ in self.possible_verb:
#                 for possible_head in token_verb.children:
#                     if possible_head.dep == dobj and possible_head.tag_ in self.possible_noun:
# #                         print(token_verb.text,token_verb.tag_, possible_head.text,possible_head.tag_)
#                         term1s.append([token_verb.text, possible_head.text])  # nsubj(aux,noun)
#                     if possible_head.dep == nsubj and possible_head.tag_ in self.possible_noun:
#                         term2s.append([token_verb.text, possible_head.text])  # acomp(aux,adj)
#         for t1 in term1s:
#             V1, N1 = t1
#             f,o = V1, N1
#             if o not in stopwords and f != o:
#                 res.append((f,o))
#         # res.extend(term2s)
#         #         print("rule4",res)
#         return res

    def rule5(self, doc):
        #         < h1, m > +conj and(h1, h2) →< h2, m >
        res = []
        term1s, term2s = [], []
        for token in doc:
            for possible_head in token.children:
                if possible_head.tag_ in self.possible_noun + self.possible_adj + self.possible_adv:
                    term1s.append([token.text, possible_head.text])  # < h1, m >
                if possible_head.dep == conj and possible_head.tag_ in self.possible_noun:
                    term2s.append([token.text, possible_head.text])  # conj and(h1, h2)

        for t1 in term1s:
            h1, m = t1
            for t2 in term2s:
                h21, h22 = t2
                if h1 == h21:
#                     if m not in stopwords:
                    f,o = h1, m
                    if ((o not in stopwords) and (f not in stopwords)) and (f != o): 
                        res.append((f, o))
                    f,o = h22, m
                    if ((o not in stopwords) and (f not in stopwords)) and (f != o): 
                        res.append((f, o))                    
                #         print("rule5",res)
        return res

    def rule6(self, doc):
        #         < h, m1 > +conj and(m1, m2) →< h, m2 >
        #   nsubj(aux,noun) + acomp(aux,m1) + conj and(m1, m2) →< h, m1 > + < h, m2 >
        res = []
        term1s, term2s, term3s = [], [], []
        for token in doc:
            if token.pos == AUX:
                token_AUX = token
                for possible_head in token_AUX.children:
                    if possible_head.dep == nsubj and possible_head.tag_ in self.possible_noun:
                        term1s.append([token_AUX.text, possible_head.text])  # nsubj(aux,noun)
                    if possible_head.dep == acomp and possible_head.tag_ in self.possible_adj:
                        term2s.append([token_AUX.text, possible_head.text])  # acomp(aux,m1)
            else:
                for possible_m in token.children:
                    if possible_m.dep_ == "conj": term3s.append([token.text, possible_m.text])

        for t1 in term1s:
            Aux1, N1 = t1
            for t2 in term2s:
                Aux2, A2 = t2
                if Aux1 == Aux2 and N1 != Aux1:
                    #                     res.append([N1,A2]) # →< N, A >
                    for t3 in term3s:
                        m1, m2 = t3
                        if A2 in t3:
#                             if m1 not in stopwords and N1 != m1: 
                                f,o = N1,m1
                                if ((o not in stopwords) and (f not in stopwords)) and (f != o):     
                                    res.append((f, o))  # →< N, A > 
                                f,o = N1, m2
                                if ((o not in stopwords) and (f not in stopwords)) and (f != o):     
                                    res.append((f, o))  # →< N, A > 
                        #         print("rule6",res)
        return res

    def rule7(self, doc):
        #         < h, m > +neg(m, not) →< h, not + m >
        #          acomp(aux,adj) + neg(aux,part) + nsubj(aux,N1) + compound(N1,N2) => <N1 + N2 , part + adj>
        res = []
        term1s, term2s, term3s, term4s = [], [], [], []
        for token in doc:
            if token.pos == AUX:
                token_AUX = token
                for possible_head in token_AUX.children:
                    if possible_head.dep == acomp and possible_head.tag_ in self.possible_adj:
                        term1s.append([token_AUX.text, possible_head.text])  # acomp(aux,adj)
                    if possible_head.dep == neg and possible_head.pos == PART:
                        term2s.append([token_AUX.text, possible_head.text])  # neg(aux,part)
                    if possible_head.dep == nsubj and possible_head.tag_ in self.possible_noun:
                        term3s.append([token_AUX.text, possible_head.text])  # nsubj(aux,N1)
            else:
                for possible_m in token.children:
                    #                     print(token,possible_m,possible_m.dep_)
                    if possible_m.dep_ == "compound" and possible_m.tag_ in self.possible_noun:  # compound(N1,N2)
                        term4s.append([token.text, possible_m.text])

        for t1 in term1s:
            Aux1, adj = t1
            for t2 in term2s:
                Aux2, part = t2
                if Aux1 == Aux2:
                    #                     print(part , adj) # <part , adj>
                    for t3 in term3s:
                        Aux3, N1 = t3
                        if Aux1 == Aux3:
                            for t4 in term4s:
                                N41, N42 = t4
                                if N1 in t4:
#                                     res.append([N42, N41, part, adj])  # => <N1 + N2 , part + adj>
                                    f,o = "%s %s"%(N42, N41),"%s %s"%(part, adj)
                                    if (o not in stopwords) and (o not in f):
                                        res.append((f,o))  # => <N1 + N2 , part + adj>

                                #         print("rule7",res)
        return res

    def rule8(self, doc):
        #         < h, m > +nn(h, N) →< N + h, m >
        #       compound(h,N1) + nsubj(V1,N1) + acomp(V1,A) -> <h + N1 , A>
        res = []
        term1s, term2s, term3s = [], [], []
        for token in doc:
            for possible_head in token.children:
                if possible_head.dep_ == "compound" and possible_head.tag_ in self.possible_noun:
                    term1s.append([token.text, possible_head.text])  # compound(h,N1)
                if possible_head.dep == nsubj and token.tag_ in self.possible_verb and possible_head.tag_ in self.possible_noun:
                    term2s.append([token.text, possible_head.text])  # nsubj(V1,N1)
                if possible_head.dep == acomp and token.tag_ in self.possible_verb and possible_head.tag_ in self.possible_adj:
                    term3s.append([token.text, possible_head.text])  # acomp(V1,A)
        for t1 in term1s:
            N1, N2 = t1
            for t2 in term2s:
                V21, N22 = t2
                if N22 in t1:
                    #                     print(N2,N1) # <h + N1>
                    for t3 in term3s:
                        V31, A = t3
                        if V31 == V21:
                            f,o = "%s %s"%(N2, N1), A
                            if (o not in stopwords) and (o not in f):
                                res.append((f,o))  # -> <h + N1 , A>

                        #         print("rule8",res)
        return res

    def rule9(self, doc):
        #         < h, m > +nn(N, h) →< h + N, m >
        #         compound(N1,N2) + dobj(V,N1) -> < N2 + N1, V >
        res = []
        term1s, term2s = [], []
        for token in doc:
            for possible_head in token.children:
                if possible_head.dep_ == "compound" and possible_head.tag_ in self.possible_noun:
                    term1s.append([token.text, possible_head.text])  # compound(N1,N2)
                if possible_head.dep == dobj and token.tag_ in self.possible_adj + self.possible_verb and possible_head.tag_ in self.possible_noun:
#                     print(token.text,token.tag_, possible_head.text,possible_head.tag_)
                    term2s.append([token.text, possible_head.text])  # dobj(V,N1)
        for t1 in term1s:
            N1, N2 = t1
            for t2 in term2s:
                V, N22 = t2
                if N22 in t1:
                    f,o = "%s %s"%(N2, N1), V
                    if (o not in stopwords) and (o not in f):
                        res.append((f,o))  # -> <h + N1 , A>

                #         print("rule9",res)
        return res

    def run(self):
        self.review = self.review
        doc = nlp(self.review)
        #         displacy.render(doc, style='dep', jupyter = True) # dependency parse tree
        deplist = []
        dep1 = self.rule1(doc)
        dep2  = self.rule2(doc)
        dep3  = self.rule3(doc)
#         dep4  = self.rule4(doc)
        dep5  = self.rule5(doc)
        dep6  = self.rule6(doc)
        dep7  = self.rule7(doc)
        dep8  = self.rule8(doc)
        dep9  = self.rule9(doc)
#         print("dep1:",str(dep1))
#         print("dep2:",str(dep2))
#         print("dep3:",str(dep3))
#         print("dep4:",str(dep4))
#         print("dep5:",str(dep5))
#         print("dep6:",str(dep6))
#         print("dep7:",str(dep7))
#         print("dep8:",str(dep8))
#         print("dep9:",str(dep9))
#         print("-------------------------------------")
#         print(dep1 , dep2 , dep3 , dep4 , dep5 , dep6 , dep7 , dep8 , dep9)
        dep = dep1 + dep2 + dep3 + dep5 + dep6 + dep7 + dep8 + dep9
        return dep
	
class FOP_rule_Depend2():
    def __init__(self, review):
        self.review = review
        self.possible_verb = ['VB','MD','VBG','VBN']
        self.possible_adj = ["JJ","JJR"]
        self.possible_noun = ["NN","NNP","NNPS","NNS"]
        self.possible_adv = ['RB','RBR','RBS']
        # spacy 是head指向dependecy(child)
        # 一個word可以有很多child

    def rule10(self, doc):
        #  nsubj(f,o)
        res = []
        for token in doc:
            for children in token.children:
                if children.dep == nsubj:    
                    if children.tag in self.possible_noun:
    #                     print(token.text,token.tag_, children.text,children.tag_)
                        f,o = children,token
                        if (o.text in stopwords) or (f.text in stopwords): continue
                        if (o.text == f.text): continue
                        res.append((f.text,o.text))
                        #         print("rule10",res)
        return res

    
    def rule12(self, doc):
        #  amod(f,o)
        res = []
        for token in doc:
            for children in token.children:
                if children.dep == amod:                
#                     print(token.text,token.tag_, children.text,children.tag_)
                    f,o = token,children
                    if (o.text in stopwords) or (f.text in stopwords): continue
                    if (o.text == f.text): continue
                    res.append((f.text,o.text))
                    #         print("rule12",res)
        return res


    def rule13(self, doc):
        #  advmod(f,o)
        res = []
        for token in doc:
            for children in token.children:
                if children.dep == advmod and token.tag_ in self.possible_adj + self.possible_verb:                
#                     print(token.text,token.tag_, children.text,children.tag_)
                    f,o = token,children
                    if (o.text in stopwords) or (f.text in stopwords): continue
                    if (o.text == f.text): continue
                    res.append((f.text,o.text))
                    #         print("rule12",res)
        return res
    
    def rule14(self, doc):
        #  dobj(f,o)
        res = []
        for token in doc:
            for children in token.children:
                if children.dep == dobj and children.pos in [NOUN,PROPN] and token.tag_ in self.possible_verb:  
                    if (token.tag_ in self.possible_adj + self.possible_adv):#
                        f,o = children,token
                        if (o.text in stopwords) or (f.text in stopwords): continue
                        if (o.text == f.text): continue
                        res.append((f.text,o.text))
                        #         print("rule12",res)
        return res

    
    def run(self):
        self.review = self.review.replace(" is "," ")
        self.review = self.review.replace("  "," ")
        self.review = re.sub(r'(?:^| )\w(?:$| )', " ", self.review).strip() # remove single alphbet
#         print(self.review)
#         print(get_pos_sequence(self.review)[1])    
        doc = nlp(self.review)
#         displacy.render(doc, style='dep', jupyter = True) # dependency parse tree
        deplist = []
        dep10 = self.rule10(doc)
        dep12  = self.rule12(doc)
        dep13  = self.rule13(doc)
        dep14  = self.rule14(doc)        
#         print("dep1:",str(dep10))
#         print("dep3:",str(dep12))
#         print("dep4:",str(dep13))
#         print("dep5:",str(dep14))
        
#         print("-------------------------------------")
        dep = dep10 + dep12 + dep13 + dep14
        return dep
    
    
def TwoRuleFOPDep(rev_sent):
    fops1 = FOP_rule_Depend(rev_sent).run() 
    fops2 = FOP_rule_Depend2(rev_sent).run()
    fops = []
    [fops.append(fop) for fop in fops1 if fop not in fops]
    [fops.append(fop) for fop in fops2 if fop not in fops]    
    return fops