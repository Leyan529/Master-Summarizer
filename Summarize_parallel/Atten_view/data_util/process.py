# ! pip install nltk==3.5
import nltk
import re
from transformers import BertTokenizer 
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import spacy
import en_core_web_sm
from spacy.matcher import Matcher
nlp = en_core_web_sm.load()

from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
# import networkx as nx
import os
import pickle
from data_util.my_stopwords import *
from data_util.extract_key import extract_PF

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
# from ekphrasis.classes.segmenter import Segmenter

from ekphrasis.classes.segmenter import Segmenter
# segmenter using the word statistics from english Wikipedia
seg_eng = Segmenter(corpus="twitter") 

from ekphrasis.classes.spellcorrect import SpellCorrector
sp = SpellCorrector(corpus="twitter") 

alphbet_stopword = ['','b','c','d','e','f','g','h','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','#']

# 斷詞辭典
from nltk.corpus import stopwords as nltk_stopwords
nltk_stopwords = set(nltk_stopwords.words("english"))
stpwords_list3 = [f.replace("\n","") for f in open("data_util/stopwords.txt","r",encoding = "utf-8").readlines()]
stpwords_list3.remove("not")
stopwords = list(html_escape_table + stpwords_list2) + list(list(nltk_stopwords) + list(stpwords_list1) + list(stpwords_list3))

# stopwords = list(html_escape_table)  #+ list(stpwords_list1) + list(stpwords_list3)
print("斷詞辭典 已取得")

# Total Opinion
opinion_lexicon = {}
for filename in os.listdir('data_util/opinion-lexicon-English/'):      
    if "txt" not in filename: continue
    print(filename)
    with open('data_util/opinion-lexicon-English/'+filename,'r') as f_input:
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


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date','hashtag','number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    spell_correction = True,
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)    

# ---------------------pipeline----------------------------
# step1 縮寫還原, 過濾html字元
import re
def lower_remove_simple_html(text):
    text = str(text)
    text = text.lower()
    text = text.strip()

    for k, v in contraction_mapping.items():
        if k in text:
            text = text.replace(k, v)
    for k, v in special_contractions_mapping.items():
        if k in text:
            text = text.replace(k, v)

    for k in html_escape_table:
        if k in text:
            text = text.replace(k, "")   

    text = remove_tags(text)
    return text
# step1.5 移除符號特徵 + 小數點數值
def remove_symbol(text):
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
    return text

# step2 nltk.sent_tokenize + Bert 絕對斷詞
def squeeze_sym(s):  
    char = "."
    while char * 2 in s:
        s = s.replace(char * 2, char)
    s = s.replace('.', ' ')
    return s

# 排除英數字字元(可不做)
def clean_wordlist(wordlist):
    wordlist = [
        ''.join(re.findall(r'[A-Za-z]', word)) \
        if (word.isalnum() and not (word.isdigit()))
        else word
        for word in wordlist
    ]
    return wordlist

def nltk_bert_token_sents(text):
    text = [
                " ".join([token for token in bert_tokenizer.tokenize(sent)])
                for sent in nltk.sent_tokenize(text)
           ]
    text = " ".join(text).replace(" ##","")
    # text = " ".join(clean_wordlist(text.split(" ")))
    return text

def ekphrasis_process(paragraphs):
    # abs correct + segment
    pre_process_tokens = text_processor.pre_process_doc(paragraphs)
    # Word Segmentation
    segment_tokens = []
    for word in pre_process_tokens:
        segment_tokens.extend(seg_eng.segment(word).split(" "))
    segment_tokens = [token for token in segment_tokens if token not in alphbet_stopword]
        
    # Spell Correction(for dictionary corpus)   
    # correct_tokens = [sp.correct(token) if token != '.' else token for token in segment_tokens ]

    text = remove_tags(" ".join(segment_tokens))
    remove_chars = '["#$%&\'\"\()*+:<=>?@★【】《》“”‘’[\\]^_`{|}~]+'
    text = re.sub(remove_chars, "", text)  # remove number and segment
    text = squeeze(text)
    return text

# step3 萃取review 名詞特徵 + 詞性還原
def nltk_noun_pharse_lemm(text):
    remove_chars = '["#$%&\'\"\()*+:<=>?@★【】《》“”‘’[\\]^_`{|}~]+'
    text = re.sub(remove_chars, "", text)  # remove number and segment
    
    new_sents, noun_feats = [], []
    for sent in nltk.sent_tokenize(text):
        
        # doc = nlp(sent)
        # sentence = " ".join(
        #     [
        #         word.lemma_ if word.pos_.startswith('N') or word.pos_.startswith('J') or
        #                     word.pos_.startswith('V') or word.pos_.startswith('R') else word.orth_
        #         for word in doc
        #         if (
        #         (not word.is_space and
        #         not word.is_bracket and
        #         # not word.is_digit and
        #         not word.is_left_punct and
        #         not word.is_right_punct and
        #         not word.is_bracket and
        #         not word.is_quote and
        #         not word.is_currency and
        #         not word.is_punct 							
        #         # word.tag_ not in ["SYM", "HYPH"] and
        #         # word.lemma_ != "-PRON-"
        #         )
        #         )
        #     ]
        # ) 
        sentence = TextBlob(sent)
        sentence = " ".join([str(w.singularize()) for w in sentence.words])
        # print(sentence)
        if len(sentence.split(" ")) > 3:
            sentence = sentence + " ."
            new_sents.append(sentence)
            text_keywords = extract_PF(sentence).run()
            noun_feats.extend(text_keywords)
    
#     newString = re.sub(r'http\S+', '', text)
    return new_sents, list(set(noun_feats))



def squeeze(s):  
    char = " "
    while char * 2 in s:
        s = s.replace(char * 2, char)
    return s

def review_clean(text):
    text = lower_remove_simple_html(text)
    text = remove_symbol(text)
    # text = nltk_bert_token_sents(text)  
    text = ekphrasis_process(text)
    text, feats = nltk_noun_pharse_lemm(text)
    text = " ".join(text)
    text = squeeze(text)
    return text, feats, nltk.sent_tokenize(text)

def summary_clean(text):
    text = lower_remove_simple_html(text)    
    text = squeeze_sym(text)
    text = remove_symbol(text) 
    # text = nltk_bert_token_sents(text)  
    text = ekphrasis_process(text)
    text, _ = nltk_noun_pharse_lemm(text) 
    text = "<s> " + " ".join(text) + " </s>"
    text = text.replace("." , "")
    text = squeeze(text)
    return text