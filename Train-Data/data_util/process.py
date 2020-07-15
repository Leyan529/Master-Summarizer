# ! pip install nltk==3.5
import nltk
import re

import spacy
import en_core_web_sm
from spacy.matcher import Matcher
nlp = en_core_web_sm.load()


import string

# from nltk.corpus import stopwords
# from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
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
seg_eng = Segmenter(corpus="twitter") # english or twitter

from ekphrasis.classes.spellcorrect import SpellCorrector
sp = SpellCorrector(corpus="english") # english or twitter

alphbet_stopword = ['','b','c','d','e','f','g','h','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','#']

# 斷詞辭典
from nltk.corpus import stopwords as nltk_stopwords
nltk_stopwords = set(nltk_stopwords.words("english"))
stpwords_list3 = [f.replace("\n","") for f in open("data_util/stopwords.txt","r",encoding = "utf-8").readlines()]
stpwords_list3.remove("not")
stopwords = list(html_escape_table + stpwords_list2) + list(list(nltk_stopwords) + list(stpwords_list1) + list(stpwords_list3))
stopwords = stopwords + ["."] + alphbet_stopword
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
        'time', 'date','hashtag'],
    # terms that will be annotated
    # annotate={"hashtag", "allcaps", "elongated", "repeated",
    #     'emphasis', 'censored'},
    annotate={"hashtag", "allcaps", "repeated",'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="english", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    spell_correction = False,
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
    # mode = 'fast'
)    

# ---------------------pipeline----------------------------
# step1 縮寫還原, 過濾html字元
import re
def lower_contraction(text):    
    text = str(text)
    text = text.lower()
    text = text.strip()
    text = re.sub(' +',' ',text) # Removing extra spaces    
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
    # # 移除符號特徵
    # # ----------------------------------------------------------------------
    # pattern = re.compile(r"([\d\w\.-]+[-'//.][\d\w\.-]+)")  # |  ([\(](\w)+[\)])
    # keys = pattern.findall(text)           
    # # 移除符號特徵    
    # # keywords.extend(keys)
    # for k in keys:    
    #     k = k.replace("(","").replace(")","")
    #     text = text.replace(k,"(" + k + ")")
    #     text = text.replace("((","(").replace("))",")")     
    #     text = text.replace("(" + k + ")","") 
    #     text = text.strip()
    def clean_text_round1(text):
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('\[*?\]', '', text)
        # text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        # text = re.sub('\w*\d\w*', '', text) # W*:A-Z  d:digit   # 取消刪除數字    
        return text
    def clean_text_round2(text):
        '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', ' . ', text)
        text = re.sub(' +',' ',text) # Removing extra spaces
        return text
    def clean_text_round3(text):
        '''remove containing numbers/extra space'''
        text = re.sub(' +',' ',text) # Removing extra spaces
        text = re.sub('\x95', ' ', text)
        text = re.sub('nbsp', '', text)
        text = re.sub('á', 'a', text)
        text = re.sub('é', 'e', text)
        text = re.sub('í', 'i', text)
        text = re.sub('ó', 'o', text)
        text = re.sub('ú', 'u', text)
        text = re.sub('[%s]' % re.escape(r"!#$%&'()*+,-/:;<=>?@[\]^_`{|}~"), '', text)
        # text = re.sub(r" ?\([^\D]+\)", "", text)
        return text

    pattern = re.compile(r"([\d\w\.-]+[-'//.][\d\w\.-]+)")  # |  ([\(](\w)+[\)])
    keys = pattern.findall(text)            
        
    for k in keys:    
        k = k.replace("(","").replace(")","")
        text = text.replace(k,"(" + k + ")")
        text = text.replace("((","(").replace("))",")")     
        text = text.replace("(" + k + ")","") # 移除符號特徵
        text = text.strip()

    text = clean_text_round1(text)
    text = clean_text_round2(text)
    text = clean_text_round3(text)
    return text

# step2 nltk.sent_tokenize + Bert 絕對斷詞
def squeeze_sym(s):  
    char = "."
    while char * 2 in s:
        s = s.replace(char * 2, char)
    s = s.replace('.', ' ')
    return s



def ekphrasis_process(text):
    # ekphrasis 語料修正 + 語料切詞
    # abs correct + segment
    pre_process_tokens = text_processor.pre_process_doc(text)
    # Word Segmentation
    segment_tokens = []
    for word in pre_process_tokens:
        segment_tokens.extend(seg_eng.segment(word).split(" "))
    segment_tokens = [token for token in segment_tokens if token not in alphbet_stopword]
        
    # Spell Correction(for dictionary corpus)   
    correct_tokens = [token if ((token in stopwords)or(token.isalnum())) else sp.correct(token) for token in segment_tokens ]

    text = remove_tags(" ".join(segment_tokens))
    # text = remove_tags(" ".join(correct_tokens))
    # remove_chars = '["#$%&\'\"\()*+:<=>?@★【】《》“”‘’[\\]^_`{|}~]+'
    # text = re.sub(remove_chars, "", text)  # remove number and segment
    text = squeeze(text)
    return text


lemmatizer = WordNetLemmatizer()
def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:                    
    return None

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))    
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        # if word in alphbet_stopword: continue
        if tag is None:                        
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    text = " ".join(res_words)
    # text = text.replace(" thi "," this ")
    # text = text.replace(" wa "," was ")
    # text = text.replace(" ha "," has ")
    # text = text.replace(" tha "," that ")
    return text

# step3 萃取review 名詞特徵 + 詞性還原
def nltk_noun_pharse_lemm(text):
    remove_chars = '["#$%&\'\"\()*+:<=>?@★【】《》“”‘’[\\]^_`{|}~]+'
    text = re.sub(remove_chars, "", text)  # remove number and segment
    
    new_sents, noun_feats = [], []
    for sent in nltk.sent_tokenize(text):
        '''spacy'''
        # doc = nlp(sent)
        # sentence = " ".join(
        #     [
        #         word.lemma_ if word.pos_.startswith('N') or word.pos_.startswith('J') or
        #                     word.pos_.startswith('V') or word.pos_.startswith('R') else word.orth_
        #         for word in doc
        #         if (
        #         (not word.is_space and # 過濾空白
        #         # not word.is_digit and
        #         not word.is_left_punct and # 過濾左標點符號
        #         not word.is_right_punct and # 過濾右標點符號
        #         not word.is_bracket and # 過濾括號
        #         not word.is_quote and # 過濾引號
        #         not word.is_currency and # 過濾錢幣符號
        #         not word.is_punct # 過濾標點符號							
        #         # word.tag_ not in ["SYM", "HYPH"] and
        #         # word.lemma_ != "-PRON-"
        #         )
        #         )
        #     ]
        # ) 
        '''只有複數變單數'''
        # sentence = TextBlob(sent)
        # sentence = " ".join([str(w.singularize()) for w in sentence.words])
        '''nltk WordNetLemmatizer'''
        sentence = lemmatize_sentence(sent)
        sentence = re.sub('[%s]' % re.escape(string.punctuation), '', sentence)
        sentence = re.sub(r"\b[bcdefghjklmnopqrstuvwxyz#]\b", "", sentence)
        sentence = re.sub(' +',' ',sentence) # Removing extra spaces        

        # print(sentence)
        if len(sentence.split(" ")) > 2:
            sentence = sentence + " ."
            new_sents.append(sentence)
            text_keywords = extract_PF(sentence).run()
            noun_feats.extend(text_keywords)
    
    # newString = re.sub(r'http\S+', '', text)
    return new_sents, list(set(noun_feats))



def squeeze(s):  
    char = " "
    while char * 2 in s:
        s = s.replace(char * 2, char)
    return s

def review_clean(text):
    # text = """ quality okay, but annoying white noise """
    # print("---------------orign review-------------")
    # print(text)
    feats = ['']
    text = lower_contraction(text)  
    text = remove_symbol(text)    
    # text = nltk_bert_token_sents(text) 
    text = ekphrasis_process(text)
    text, feats = nltk_noun_pharse_lemm(text)
    text = " ".join(text)
    '''remove continuous same words'''
    while re.search(r'\b(.+)(\s+\1\b)+', text):
        text = re.sub(r'\b(.+)(\s+\1\b)+', r'\1', text)
    text = re.sub('\d \d+',' ',text) # Removing extra numbers
    text = re.sub(' +',' ',text) # Removing extra spaces    
    text = squeeze(text)
    # print("--------------- review-------------")
    # print(nltk.sent_tokenize(text))
    # print('*')
    # print('*')
    # print('*')
    # print('*')
    return text, feats, nltk.sent_tokenize(text)

def summary_clean(text):
    # text = "quality okay, but annoying white noise"
    # print("---------------orign summary-------------")
    # print(text)
    text = lower_contraction(text)    
    text = squeeze_sym(text)
    text = remove_symbol(text)     
    # text = nltk_bert_token_sents(text)  
    text = ekphrasis_process(text)
    text, _ = nltk_noun_pharse_lemm(text) 
    text = re.sub('\w*\d\w*', '', text) # W*:A-Z  d:digit   # 刪除摘要數字 
    text = "<s> " + " ".join(text) + " </s>"
    text = text.replace("." , "")
    '''remove continuous same words'''
    while re.search(r'\b(.+)(\s+\1\b)+', text):
        text = re.sub(r'\b(.+)(\s+\1\b)+', r'\1', text)
    text = re.sub('\d \d+',' ',text) # Removing extra numbers
    text = re.sub(' +',' ',text) # Removing extra spaces       
    text = squeeze(text)
    # print("---------------summary-------------")
    # print(text)
    # print('*')
    # print('*')
    # print('*')
    # print('*')
    return text

