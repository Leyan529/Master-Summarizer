from bs4 import BeautifulSoup
from summa import keywords as TextRank
from extract_key import *

# from data_util.stopwords import *
# from data_util.preprocess import *
import nltk
import pandas as pd
from textblob import TextBlob

from data_util.process import review_clean, summary_clean

def get_keys(review):
    POS_keys , DEP_keys, Noun_adj_keys = [] , [] , []
    for sent in review.split(" . "):
        # pos
        POS_keys = POS_keys + extract_POS(sent).run()[0]
        # dep
        DEP_keys = DEP_keys + extract_DEP(sent).run()[0]
        # noun_adj
        Noun_adj_keys = Noun_adj_keys + noun_adj(sent)[0]
    
    # TextRank
    TextRank_keywords = []

    for words in TextRank.keywords(review).split('\n'):
        TextRank_keywords.extend(words.split(" "))

    return POS_keys, DEP_keys, Noun_adj_keys

# def text_cleaner(text):
#     # print(text)
#     text = squeeze3(text) # 優先過濾...

#     # 移除符號特徵
#     # ----------------------------------------------------------------------
#     pattern = re.compile(r"([\d\w\.-]+[-'//.][\d\w\.-]+)")  # |  ([\(](\w)+[\)])
#     keys = pattern.findall(text)           
#     # 移除符號特徵    
#     # keywords.extend(keys)
#     for k in keys:    
#         k = k.replace("(","").replace(")","")
#         text = text.replace(k,"(" + k + ")")
#         text = text.replace("((","(").replace("))",")")     
#         text = text.replace("(" + k + ")","") 
#         text = text.strip()
#     # ----------------------------------------------------------------------

#     text = remove_word5(text)
#     newString = text.lower()
#     # newString = remove_tags(newString)
#     newString = re.sub(r'http\S+', '', newString)
#     # newString = BeautifulSoup(newString, "lxml").text
#     newString = re.sub(r'\([^)]*\)', '', newString)
#     newString = re.sub('"','', newString)
#     # newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
#     newString = re.sub(r"'s\b","",newString)
#     # newString = re.sub("[^a-zA-Z.]", " ", newString) # 不保留數字
#     newString = re.sub("[^0-9a-zA-Z.]", " ", newString)   # 保留數字  
#     tokens = [w for w in newString.split() if w not in alphbet_stopword]    

#     long_words=[]
#     for i in tokens:
#         if len(i)>=3:                  #removing short word
#             long_words.append(i)   
#     newString = (" ".join(long_words)).strip()    
    
#     newString_sents = []
#     text_keywords = []
#     for sent in nltk.sent_tokenize(newString):
#         sent = " ".join([token for token in bert_tokenizer.tokenize(sent)])
#         sent = sent.replace(" ##","")

#         lemm_text = lemm_sent_process5(sent)
#         if len(lemm_text.split(" "))<3:  continue
#         lemm_text = squeeze2(lemm_text)
#         if lemm_text[-1] != ".": lemm_text = lemm_text+ " ." # 強制加上dot
#         newString_sents.append(lemm_text)
#         # text_keywords2 = PF_rule_POS(lemm_text).run()
#         # text_keywords.extend(text_keywords2)
    
#     # newString_sents = [line for line in newString_sents if len(line) > 5]
#     newString = " \n".join(newString_sents) # 句語句之間分隔
#     newString = squeeze2(newString)
#     newString = newString.replace(" . . "," . ")
#     # print(newString)
#     return newString,newString_sents

# def lemm(reviewtext):

#     reviewtext, _ = text_cleaner(reviewtext)
#     reviewtext = squeeze2(reviewtext)
#     reviewtext = reviewtext.replace("\n","")
#     reviewtext = " ".join([t for t in reviewtext.split(" ") if t != ""]).strip()
#     return reviewtext

# def lemm_summary(summarytext):
#     summarytext, _ = text_cleaner(summarytext)
#     summarytext = squeeze2(summarytext)
#     summarytext = summarytext.replace("\n","")
#     summarytext = " ".join([t for t in summarytext.split(" ") if t != ""]).strip()
#     # return "<s> "  + summarytext + " <s>"
#     return summarytext

'''return avaliable review - reference summary'''
def parse_page(htmlpage):
    soup = BeautifulSoup(htmlpage,'lxml')
    blocks = soup.findAll("div", {"data-hook" : "review"})
    row = []
    num = 0
    for block in blocks:
        try:
            summary = block.findAll("a", {"class" : "a-link-normal"})[1].text.strip()
            reviewtext = block.find("div", {"class" : "a-spacing-small"}).text.strip()
            rating = float(block.findAll("a", {"class" : "a-link-normal"})[0].text.strip().split(" ")[0])
            date = block.find("span", {"class" : "review-date"}).text.strip()
            date = " ".join(date.split(" ")[6:])
        except:
            continue

        if rating >= 4:
            binaryrating = 'positive'
        else:
            binaryrating = 'negative'

        lemm_reviewtext = review_clean(reviewtext)[0]
        summary = summary_clean(summary)
        summary_blob = TextBlob(summary)
        summary_polarity = summary_blob.sentiment.polarity
        summary_subjectivity = summary_blob.sentiment.subjectivity
        if len(lemm_reviewtext.split(" ")) <= 50: continue
        # if len(summary.split(" ")) <= 5: continue
        POS_keys, DEP_keys, Noun_adj_keys = get_keys(lemm_reviewtext)
        row.append([rating, date, summary, reviewtext, lemm_reviewtext, binaryrating, POS_keys, DEP_keys, Noun_adj_keys, summary_polarity, summary_subjectivity])
        # row = row + [date, summary, reviewtext, rating, binaryrating]
        num += 1    

    
    return pd.DataFrame(row, columns=['rating','date', 'summary', 'reviewtext', 'lemm_reviewtext', 'binaryrating', \
                                                    'POS_keys', 'DEP_keys', 'Noun_adj_keys','summary_polarity','summary_subjectivity'])