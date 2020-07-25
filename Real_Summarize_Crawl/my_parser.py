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
        # POS_keys = POS_keys + extract_POS(sent).run()[0]
        # dep
        # DEP_keys = DEP_keys + extract_DEP(sent).run()[0]
        # noun_adj
        Noun_adj_keys = Noun_adj_keys + noun_adj(sent)[0]
    
    # TextRank
    # TextRank_keywords = []

    # for words in TextRank.keywords(review).split('\n'):
    #     TextRank_keywords.extend(words.split(" "))

    return Noun_adj_keys

'''return avaliable review - reference summary'''
def parse_page(asin, htmlpage):
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
        lemm_summary = summary_clean(summary)
        summary_blob = TextBlob(lemm_summary)
        summary_polarity = summary_blob.sentiment.polarity
        summary_subjectivity = summary_blob.sentiment.subjectivity
        if len(lemm_reviewtext.split(" ")) < 50: continue
        # if len(summary.split(" ")) <= 5: continue
        Noun_adj_keys = get_keys(lemm_reviewtext)
        summary_conflict = (binaryrating== 'positive' and summary_blob.sentiment.polarity<0) or (binaryrating== 'negative' and summary_blob.sentiment.polarity>0)
        row.append([asin, rating, date, reviewtext, lemm_reviewtext, summary, lemm_summary, binaryrating, Noun_adj_keys, summary_polarity, summary_subjectivity, summary_conflict])
        # row = row + [date, summary, reviewtext, rating, binaryrating]
        num += 1    

    
    return pd.DataFrame(row, columns=['asin','rating','date', 'reviewtext', 'lemm_reviewtext','summary','lemm_summary', 'binaryrating', \
                                                     'Noun_adj_keys','summary_polarity','summary_subjectivity', 'summary_conflict'])