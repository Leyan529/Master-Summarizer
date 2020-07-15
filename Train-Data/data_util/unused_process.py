# from textblob import TextBlob
# from textblob import Word
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
TextBlob = Blobber(analyzer=NaiveBayesAnalyzer())
# ----------------------------------------------------------------------------------------

# 排除英數字字元(可不做)
def clean_wordlist(wordlist):
    wordlist = [
        ''.join(re.findall(r'[A-Za-z]', word)) \
        if (word.isalnum() and not (word.isdigit()))
        else word
        for word in wordlist
    ]
    return wordlist

# ----------------------------------------------------------------------------------------
def TextBlob_feat(text):
    noun_phrases = TextBlob(text).noun_phrases
    f = []
    for phrase in noun_phrases:
        blob_phrase = TextBlob(phrase)
        for word, pos in blob_phrase.tags:
            if pos.startswith('N'):
                f.append(word)
    return list(set(f))

def TextBlob_noun_pharse_lemm(text):
    sentence = []
    # text = str(TextBlob(text).correct())
    for blob_sent in TextBlob(text).sentences:
        word_list = []
        for word, pos in blob_sent.correct().tags:
            if pos.startswith('N'):
                word_list.append(word.lemmatize("n"))
            elif pos.startswith('V'):
                word_list.append(word.lemmatize("v"))
            elif pos.startswith('R'):
                word_list.append(word.lemmatize("r"))
            else:
                word_list.append(word)
        sent = " ".join(word_list) + ' . '
        sentence.append(sent)
    text = "".join(sentence)
    feats = []
    feats = TextBlob_feat(text)
    return text, feats, sentence
# Too slow
def TextBlob_review_clean(text):
    text = lower_contraction(text)
    text, feats, sentence = TextBlob_noun_pharse_lemm(text)
    text = squeeze(text)
    return text, feats, sentence

def TextBlob_summary_clean(text):
    text = lower_contraction(text)    
    text, feats, sentence = TextBlob_noun_pharse_lemm(text)
    text = text.replace(" . "," ")
    text =  "<s> " + text + " </s>"
    text = squeeze(text)
    return text    