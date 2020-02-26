import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from stopwords import *
from nltk.corpus import stopwords
import re

import spacy
import en_core_web_sm
# nlp = en_core_web_sm.load(disable = ['ner', 'tagger','parser'])
nlp = en_core_web_sm.load()

# nlp.add_pipe(nlp.create_pipe('sentencizer'))

i = 0
# --------------------------------------------------------------
def removeSpecWord(document):  # 搜索關鍵保留字，並重新清理句子
    keywords = []
    newdocument = ""
    for sent in nltk.sent_tokenize(document):

        sent = sent.lower()
        for k, v in contractions.items():
            if k in sent:
                sent = sent.replace(k, v)

        for k in html_escape_table:
            if k in sent:
                sent = sent.replace(k, "")

        for token in nlp(sent):
            temp_keyword = []
            symbol = []
            if re.search("\d", str(token)):
                temp_keyword.append(str(token))
            if "-" in str(token) or "/" in str(token):
                symbol.append(str(token))
                if token.pos_ not in ["PUNCT", "SYM"]:
                    temp_keyword.append(str(token))
            if re.search(re.compile('[0-9’!"#$%&\'\"\,()*+:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}-~]+'), str(token)):
                symbol.append(str(token))
                if token.pos_ not in ["PUNCT", "SYM"]:
                    temp_keyword.append(str(token))
            if token.pos_ in ["PUNCT", "SYM"]:
                symbol.append(str(token))
            keywords.extend(temp_keyword)

        cleansent = sent
        for k in keywords:
            cleansent = cleansent.replace(str(k), "")
            cleansent = remove_word2(cleansent)
            cleansent = cleansent.strip()
        cleansent = remove_word2(" ".join(cleansent.split(" "))) + ".\n"
        newdocument = newdocument + cleansent

    # print("\ndocument\n",document)
    # print("\nnewdocument\n",newdocument)
    # print("---------------------------------")
    return keywords, newdocument



# lemm process (不加逗號，已斷句)
def lemm_sent_process(text,remove_stopwords=False,summary=False,mode = "nltk",withdot =False):
    text = text.lower()
    stops = set(stopwords.words("english"))
    for k, v in contractions.items():
        if k in text:
            text = text.replace(k, v)

    for k in html_escape_table:
        if k in text:
            text = text.replace(k, "")

    lines = nltk.sent_tokenize(text)
    # Separate out reviewText and summary sentences
    text_lines = []
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        else:
            line = remove_word2(remove_word(line))
            # nltk
            if mode == "nltk":
          #------------------------------------------------------------------
                word_tokens = word_tokenize(line)
                new_word_tokens = []
                for w in word_tokens:
                    new_word_tokens.append(w)

                word_tokens = new_word_tokens
                word_tokens = pos_tag(word_tokens)  # 获取单词词性
                wnl = WordNetLemmatizer()
                # print(line)
                filtered_sentence = []
                for w in word_tokens:
                    word, tag = w[0], w[1]
                    if word in list(html_escape_table + stpwords_list2): continue
                    if remove_stopwords:
                        if word in list(list(stops) + list(stpwords_list1)): continue
                    # if word not in stop_words:
                    wordnet_pos = get_wordnet_pos(tag)  # or wordnet.NOUN
                    if wordnet_pos != None:
                        lemmatize_word = wnl.lemmatize(word, pos=wordnet_pos)  # 词形还原
                        filtered_sentence.append(lemmatize_word)
                    else:
                        filtered_sentence.append(word)

                # filtered_sentence = [sent for sent in filtered_sentence if sent != "" ]
                line = " ".join(filtered_sentence)
                # line = remove_word2(remove_word(line))
                if line == "": continue
                if summary:
                    text_lines.append(" <s> " + line + " </s> ")
                else:
                    if withdot == True:
                        text_lines.append(line + ".\n")
                    else:
                        text_lines.append(line)
         #--------------------------------------------------------------------------------
            # spacy
            elif mode == "spacy":
                # line = remove_word2(remove_word(line))
                doc = nlp(line)
                # 通過設置doc.is_parsed = True來欺騙spaCy忽略它，即通過讓它相信分配了依賴關係解析並以這種方式應用了句子邊界
                # doc.is_parsed = True
                sentences =  " ".join(
                        [
                            word.lemma_ if word.pos_.startswith('N') or word.pos_.startswith('J') or word.pos_.startswith(
                                'V') or word.pos_.startswith('R') else word.orth_
                            for word in doc
                            if (
                            (not word.is_space and
                             not word.is_bracket and
                             not word.is_digit and
                             not word.is_left_punct and
                             not word.is_right_punct and
                             not word.is_bracket and
                             not word.is_quote and
                             not word.is_currency and
                             not word.is_punct and
                             not (word.is_stop and remove_stopwords) and
                             word.tag_ != "SYM" and word.tag_ != "HYPH" and
                             word.lemma_ != "-PRON-" and
                             (word.lemma_ not in stpwords_list2)
                             ))
                        ]
                    )
                if sentences == "":continue
                if withdot == True:
                    text_lines.append(sentences + ".\n")
                else:
                    text_lines.append(sentences+"\n")
    # --------------------------------------------------------------------------------
    text_lines = "".join(text_lines)
    return text_lines

# --------------------------------------------------------------
def remove_word(text):
    text = text.lower()
    text = text.replace("-", ' ')
    remove_chars = '[0-9’!"#$%&\'\"\,()*+:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, "", text)  # remove number and segment
    text = text.replace("\\", ' ')
    text = text.replace("/", ' ')
    text = text.replace(",", ' ')
    text = " ".join(text.split())
    return text

def remove_word2(text):
    text = text.lower()
    text = text.replace("/", " ")
    text = text.replace(".", ' ')
    text = text.replace("  ", ' ')
    text = text.replace("-", ' ')

    #     text = text.replace("\n", "").replace("\t", "")
    remove_chars = '[0-9’!"#$%&\'\"\,()*+:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, "\n", text)  # remove number and segment
    text = re.sub("(\s\d+)", " ", text)
    #     text = re.sub(r'(?:^| )\w(?:$| )', " ", text).strip() # remove single alphbet
    text = text.replace("\\", ' ')
    text = text.replace("/", ' ')
    text = text.replace(",", ' ')
    text = " ".join(text.split())
    return text

def squeeze(s):
    char = " ."
    while char*2 in s:
        s=s.replace(char*2,char)

    char = "."
    while char * 2 in s:
        s = s.replace(char * 2, char)

    char = " "
    while char * 2 in s:
        s = s.replace(char * 2, char)

    #s = s.replace(" .",".")
    s = re.sub(r'(?:^| )\w(?:$| )', " ", s).strip() # remove single alphbet
    s = s.replace("."," .")
    s = s.replace("  "," ")
    s = s.replace(" .",".")
    return s
def calc_len(s):
    # # s = s.replace(" . "," ")
    # s = s.replace(".", " ")
    # s = s.replace("  ", " ")
    return len(word_tokenize(s))

# ------------------------------------------------------------
# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None




