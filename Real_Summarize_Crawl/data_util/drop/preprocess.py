import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from data_util.stopwords import *

import re

import spacy
# gpu = spacy.prefer_gpu()
# print('GPU:', gpu)
from spacy.matcher import Matcher
import en_core_web_sm

# nlp = en_core_web_sm.load(disable = ['ner', 'tagger','parser'])
nlp = en_core_web_sm.load()

from nltk.corpus import stopwords
# from stopwords import *

stops = set(stopwords.words("english"))
stpwords_list3 = [f.replace("\n", "") for f in open("data_util/stopwords.txt", "r", encoding="utf-8").readlines()]
stpwords_list3.remove("not")
total_stopwords = list(html_escape_table + stpwords_list2) + list(
    list(stops) + list(stpwords_list1) + list(stpwords_list3))
total_stopwords = set(total_stopwords)
total_stopwords.remove("the")
total_stopwords.remove("not")

ent_list = ['PERSON',
            'NORP',
            'FAC',
            'ORG',
            'GPE',
            'LOC',
            'PRODUCT',
            'EVENT',
            'WORK_OF_ART',
            'LAW',
            'LANGUAGE',
            'DATE',
            'TIME',
            'PERCENT',
            'MONEY',
            'QUANTITY',
            'ORDINAL',
            'CARDINAL']

i = 0


def SentProcess(text):
    newText = ""
    text = text.replace(",", "").replace("i.e.", "").replace("\n", "")
    for sent in text.split("."):
        sent = sent.strip()

        for k, v in contractions.items():
            if k in sent:
                sent = sent.replace(k, v)

        for k in html_escape_table:
            if k in sent:
                sent = sent.replace(k, "")
        sent = remove_word2(sent)
        if len(sent) < 5: continue
        if len(sent) == "": continue
        newText = newText + sent + ".<end>"
    return newText


# lemm process (不加逗號，已斷句)
def lemm_sent_process(text, remove_stopwords=False, summary=False, mode="nltk", withdot=False):
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
                # ------------------------------------------------------------------
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
                        if word in list(list(stops) + list(stpwords_list1) + list(stpwords_list3)): continue
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
                        text_lines.append(line + "\n")
                    # --------------------------------------------------------------------------------
            # spacy
            elif mode == "spacy":
                # line = remove_word2(remove_word(line))
                doc = nlp(line)
                # 通過設置doc.is_parsed = True來欺騙spaCy忽略它，即通過讓它相信分配了依賴關係解析並以這種方式應用了句子邊界
                # doc.is_parsed = True
                sentences = " ".join(
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
                         (word.lemma_ not in list(html_escape_table + stpwords_list2)) and
                         not (word.is_stop and remove_stopwords) and
                         word.tag_ != "SYM" and word.tag_ != "HYPH" and
                         word.lemma_ != "-PRON-" and
                         (word.lemma_ not in stpwords_list3)
                         ) or word.lemma_ in ["the", "not"])
                    ]
                )

                if sentences == "": continue
                if withdot == True:
                    text_lines.append(sentences + ".\n")
                else:
                    text_lines.append(sentences + "\n")
    # --------------------------------------------------------------------------------
    text_lines = "".join(text_lines)
    return text_lines


# lemm process (不加逗號，已斷句)
def lemm_sent_process2(text, remove_stopwords=False, summary=False, mode="nltk", withdot=False):
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
    pos_lines = []
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        else:
            line = remove_word2(remove_word(line))
            # nltk
            if mode == "nltk":
                # ------------------------------------------------------------------
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
                        if word in list(list(stops) + list(stpwords_list1) + list(stpwords_list3)): continue
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
                        text_lines.append(line + "\n")
                    # --------------------------------------------------------------------------------
            # spacy
            elif mode == "spacy":
                # line = remove_word2(remove_word(line))
                doc = nlp(line)
                # 通過設置doc.is_parsed = True來欺騙spaCy忽略它，即通過讓它相信分配了依賴關係解析並以這種方式應用了句子邊界
                # doc.is_parsed = True
                sentences = " ".join(
                    [
                        word.lemma_ if word.pos_.startswith('N') or word.pos_.startswith('J') or
                                       word.pos_.startswith('V') or word.pos_.startswith('R') else word.orth_
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
                         word.ent_type_ not in ent_list and
                         (word.lemma_ not in list(html_escape_table + stpwords_list2)) and
                         (word.orth_ not in list(html_escape_table + stpwords_list2)) and
                         #                              not (word.is_stop and remove_stopwords) and
                         word.tag_ not in ["SYM", "HYPH"] and
                         word.lemma_ != "-PRON-"
                         #                              word.lemma_ not in total_stopwords
                         ))
                    ]
                )
                pos_sent = ["%s/%s" % (word.lemma_, word.tag_)
                            if ((word.tag_.startswith('N') or word.tag_.startswith(
                    'J') or word.tag_.startswith('V') or word.tag_.startswith('R')))
                            else "%s/%s" % (word.orth_, word.tag_)
                            for word in doc
                            if (word.lemma_ not in list(html_escape_table + stpwords_list2)) and
                            (word.orth_ not in list(html_escape_table + stpwords_list2)) and
                            word.ent_type_ not in ent_list
                            ]
                #                 pos_sent = ' '.join([tup[1] + '/' + tup[0] for tup in pos_sent])
                #                 print(pos_sent)
                pos_sent = ' '.join(pos_sent)
                if sentences == "": continue
                if withdot == True:  sentences = sentences + "."
                if summary: sentences = ' <s> ' + sentences + " </s> "
                sentences = sentences + "\n"			
                
                text_lines.append(sentences)

            # if not summary:
            #                     if withdot == True:
            #                         text_lines.append(sentences + ".\n")
            #                     else:
            #                         text_lines.append(sentences+"\n")
            #                 else:
            #                     text_lines.append(sentences+"\n")
            pos_lines.append(pos_sent)
        # --------------------------------------------------------------------------------


    text_lines = "".join(text_lines)
    return text_lines, pos_lines

def lemm_sent_process3(text, remove_stopwords=False, summary=False, mode="nltk", withdot=False):
    nlp = en_core_web_sm.load()
    stops = set(stopwords.words("english"))
    
    lines = nltk.sent_tokenize(text)
    # Separate out reviewText and summary sentences
    text_lines = []
    pos_lines = []
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        else:
            # line = remove_word2(remove_word(line))
            # spacy
            if mode == "spacy":
                # line = remove_word2(remove_word(line))
                doc = nlp(line)
                # 通過設置doc.is_parsed = True來欺騙spaCy忽略它，即通過讓它相信分配了依賴關係解析並以這種方式應用了句子邊界
                # doc.is_parsed = True
                sentences = " ".join(
                    [
                        word.lemma_ if word.pos_.startswith('N') or word.pos_.startswith('J') or
                                       word.pos_.startswith('V') or word.pos_.startswith('R') else word.orth_
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
                         word.ent_type_ not in ent_list and
                         (word.lemma_ not in list(html_escape_table + stpwords_list2)) and
                         (word.orth_ not in list(html_escape_table + stpwords_list2)) and
                         #                              not (word.is_stop and remove_stopwords) and
                         word.tag_ not in ["SYM", "HYPH"] and
                         word.lemma_ != "-PRON-"
                         #                              word.lemma_ not in total_stopwords
                         ))
                    ]
                )
                
                if sentences == "": continue
                if withdot == True:  sentences = sentences + "."
                if summary: sentences = ' <s> ' + sentences + " </s> "
                sentences = sentences + "\n"			
                
                text_lines.append(sentences)

    text_lines = "".join(text_lines)
    return text_lines

def lemm_sent_process4(text, remove_stopwords=False, summary=False, mode="nltk", withdot=False):
		stops = set(stopwords.words("english"))
		# print('xxx')
		lines = nltk.sent_tokenize(text)
		# Separate out reviewText and summary sentences
		text_lines = []
		pos_lines = []
		for idx, line in enumerate(lines):
			if line == "":
				continue  # empty line
			else:
				# line = remove_word2(remove_word(line))
				# spacy
				if mode == "spacy":
					# line = remove_word2(remove_word(line))
					doc = nlp(line)
					# 通過設置doc.is_parsed = True來欺騙spaCy忽略它，即通過讓它相信分配了依賴關係解析並以這種方式應用了句子邊界
					# doc.is_parsed = True
					sentences = " ".join(
						[
							word.lemma_ if word.pos_.startswith('N') or word.pos_.startswith('J') or
										word.pos_.startswith('V') or word.pos_.startswith('R') else word.orth_
							for word in doc
							if (
							(not word.is_space and
							not word.is_bracket and
							# not word.is_digit and
							not word.is_left_punct and
							not word.is_right_punct and
							not word.is_bracket and
							not word.is_quote and
							not word.is_currency 
							# not word.is_punct and							
							# word.tag_ not in ["SYM", "HYPH"] and
							# word.lemma_ != "-PRON-"
							))
						]
					)
					
					if sentences == "": continue
					# if withdot == True:  sentences = sentences + "."
					if summary: sentences = ' <s> ' + sentences + " </s> "
					sentences = sentences + "\n"			
					if not withdot: sentences = sentences.replace(" . "," ")

					text_lines.append(sentences)
					

		text_lines = "".join(text_lines)
		return text_lines

def lemm_sent_process5(text,summary=False):
    doc = nlp(text)
    # 通過設置doc.is_parsed = True來欺騙spaCy忽略它，即通過讓它相信分配了依賴關係解析並以這種方式應用了句子邊界
    # doc.is_parsed = True
    lines = nltk.sent_tokenize(text)
    text_lines = []
    for line in lines:
        sentence = " ".join(
            [
                word.lemma_ if word.pos_.startswith('N') or word.pos_.startswith('J') or
                            word.pos_.startswith('V') or word.pos_.startswith('R') else word.orth_
                for word in doc
                if (
                (not word.is_space and
                not word.is_bracket and
                # not word.is_digit and
                not word.is_left_punct and
                not word.is_right_punct and
                not word.is_bracket and
                not word.is_quote and
                not word.is_currency 
                # not word.is_punct and							
                # word.tag_ not in ["SYM", "HYPH"] and
                # word.lemma_ != "-PRON-"
                ))
            ]
        )
        if summary:
            sentence = '<s> ' + sentence + " </s>"
        text_lines.append(sentence)
    return " ".join(text_lines)

def lemm_sent_process6(text,summary=False):
    doc = nlp(text)
    # 通過設置doc.is_parsed = True來欺騙spaCy忽略它，即通過讓它相信分配了依賴關係解析並以這種方式應用了句子邊界
    # doc.is_parsed = True
    # lines = nltk.sent_tokenize(text)
    lines = list(doc.sents)
    text_lines = []
    # print(lines)
    for line in lines:
        sentence = " ".join(
            ['<s>']+
            [
                word.lemma_ if word.pos_.startswith('N') or word.pos_.startswith('J') or
                            word.pos_.startswith('V') or word.pos_.startswith('R') else word.orth_
                for word in doc
                if (
                (not word.is_space and
                not word.is_bracket and
                # not word.is_digit and
                not word.is_left_punct and
                not word.is_right_punct and
                not word.is_bracket and
                not word.is_quote and
                not word.is_currency 
                # not word.is_punct and							
                # word.tag_ not in ["SYM", "HYPH"] and
                # word.lemma_ != "-PRON-"
                ))
            ] 
            + ['</s>']
        )
        # if summary:
        #     sentence = '<s> ' + sentence + " </s>"
        text_lines.append(sentence)
        text_lines = text_lines[:len(lines)-1]
    # print(text_lines)
    return " ".join(text_lines)    


def getRelations(text):
    doc = nlp(text)
    feat_depend_info = []
    for token in doc:
        if (token.tag_.startswith("N")) and (
                        token.head.tag_.startswith("J") or token.head.tag_.startswith(
                        "V") or token.head.tag_.startswith(
                    "N")):
            if str(token) != str(token.head) and (token.dep_ in ["acl", "acomp", "advcl", "amod", "appos",
                                                                 "ccomp", "compound", "conj", "npmod", "prep",
                                                                 "quantmod", "partmod", "nn", "npadvmod",
                                                                 "dobj", "neg", "nsubj"]):

                info = "{4}({0}/{2} -->{1}/{3})".format(str(token.head), str(token), token.head.tag_, token.tag_,
                                                        str(token.dep_) + str(descr_label[str(token.dep_).upper()]))
                # print(str(descr_label[str(token.dep_).upper()]))

                feat_depend_info.append(info)
            else:
                continue
    return feat_depend_info


def replace_sent_feat(sent, feats):
    for feat in feats:
        if feat + "(" in sent:
            sent = sent.replace(feat + "(", feat + "(f/")
    return sent + "\n"


def replace_sent_pharsefeat(sent, pf_list):
    for pf_pharse in pf_list:
        if pf_pharse + "(" in sent:
            sent = sent.replace(pf_pharse + "(", pf_pharse + "(/f")
    return sent + "\n"


def replace_article_PCfeat(article, PC):
    for complex_feat in PC:
        if len(complex_feat) == 1:
            complex_feat = " ".join(complex_feat)
            if complex_feat + "(" in article:
                article = article.replace(complex_feat + "(", complex_feat + "(f/")

    # for complex_feat in PC:
    #     if len(complex_feat) > 1:
    #         complex_feat = " ".join(complex_feat)
    #         if complex_feat in article:
    #             article = article.replace(complex_feat, "[" + complex_feat + "](f) ")
    return article


def replace_article_opinion(article, FOS):
    for _, opinion in FOS:
        if opinion + "(" in article:
            article = article.replace(opinion + "(", opinion + "(o/")
    return article


def replace_article_fo(article, PC, FOS):
    article = replace_article_PCfeat(article, PC)
    article = replace_article_opinion(article, FOS)
    return article


def make_array_single_dimension(l):
    l2 = []

    for x in l:
        if type(x).__name__ == "list":
            l2 += make_array_single_dimension(x)
        else:
            if len(x) > 3:
                l2.append(x)

    return l2


"""
reference https://spacy.io/usage/rule-based-matching
matched_sents
"""


def collect_sents(matcher, doc, i, matches):
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
    matched_sents.append({"text": sent.text, "ents": match_ents})


def match_pattern(sent):
    matcher = Matcher(nlp.vocab)
    matched_sents = []  # Collect data of matched sentences to be visualized
    pattern = [{"LOWER": "facebook"}, {"LEMMA": "be"}, {"POS": "ADV", "OP": "*"},
               {"POS": "ADJ"}]
    matcher.add("FacebookIs", collect_sents, pattern)  # add pattern
    doc = nlp("I'd say that Facebook is evil. – Facebook is pretty cool, right?")
    matches = matcher(doc)

    # Serve visualization of sentences containing match with displaCy
    # set manual=True to make displaCy render straight from a dictionary
    # (if you're not running the code within a Jupyer environment, you can
    # use displacy.serve instead)
    displacy.render(matched_sents, style="ent", manual=True)
    sent = nltk.word_tokens(sent)
    for items in matches:
        id, st, en = items
        print(sent[st:en])  # match segment


def show_pos_sent(text):
    lines = text.split("\n")
    pos_article = []
    for sent in lines:
        pos_sent = []
        # word_tokens = pos_tag(word_tokens)
        for token in nltk.word_tokenize(sent):
            word, pos = pos_tag([token])[0]
            pos_sent.append("%s(%s)" % (word, pos))
        pos_sent = " ".join(pos_sent)
        pos_article.append(pos_sent)
    pos_article = "\n".join(pos_article)
    return pos_article


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
    text = re.sub(r'(?:^| )\w(?:$| )', " ", text).strip()  # remove single alphbet
    text = text.replace("\\", ' ')
    text = text.replace("/", ' ')
    text = text.replace(",", ' ')
    text = text.replace("?", ' ')
    text = " ".join(text.split())
    return text


def remove_word3(text):
    text = text.lower()
    text = text.replace("-", ' ')
    remove_chars = '[0-9’!"#$%&\'\"\,()*+:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, "", text)  # remove number and segment
    text = text.replace("\\", ' ')
    text = text.replace("/", ' ')
    text = text.replace(",", ' ')
    text = re.sub(r'(?:^| )\w(?:$| )', " ", text).strip()  # remove single alphbet
    #     text = " ".join(text.split())
    return text

def remove_word4(text):
    text = str(text)
    text = text.lower()

    for k, v in contractions.items():
        if k in text:
            text = text.replace(k, v)

    for k in html_escape_table:
        if k in text:
            text = text.replace(k, "")

    text = text.replace("-", ' ')
    text = text.replace(".\n", ' . ')
    text = text.replace(". ", ' . ')
    text = text.replace("\"", '')
    text = text.replace("\n", '') 
    
    '''
    remove_chars = '[’"#$%&\'\"\,()*+:<=>?@★【】《》“”‘’[\\]^_`{|}~]+'
    text = re.sub(remove_chars, "", text)  # remove number and segment
    '''
    remove_chars = '["#$%&\'\"\()*+:<=>?@★【】《》“”‘’[\\]^_`{|}~]+'
    text = re.sub(remove_chars, "", text)  # remove number and segment

    remove_chars = '[!;，。?、…？！]+'
    text = re.sub(remove_chars, " . ", text)  # remove number and segment
    text = text.replace("\\", ' ')
    text = text.replace("/", ' ')
    text = text.replace(",", ' ')
    # text = re.sub(r'(?:^| )\w(?:$| )', " ", text).strip()  # remove single alphbet
    text = " ".join([t for t in text.split(" ") if t !=""])
    #     text = " ".join(text.split())
    return text

def calc_len(s):
    # # s = s.replace(" . "," ")
    # s = s.replace(".", " ")
    # s = s.replace("  ", " ")
    return len(word_tokenize(s))


def get_pos_sequence(text, remove_stopwords=False, summary=False, mode="spacy", withdot=False):
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
                # ------------------------------------------------------------------
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
                line = remove_word2(remove_word(line))
                if line == "": continue
                if summary:
                    text_lines.append(" <s> " + line + " </s> ")
                else:
                    if withdot == True:
                        text_lines.append(line + ".\n")
                    else:
                        text_lines.append(line + "\n")
                    # --------------------------------------------------------------------------------
            # spacy
            elif mode == "spacy":
                doc = nlp(line)
                # 通過設置doc.is_parsed = True來欺騙spaCy忽略它，即通過讓它相信分配了依賴關係解析並以這種方式應用了句子邊界
                # doc.is_parsed = True
                sent = [
                    (word.tag_, word.orth_)
                    if ((word.tag_.startswith('N') or word.tag_.startswith(
                        'J') or word.tag_.startswith('V') or word.tag_.startswith('R')))
                    else (word.tag_, word.orth_)
                    for word in doc
                    # if((not word.is_space and
                    #      not word.is_bracket and
                    #      not word.is_digit and
                    #      not word.is_left_punct and
                    #      not word.is_right_punct and
                    #      not word.is_bracket and
                    #      not word.is_quote and
                    #      not word.is_currency and
                    #      not word.is_punct and
                    #      not (word.is_stop and remove_stopwords) and
                    #      word.tag_ != "SYM" and word.tag_ != "HYPH" and word.tag_ != "HYPH" and word.ent_type_ == ""
                    #      and word.ent_type_ not in ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT",
                    #                                 "EVENT",
                    #                                 "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME",
                    #                                 "PERCENT",
                    #                                 "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
                    #      and word.lemma_ != "-PRON-" and (word.lemma_ not in stpwords_list2)))

                ]

                if sent == "": continue
                text_lines.append(sent)
    if len(text_lines) != 0:
        pos_lines = []
        for line in text_lines:
            pos_sent = ' '.join([word + '/' + pos for pos, word in line])
            pos_lines.append(pos_sent)

        return text_lines, pos_lines
    else:
        return text_lines, None


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


descr_label = {"ACL": "名詞的從句修飾語",
               "ACOMP": "形容詞補語",
               "ADVCL": "狀語從句修飾語",
               "ADVMOD": "狀語",
               "AGENT": "代理人",
               "AMOD": "形容詞修飾語",
               "APPOS": "介詞修飾語",
               "ATTR": "屬性",
               "AUX": "輔助的",
               "AUXPASS": "輔助（被動）",
               "CASE": "案例標記",
               "CC": "並列連詞",
               "CCOMP": "子句補語",
               "COMPOUND": "複合修飾詞",
               "CONJ": "連詞",
               "CSUBJ": "子句主語",
               "CSUBJPASS": "從句（被動）",
               "DATIVE": "和格",
               "DEP": "未分類的依賴",
               "DET": "確定者",
               "DOBJ": "直接賓語",
               "EXPL": "侵略性",
               "INTJ": "感嘆詞",
               "MARK": "標記",
               "META": "元修飾符號",
               "NEG": "取反修飾符號",
               "NOUNMOD": "標稱修飾符號",
               "NPMOD": "名詞短語作為副詞修飾語",
               "NSUBJ": "名義主題",
               "NSUBJPASS": "標稱主題（被動）",
               "NUMMOD": "數字修飾符號",
               "OPRD": "對象謂詞",
               "PARATAXIS": "準軸",
               "PCOMP": "介詞的補語",
               "POBJ": "介詞賓語",
               "POSS": "擁有修飾符號",
               "PRECONJ": "相關前合取",
               "PREDET": "預定器",
               "PREP": "介詞修飾語",
               "PRT": "粒子",
               "PUNCT": "標點",
               "QUANTMOD": "量詞修飾語",
               "RELCL": "相對從句修飾符號",
               "ROOT": "根",
               "XCOMP": "開放式補語",
               "COMPLM": "補體",
               "INFMOD": "不定式修飾符號",
               "PARTMOD": "分詞修飾語",
               "HMOD": "連字符修飾符號",
               "HYPH": "連字號",
               "IOBJ": "間接賓語",
               "NUM": "數字修飾符號",
               "NUMBER": "數字複合修飾符號",
               "NMOD": "標稱修飾符號",
               "NN": "名詞複合修飾語",
               "NPADVMOD": "名詞短語作為副詞修飾語",
               "POSSESSIVE": "所有格修飾語",
               "RCMOD": "相對從句修飾符號"}


# ------------------------------------------------------------
def get_Lemm_NLTK(text, remove_stopwords=False, summary=False):
    text = text.lower()
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
                # if word not in stop_words:
                wordnet_pos = get_wordnet_pos(tag)  # or wordnet.NOUN
                if wordnet_pos != None:
                    lemmatize_word = wnl.lemmatize(word, pos=wordnet_pos)  # 词形还原
                    filtered_sentence.append(lemmatize_word)
                else:
                    filtered_sentence.append(word)

            line = " ".join(filtered_sentence)
            if summary:
                text_lines.append(" <s> " + line + " </s> ")
            else:
                text_lines.append(line)

    # Make reviewText into a single string
    text = ' '.join(text_lines)

    # text = " ".join(sentences)
    text = remove_word2(text)

    # Optionally, remove stop words

    text = text.split()
    stops = set(stopwords.words("english"))
    if remove_stopwords:
        text = [w for w in text if not (w in (stops or stpwords_list1 or stpwords_list2))]
    else:
        text = [w for w in text if not (w in (html_escape_table or stpwords_list2))]
    text = " ".join(text)
    text = text + "."
    return text


def squeeze(s):
    char = " ."
    while char * 2 in s:
        s = s.replace(char * 2, char)

    char = "."
    while char * 2 in s:
        s = s.replace(char * 2, char)

    char = " "
    while char * 2 in s:
        s = s.replace(char * 2, char)

    # s = s.replace(" .",".")
    s = re.sub(r'(?:^| )\w(?:$| )', " ", s).strip()  # remove single alphbet
    s = s.replace(".", " .")
    s = s.replace("  ", " ")
    s = s.replace(" .", ".")
    return s

def squeeze2(s):  
    char = " "
    while char * 2 in s:
        s = s.replace(char * 2, char)

    char = "."
    while char * 2 in s:
        s = s.replace(char * 2, char)
    return s

def squeeze3(s):     
    char = "."
    while char * 2 in s:
        s = s.replace(char * 2, char)
    return s


def get_Lemm_Spacy(text, remove_stopwords=False):
    text = text.lower()
    for k, v in contractions.items():
        if k in text:
            text = text.replace(k, v)

    for k in html_escape_table:
        if k in text:
            text = text.replace(k, "")

    doc = nlp(remove_word(text))
    # 通過設置doc.is_parsed = True來欺騙spaCy忽略它，即通過讓它相信分配了依賴關係解析並以這種方式應用了句子邊界
    # doc.is_parsed = True
    sentences = [
        " ".join(
            [
                word.lemma_ if word.pos_.startswith('N') or word.pos_.startswith('J') or word.pos_.startswith(
                    'V') or word.pos_.startswith('R') else word.orth_
                for word in sent
                if (
                (not word.is_space and
                 not word.is_bracket and
                 not word.is_digit and
                 not word.is_left_punct and
                 not word.is_right_punct and
                 not word.is_bracket and
                 not word.is_quote and
                 not word.is_currency and
                 #          not word.is_punct and
                 not (word.is_stop and remove_stopwords) and
                 word.tag_ != "SYM" and word.tag_ != "HYPH" and word.tag_ != "HYPH" and word.ent_type_ == "" and
                 word.lemma_ != "-PRON-" and (word.lemma_ not in stpwords_list2)))
            ]
        )
        for sent in doc.sents]

    text = " ".join(sentences)
    text = remove_word2(text)
    text = text + "."
    return text


def get_pos_sequence2(text, remove_stopwords=False):
    text = text.lower()
    for k, v in contractions.items():
        if k in text:
            text = text.replace(k, v)

    for k in html_escape_table:
        if k in text:
            text = text.replace(k, "")

    doc = nlp(remove_word(text))
    # 通過設置doc.is_parsed = True來欺騙spaCy忽略它，即通過讓它相信分配了依賴關係解析並以這種方式應用了句子邊界
    # doc.is_parsed = True
    sentences = [
        [
            (word.pos_, word.lemma_) if word.pos_.startswith('N') or word.pos_.startswith('J') or word.pos_.startswith(
                'V') or word.pos_.startswith('R') else (word.pos_, word.orth_)
            for word in sent
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
             word.tag_ != "SYM" and word.tag_ != "HYPH" and word.tag_ != "HYPH" and word.ent_type_ == ""
             and word.ent_type_ not in ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
                                        "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT",
                                        "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
             and word.lemma_ != "-PRON-" and (word.lemma_ not in stpwords_list2)))
        ]
        for sent in doc.sents]
    return sentences
    # ------------------------------------------------------------

    #