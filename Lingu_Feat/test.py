from spacy import displacy
from collections import Counter
import en_core_web_sm
import networkx as nx

import spacy
from nltk import Tree
from spacy import displacy

from show import find_shortest_path, show_tree, show_displacy

nlp = en_core_web_sm.load()

import re
from spacy.symbols import cop, acomp, amod, conj, neg, nn, nsubj, dobj,prep,advmod
from spacy.symbols import VERB, NOUN, PROPN, ADJ, ADV, AUX, PART
from spacy.matcher import Matcher
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
stopwords = list(stops)

def cp1(token, cp1_buffer, final):
    '''< h1, m > +conj and(h1, h2) →< h2, m >'''
    # This camera has great zoom and resolution
    # zoom(NN) ----amod----> great(JJ)
    # zoom(NN) ----conj----> resolution(NN)
    if token.head.tag_ in possible_noun and token.dep_ == 'conj':
        # cp1_buffer.add(token)
        # cp1_buffer.add(token.head)
        place_buffer(cp1_buffer, token)
    if token.head.tag_ in possible_noun and token.dep_ == 'amod':
        # cp1_buffer.add(token)
        # cp1_buffer.add(token.head)
        place_buffer(cp1_buffer, token)
    if final:
        f_list = list(set(cp1_buffer['f']))
        o_list = list(set(cp1_buffer['o']))
        if len(f_list) == 0 or len(o_list) == 0: return ()
        res = (f_list[0],o_list[0])
        return res
    return cp1_buffer

def cp2(token, cp2_buffer, final):
    '''cop(A, V ) + nsubj(A, N) →< N, A > '''
    # The camera case looks nice . 
    # looks(VBZ) ----nsubj----> case(NN)
    # looks(VBZ) ----acomp----> nice(JJ)
    if token.head.tag_ in possible_verb and token.dep_ == 'nsubj':
        # cp2_buffer.add(token)
        # cp2_buffer.add(token.head)
        place_buffer(cp2_buffer, token)
    if token.head.tag_ in possible_verb and token.dep_ == 'acomp':
        # cp2_buffer.add(token)
        # cp2_buffer.add(token.head)
        place_buffer(cp2_buffer, token)
    if final:
        f_list = list(set(cp2_buffer['f']))
        o_list = list(set(cp2_buffer['o']))
        if len(f_list) == 0 or len(o_list) == 0: return ()
        res = (f_list[0],o_list[1])
        return res
    return cp2_buffer

def cp3(token, cp3_buffer, final):
    '''cop(A, V ) + nsubj(A, N) →< N, A > '''
    # The screen wide and clear 
    # screen(NN) ----amod----> wide(JJ)
    # wide(JJ) ----conj----> clear(JJ)
    if token.head.tag_ in possible_adj and token.dep_ == 'conj':
        # cp3_buffer.add(token)
        # cp3_buffer.add(token.head)
        place_buffer(cp3_buffer, token)
    if token.head.tag_ in possible_noun and token.dep_ == 'amod':
        # cp3_buffer.add(token)
        # cp3_buffer.add(token.head)
        place_buffer(cp3_buffer, token)
    if final:
        f_list = list(set(cp3_buffer['f']))
        o_list = list(set(cp3_buffer['o']))
        if len(f_list) == 0 or len(o_list) == 0: return ()
        res = (f_list[0],o_list[0])
        return res
    return cp3_buffer

def cp4(token, cp4_buffer, final):
    '''dobj(V, N) + nsubj(V, N0) →< N, V > '''
    #  I love the picture quality 
    # quality(NN) ----compound----> picture(NN)
    # love(VBP) ----dobj----> quality(NN)
    if token.head.tag_ in possible_noun and token.dep_ == 'compound':
        # cp4_buffer.add(token)
        # cp4_buffer.add(token.head)
        place_buffer(cp4_buffer, token)
    if token.head.tag_ in possible_verb and token.dep_ == 'dobj':
        # cp4_buffer.add(token)
        # cp4_buffer.add(token.head)
        place_buffer(cp4_buffer, token)
    if final:
        f_list = list(set(cp4_buffer['f']))
        o_list = list(set(cp4_buffer['o']))
        if len(f_list) == 0 or len(o_list) == 0: return ()
        res = ("%s %s"%(f_list[0],o_list[0]), "%s %s"%(f_list[1],o_list[0]))
        return res
    return cp4_buffer

def cp5(token, cp5_buffer, final):
    '''< h1, m > +conj and(h1, h2) →< h2, m >   '''
    #  This camera has great zoom and resolution  
    # zoom(NN) ----conj----> resolution(NN)
    # zoom(NN) ----amod----> great(JJ)
    if token.head.tag_ in possible_noun and token.dep_ == 'conj':
        # cp5_buffer.add(token)
        # cp5_buffer.add(token.head)
        place_buffer(cp5_buffer, token)
    if token.tag_ in possible_adj and token.dep_ == 'amod':
        # cp5_buffer.add(token)
        # cp5_buffer.add(token.head)
        place_buffer(cp5_buffer, token)
    if final:
        f_list = list(set(cp5_buffer['f']))
        o_list = list(set(cp5_buffer['o']))
        if len(f_list) == 0 or len(o_list) == 0: return ()
        res = ("%s %s"%(f_list[0],o_list[0]), "%s %s"%(f_list[1],o_list[0]))
        return res
    return cp5_buffer

def cp6(token, cp6_buffer, final):
    '''< h, m1 > +conj and(m1, m2) →< h, m2 >  '''
    #  The screen wide and clear   
    # screen(NN) ----amod----> wide(JJ)
    # wide(JJ) ----conj----> clear(JJ)
    if token.head.tag_ in possible_adj and token.dep_ == 'conj':
        # cp6_buffer.add(token)
        # cp6_buffer.add(token.head)
        place_buffer(cp6_buffer, token)
    if token.head.tag_ in possible_noun and token.dep_ == 'amod':
        # cp6_buffer.add(token)
        # cp6_buffer.add(token.head)
        place_buffer(cp6_buffer, token)
    if final:
        f_list = list(set(cp6_buffer['f']))
        o_list = list(set(cp6_buffer['o']))
        if len(f_list) == 0 or len(o_list) == 0: return ()
        res = ("%s %s"%(f_list[0],o_list[0]), "%s %s"%(f_list[0],o_list[1]))
        return res
    return cp6_buffer

def cp7(token, cp7_buffer, final):
    '''< h, m1 > +conj and(m1, m2) →< h, m2 >  '''
    #  The battery life not long   
    # life(NN) ----compound----> battery(NN)
    # long(RB) ----neg----> not(RB)
    if token.head.tag_ in possible_noun and token.dep_ == 'compound':
        place_buffer(cp7_buffer, token)
    if token.head.tag_ in possible_adv and token.dep_ == 'neg':
        place_buffer(cp7_buffer, token)
    if final:
        f_list = list(set(cp7_buffer['f']))
        o_list = list(set(cp7_buffer['o']))
        if len(f_list) == 0 or len(o_list) == 0: return ()
        res = ("%s %s"%(f_list[0],f_list[1]), "%s %s"%(o_list[0],o_list[1]))
        return res
    return cp7_buffer

def cp8(token, cp8_buffer, final):
    '''< h, m1 > +conj and(m1, m2) →< h, m2 >  '''
    #  The camera case looks nice    
    # case(NN) ----compound----> camera(NN)
    # looks(VBZ) ----acomp----> nice(JJ)
    if token.head.tag_ in possible_noun and token.dep_ == 'compound':
        place_buffer(cp8_buffer, token)
    if token.head.tag_ in possible_verb and token.dep_ == 'acomp':
        place_buffer(cp8_buffer, token)
    if final:
        f_list = list(set(cp8_buffer['f']))
        o_list = list(set(cp8_buffer['o']))
        if len(f_list) == 0 or len(o_list) == 0: return ()
        return ("%s %s"%(f_list[1],f_list[0]), o_list[0])
    return cp8_buffer

def cp9(token, cp9_buffer, final):
    '''< h, m > +nn(N, h) →< h + N, m >   '''
    #   I love the picture quality     
    # quality(NN) ----compound----> picture(NN)
    # love(VBP) ----dobj----> quality(NN)
    if token.head.tag_ in possible_noun and token.dep_ == 'compound':
        place_buffer(cp9_buffer, token)
    if token.head.tag_ in possible_verb and token.dep_ == 'dobj':
        place_buffer(cp9_buffer, token)
    if final:
        f_list = list(set(cp9_buffer['f']))
        o_list = list(set(cp9_buffer['o']))
        if len(f_list) == 0 or len(o_list) == 0: return ()
        return ("%s %s"%(f_list[0],f_list[1]), o_list[0])
    return cp9_buffer

def check_f(token):
    return token.tag_ in possible_noun
def check_o(token):
    return token.tag_ in ( possible_adj + possible_verb + possible_adv )

def place_buffer(buffer, token):
    if check_f(token): 
        buffer_f = buffer['f']
        buffer_f.append(token.text)
        buffer['f'] = buffer_f
    if check_o(token): 
        buffer_o = buffer['o']
        buffer_o.append(token.text)
        buffer['o'] = buffer_o
        
    if check_f(token.head): 
        buffer_f = buffer['f']
        buffer_f.append(token.head.text)
        buffer['f'] = buffer_f
    if check_o(token.head): 
        buffer_o = buffer['o']
        buffer_o.append(token.head.text)
        buffer['o'] = buffer_o
    return buffer



import nltk
possible_verb = ['VB','MD','VBG','VBN','VBP','VBZ']
possible_adj = ["JJ","JJR"]
possible_noun = ["NN","NNP","NNPS","NNS"]
possible_adv = ['RB','RBR','RBS']
        
# cop, acomp, amod, conj, neg, nn, nsubj, dobj,prep,advmod

# zoom(NOUN) ----conj----> resolution(NOUN)
# zoom(NOUN) ----amod----> great(ADJ)

cp1_buffer, cp2_buffer, cp3_buffer = {'f':[],'o':[]}, {'f':[],'o':[]}, {'f':[],'o':[]}
cp4_buffer, cp5_buffer, cp6_buffer = {'f':[],'o':[]}, {'f':[],'o':[]}, {'f':[],'o':[]}
cp7_buffer, cp8_buffer, cp9_buffer = {'f':[],'o':[]}, {'f':[],'o':[]}, {'f':[],'o':[]}

sent = 'The camera case looks nice .'
doc = nlp(sent)
print('sentence: %s'%(doc))


for idx, token in enumerate(doc):
    final = False
    if idx == len(doc)-1: final = True

    cp1_buffer = cp1(token, cp1_buffer, final)
    cp2_buffer = cp2(token, cp2_buffer, final)
    cp3_buffer = cp3(token, cp3_buffer, final)
    cp4_buffer = cp4(token, cp4_buffer, final)
    cp5_buffer = cp5(token, cp5_buffer, final)
    cp6_buffer = cp6(token, cp6_buffer, final)
    cp7_buffer = cp7(token, cp7_buffer, final)
    cp8_buffer = cp8(token, cp8_buffer, final)
    cp9_buffer = cp9(token, cp9_buffer, final)

total_buffer = [cp1_buffer, cp2_buffer, cp3_buffer, cp4_buffer, cp5_buffer, cp6_buffer, cp7_buffer, cp8_buffer, cp9_buffer]
DEP_keywords = []
for buffer in total_buffer:
    if len(buffer) == 0: continue
    for item in buffer:
        DEP_keywords.append(item)
new_DEP_keywords = []
[new_DEP_keywords.extend(keywords.split(" ")) for keywords in DEP_keywords]
new_DEP_keywords = list(set(new_DEP_keywords))
print(new_DEP_keywords)