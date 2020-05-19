from spacy import displacy
from collections import Counter
import en_core_web_sm
import networkx as nx

import spacy
from nltk import Tree
from spacy import displacy

nlp = en_core_web_sm.load()

import re
from spacy.symbols import cop, acomp, amod, conj, neg, nn, nsubj, dobj,prep,advmod
from spacy.symbols import VERB, NOUN, PROPN, ADJ, ADV, AUX, PART
from spacy.matcher import Matcher

from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
stopwords = list(stops)

# FO-Rule(POS)
# pattern_id : (structure , f_pos_list, o_pos_list)
POS_Tag_Structure = {
                            # high quality image
                            'JJ[O] NN[F]/NNS[F]': ( [{'TAG': {"IN": ['JJ']}}, 
                                                    {'TAG': {"IN": ['NN','NNS']}}] , [1], [0]), 
                            # amazing battery life
                            'JJ[O] NN[F]/NNS[F] NN[F]/NNS[F]': ( [{'TAG': {"IN": ['JJ']}},
                                                                {'TAG': {"IN": ['NN','NNS']}}, 
                                                                {'TAG': {"IN": ['NN','NNS']}}] , [1,2], [0]),

                            # amazed with the image
                            'JJ[O] IN DT NN[F]': (  [{'TAG': {"IN": ['JJ']}},
                                                {'TAG': {"IN": ['IN']}},
                                                {'TAG': 'DT'},
                                                {'TAG': {"IN": ['NN']}}] , [3], [0]),
                            # Good and excellent camera .
                            'JJ[O] CC JJ[O] NN[F] ': (  [{'TAG': {"IN": ['JJ']}},
                                                {'TAG': {"IN": ['CC']}},
                                                {'TAG': {"IN": ['JJ']}},
                                                {'TAG': {"IN": ['NN']}}] , [3], [0,2]),
                            # Performance is excellent .                     
                            'NN[F] VB/VBZ JJ[O]': (   [{'TAG': {"IN": ['NN']}},
                                                {'TAG': {"IN": ['VB','VBZ']}},
                                                {'TAG': {"IN": ['JJ']}}] , [0], [2]),
                            # Keanu performs well .                     
                            'NN/NNP[F] NN/VBZ JJ/RB[O]': (   [{'TAG': {"IN": ['NN','NNP']}},
                                                {'TAG': {"IN": ['NN','VBZ']}},
                                                {'TAG': {"IN": ['JJ','RB']}}] , [0], [2]),
                            # Keanu performs very well.                     
                            'NN/NNP[F] VB/VBZ RB RB[O]': (   [{'TAG': {"IN": ['NN','NNP']}},
                                                {'TAG': {"IN": ['VB','VBZ']}},
                                                {'TAG': {"IN": ['RB']}},
                                                {'TAG': {"IN": ['RB']}}] , [0], [3]),
                            # Siri can sometimes help .                     
                            'NN/NNP[F] MD RB VB[O]': (   [{'TAG': {"IN": ['NN','NNP']}},
                                                {'TAG': {"IN": ['MD']}},
                                                {'TAG': {"IN": ['RB']}},
                                                {'TAG': {"IN": ['VB']}}] , [0], [3]),
                            # I love this phone .                      
                            'VB/VBP[O] DT NN[F]': (   [{'TAG': {"IN": ['VB','VBP']}},
                                                {'TAG': {"IN": ['DT']}},
                                                {'TAG': {"IN": ['NN']}}] , [2], [0]),
                            # Camera is worse than N81 .                     
                            'NN[F] VBZ JJR[O] IN NN/NNP': (   [{'TAG': {"IN": ['NN']}},
                                                {'TAG': {"IN": ['VBZ']}},
                                                {'TAG': {"IN": ['JJR']}},
                                                {'TAG': {"IN": ['IN']}},
                                                {'TAG': {"IN": ['NN','NNP']}}] , [0], [2]),

                            # Support Bluetooth                     
                            'VB[O] NN/NNP[F]': (   [{'TAG': {"IN": ['VB']}},
                                                {'TAG': {"IN": ['NN','NNP']}}] , [1], [0]),

                            # --------------------------------------------------------------

                            # very excellent volume .                     
                            'RB/RBR/RBS JJ[O]  NN/NNS[F]': ( [{'TAG': {"IN": ['RB','RBR','RBS']}},
                                                        {'TAG': {"IN": ['JJ']}},
                                                        {'TAG': {"IN": ['NN',"NNS"]}}
                                                        ] , [2], [1]),

                            # # honestly recommend .                     
                            # 'RB/RBR/RBS JJ[O]  NN/NNS[F]': ( [{'TAG': {"IN": ['RB','RBR','RBS']}},
                            #                                 {'TAG': {"IN": ['VBN','VBD','VB']}}
                            #                                 ] , [2], [1]),

                            # look excellent .                     
                            'VB[O] JJ[F]': ( [{'TAG': {"IN": ['VB','VBN','VBD']}},
                                            {'TAG': {"IN": ['JJ']}} 
                                            ] , [0], [1]),
                            # exhausted very quickly .                     
                            'VBN/VBD/VB[F] RB/RBR/RBS RB/RBR/RBS[O]': ( [{'TAG': {"IN": ['VBN','VBD','VB']}},
                                            {'TAG': {"IN": ['RB','RBR','RBS']}},
                                            {'TAG': {"IN": ['RB','RBR','RBS']}}
                                            ] , [0], [2]),

                            # our rule 
                            # image quality high
                            'NN[F] NN[F] JJ[O]': ( [{'TAG': {"IN": ['NN']}}, {'TAG': {"IN": ['NN']}}, 
                            {'TAG': {"IN": ['JJ']}}] , [0,1], [2])
}

class extract_POS():
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

        for pattern_id, item in POS_Tag_Structure.items():
            matcher.add(pattern_id, self.collect_sents, item[0])  # add POS_Tag_Structure pattern 

        doc = nlp(sent)
        info = [(token.text, token.tag_) for token in doc]
        # print(info)
        matches = matcher(doc)


        for match_id, start, end in matches:
            # Get the string representation
            pattern_id = nlp.vocab.strings[match_id]
            span = doc[start:end]  # The matched span
            pos = [(token.text,token.tag_) for token in doc]
            span_pos = pos[start:end]
            # print(match_id, pattern_id, start, end, span.text)
            # print(span_pos)
            fos = self.match_fos(pattern_id,span_pos) 
            for fo in fos:
                if fo not in res: 
                    res.append(fo)
        return res

    def match_fos(self,pattern_id,span_pos):
        f_pos_list, o_pos_list = POS_Tag_Structure[pattern_id][1:]
        if len(f_pos_list) > 1:
            f = [span_pos[pos][0] for pos in f_pos_list]
            f = " ".join(f)
        else: f = span_pos[f_pos_list[0]][0]

        if len(o_pos_list) > 1:
            o = [span_pos[pos][0] for pos in o_pos_list]
            o = " ".join(o)
        else: o = span_pos[o_pos_list[0]][0]

        if o in list(set(stopwords)): return ()
        else: return [(f,o)]
        # return [(f,o)]

    def run(self):
        ress = self.match_pattern(self.article)
        POS_keywords = ",".join(["%s %s"%(f, o) for f, o in ress])
        POS_keywords = list(set(POS_keywords.replace(",",' ').split(" ")))

        merge = " ".join(POS_keywords)
        info = [(token.text, token.tag_) for token in nlp(merge)]
        # print(info)
        pfs = [token.text for token in nlp(merge) if token.tag_.startswith('N')]
        # print(pfs)
        return POS_keywords, pfs

class extract_DEP():
    def __init__(self, article):
        # self.article = article
        # self.matched_sents = []  # Collect data of matched sentences to be visualized
        self.possible_verb = ['VB','MD','VBG','VBN','VBP','VBZ']
        self.possible_adj = ["JJ","JJR"]
        self.possible_noun = ["NN","NNP","NNPS","NNS"]
        self.possible_adv = ['RB','RBR','RBS']
        # article = article.replace('is ','')
        article = article + ' .'
        doc = nlp(article)

        cp1_buffer, cp2_buffer, cp3_buffer = {'f':[],'o':[]}, {'f':[],'o':[]}, {'f':[],'o':[]}
        cp4_buffer, cp5_buffer, cp6_buffer = {'f':[],'o':[]}, {'f':[],'o':[]}, {'f':[],'o':[]}
        cp7_buffer, cp8_buffer, cp9_buffer = {'f':[],'o':[]}, {'f':[],'o':[]}, {'f':[],'o':[]}

        for idx, token in enumerate(doc):
            final = False
            if idx == len(doc)-1: final = True

            cp1_buffer = self.cp1(token, cp1_buffer, final)
            cp2_buffer = self.cp2(token, cp2_buffer, final)
            cp3_buffer = self.cp3(token, cp3_buffer, final)
            cp4_buffer = self.cp4(token, cp4_buffer, final)
            cp5_buffer = self.cp5(token, cp5_buffer, final)
            cp6_buffer = self.cp6(token, cp6_buffer, final)
            cp7_buffer = self.cp7(token, cp7_buffer, final)
            cp8_buffer = self.cp8(token, cp8_buffer, final)
            cp9_buffer = self.cp9(token, cp9_buffer, final)

        total_buffer = [cp1_buffer, cp2_buffer, cp3_buffer, cp4_buffer, cp5_buffer, cp6_buffer, cp7_buffer, cp8_buffer, cp9_buffer]
        DEP_keywords = []
        for buffer in total_buffer:
            if len(buffer) == 0: continue
            for item in buffer:
                DEP_keywords.append(item)
        new_DEP_keywords = []
        [new_DEP_keywords.extend(keywords.split(" ")) for keywords in DEP_keywords]
        new_DEP_keywords = list(set(new_DEP_keywords))
        self.res = new_DEP_keywords
    
    def run(self):
        
        merge = " ".join(self.res)
        info = [(token.text, token.tag_) for token in nlp(merge)]
        # print(info)

        pfs = [token.text for token in nlp(merge) if token.tag_.startswith('N')]
        # print(pfs)
        return self.res, pfs

    def cp1(self, token, cp1_buffer, final):
        '''< h1, m > +conj and(h1, h2) →< h2, m >'''
        # This camera has great zoom and resolution
        # zoom(NN) ----amod----> great(JJ)
        # zoom(NN) ----conj----> resolution(NN)
        if token.head.tag_ in self.possible_noun and token.dep_ == 'conj':
            # cp1_buffer.add(token)
            # cp1_buffer.add(token.head)
            self.place_buffer(cp1_buffer, token)
        if token.head.tag_ in self.possible_noun and token.dep_ == 'amod':
            # cp1_buffer.add(token)
            # cp1_buffer.add(token.head)
            self.place_buffer(cp1_buffer, token)
        if final:
            f_list = list(set(cp1_buffer['f']))
            o_list = list(set(cp1_buffer['o']))
            if len(f_list) == 0 or len(o_list) == 0: return ()
            res = (f_list[0],o_list[0])
            return res
        return cp1_buffer

    def cp2(self, token, cp2_buffer, final):
        '''cop(A, V ) + nsubj(A, N) →< N, A > '''
        # The camera case looks nice . 
        # looks(VBZ) ----nsubj----> case(NN)
        # looks(VBZ) ----acomp----> nice(JJ)
        if token.head.tag_ in self.possible_verb and token.dep_ == 'nsubj':
            # cp2_buffer.add(token)
            # cp2_buffer.add(token.head)
            self.place_buffer(cp2_buffer, token)
        if token.head.tag_ in self.possible_verb and token.dep_ == 'acomp':
            # cp2_buffer.add(token)
            # cp2_buffer.add(token.head)
            self.place_buffer(cp2_buffer, token)
        if final:
            f_list = list(set(cp2_buffer['f']))
            o_list = list(set(cp2_buffer['o']))
            if (len(f_list) == 0) or (len(o_list) <2): return ()
            res = (f_list[0],o_list[1])
            return res
        return cp2_buffer

    def cp3(self, token, cp3_buffer, final):
        '''cop(A, V ) + nsubj(A, N) →< N, A > '''
        # The screen wide and clear 
        # screen(NN) ----amod----> wide(JJ)
        # wide(JJ) ----conj----> clear(JJ)
        if token.head.tag_ in self.possible_adj and token.dep_ == 'conj':
            # cp3_buffer.add(token)
            # cp3_buffer.add(token.head)
            self.place_buffer(cp3_buffer, token)
        if token.head.tag_ in self.possible_noun and token.dep_ == 'amod':
            # cp3_buffer.add(token)
            # cp3_buffer.add(token.head)
            self.place_buffer(cp3_buffer, token)
        if final:
            f_list = list(set(cp3_buffer['f']))
            o_list = list(set(cp3_buffer['o']))
            if len(f_list) == 0 or len(o_list) == 0: return ()
            res = (f_list[0],o_list[0])
            return res
        return cp3_buffer

    def cp4(self, token, cp4_buffer, final):
        '''dobj(V, N) + nsubj(V, N0) →< N, V > '''
        #  I love the picture quality 
        # quality(NN) ----compound----> picture(NN)
        # love(VBP) ----dobj----> quality(NN)
        if token.head.tag_ in self.possible_noun and token.dep_ == 'compound':
            # cp4_buffer.add(token)
            # cp4_buffer.add(token.head)
            self.place_buffer(cp4_buffer, token)
        if token.head.tag_ in self.possible_verb and token.dep_ == 'dobj':
            # cp4_buffer.add(token)
            # cp4_buffer.add(token.head)
            self.place_buffer(cp4_buffer, token)
        if final:
            f_list = list(set(cp4_buffer['f']))
            o_list = list(set(cp4_buffer['o']))
            if len(f_list) <2 or len(o_list) == 0: return ()
            res = ("%s %s"%(f_list[0],o_list[0]), "%s %s"%(f_list[1],o_list[0]))
            return res
        return cp4_buffer

    def cp5(self, token, cp5_buffer, final):
        '''< h1, m > +conj and(h1, h2) →< h2, m >   '''
        #  This camera has great zoom and resolution  
        # zoom(NN) ----conj----> resolution(NN)
        # zoom(NN) ----amod----> great(JJ)
        if token.head.tag_ in self.possible_noun and token.dep_ == 'conj':
            # cp5_buffer.add(token)
            # cp5_buffer.add(token.head)
            self.place_buffer(cp5_buffer, token)
        if token.tag_ in self.possible_adj and token.dep_ == 'amod':
            # cp5_buffer.add(token)
            # cp5_buffer.add(token.head)
            self.place_buffer(cp5_buffer, token)
        if final:
            f_list = list(set(cp5_buffer['f']))
            o_list = list(set(cp5_buffer['o']))
            if len(f_list) <2 or len(o_list) == 0: return ()
            res = ("%s %s"%(f_list[0],o_list[0]), "%s %s"%(f_list[1],o_list[0]))
            return res
        return cp5_buffer

    def cp6(self, token, cp6_buffer, final):
        '''< h, m1 > +conj and(m1, m2) →< h, m2 >  '''
        #  The screen wide and clear   
        # screen(NN) ----amod----> wide(JJ)
        # wide(JJ) ----conj----> clear(JJ)
        if token.head.tag_ in self.possible_adj and token.dep_ == 'conj':
            # cp6_buffer.add(token)
            # cp6_buffer.add(token.head)
            self.place_buffer(cp6_buffer, token)
        if token.head.tag_ in self.possible_noun and token.dep_ == 'amod':
            # cp6_buffer.add(token)
            # cp6_buffer.add(token.head)
            self.place_buffer(cp6_buffer, token)
        if final:
            f_list = list(set(cp6_buffer['f']))
            o_list = list(set(cp6_buffer['o']))
            if len(f_list) == 0 or len(o_list) < 2: return ()
            res = ("%s %s"%(f_list[0],o_list[0]), "%s %s"%(f_list[0],o_list[1]))
            return res
        return cp6_buffer

    def cp7(self, token, cp7_buffer, final):
        '''< h, m1 > +conj and(m1, m2) →< h, m2 >  '''
        #  The battery life not long   
        # life(NN) ----compound----> battery(NN)
        # long(RB) ----neg----> not(RB)
        if token.head.tag_ in self.possible_noun and token.dep_ == 'compound':
            self.place_buffer(cp7_buffer, token)
        if token.head.tag_ in self.possible_adv and token.dep_ == 'neg':
            self.place_buffer(cp7_buffer, token)
        if final:
            f_list = list(set(cp7_buffer['f']))
            o_list = list(set(cp7_buffer['o']))
            if len(f_list) < 2 or len(o_list) <2: return ()
            res = ("%s %s"%(f_list[0],f_list[1]), "%s %s"%(o_list[0],o_list[1]))
            return res
        return cp7_buffer

    def cp8(self, token, cp8_buffer, final):
        '''< h, m1 > +conj and(m1, m2) →< h, m2 >  '''
        #  The camera case looks nice    
        # case(NN) ----compound----> camera(NN)
        # looks(VBZ) ----acomp----> nice(JJ)
        if token.head.tag_ in self.possible_noun and token.dep_ == 'compound':
            self.place_buffer(cp8_buffer, token)
        if token.head.tag_ in self.possible_verb and token.dep_ == 'acomp':
            self.place_buffer(cp8_buffer, token)
        if final:
            f_list = list(set(cp8_buffer['f']))
            o_list = list(set(cp8_buffer['o']))
            if len(f_list) <2 or len(o_list) == 0: return ()
            return ("%s %s"%(f_list[1],f_list[0]), o_list[0])
        return cp8_buffer

    def cp9(self, token, cp9_buffer, final):
        '''< h, m > +nn(N, h) →< h + N, m >   '''
        #   I love the picture quality     
        # quality(NN) ----compound----> picture(NN)
        # love(VBP) ----dobj----> quality(NN)
        if token.head.tag_ in self.possible_noun and token.dep_ == 'compound':
            self.place_buffer(cp9_buffer, token)
        if token.head.tag_ in self.possible_verb and token.dep_ == 'dobj':
            self.place_buffer(cp9_buffer, token)
        if final:
            f_list = list(set(cp9_buffer['f']))
            o_list = list(set(cp9_buffer['o']))
            if len(f_list) <2 or len(o_list) == 0: return ()
            return ("%s %s"%(f_list[0],f_list[1]), o_list[0])
        return cp9_buffer

    def check_f(self, token):
        return token.tag_ in self.possible_noun
    def check_o(self, token):
        return token.tag_ in ( self.possible_adj + self.possible_verb + self.possible_adv )

    def place_buffer(self, buffer, token):
        if self.check_f(token): 
            buffer_f = buffer['f']
            buffer_f.append(token.text)
            buffer['f'] = buffer_f
        if self.check_o(token): 
            buffer_o = buffer['o']
            buffer_o.append(token.text)
            buffer['o'] = buffer_o
            
        if self.check_f(token.head): 
            buffer_f = buffer['f']
            buffer_f.append(token.head.text)
            buffer['f'] = buffer_f
        if self.check_o(token.head): 
            buffer_o = buffer['o']
            buffer_o.append(token.head.text)
            buffer['o'] = buffer_o
        return buffer

def noun_adj(sent):
    # array of sentence_noun_adj pairs
    noun_adj_pairs = []

    # array consisting of noun after adjective
    checked = []

    doc = nlp(sent, "utf-8")
    # detect noun after adjective
    for i,token in enumerate(doc):
        if token.pos_ in ('NOUN','PROPN'):
        # if token.tag_ in possible_noun:
            # print(str(token))
            for j in range(0,len(doc)):
                if doc[j].pos_ == 'ADJ' and doc[j - 1].pos_ == 'ADV' and j == i - 1:
                # if doc[j].tag_ in possible_adj and doc[j - 1].tag_ in possible_adv and j == i - 1:
                    checked.append(str(token))
                    noun_adj_pairs.append((str(doc),str(token),str(doc[j - 1]) + ' ' + str(doc[j])))
                    break
                elif doc[j].pos_ == 'ADJ' and doc[j - 1].pos_ != 'ADV' and j == i - 1:
                # elif doc[j].tag_ in possible_adj and doc[j - 1].tag_ not in possible_adv and j == i - 1:
                    checked.append(str(token))
                    noun_adj_pairs.append((str(doc),str(token),str(doc[j])))
                    break

    # detect noun before adjective
    for i,token in enumerate(doc):
        if str(token) not in checked:
            if token.pos_ not in ('NOUN','PROPN'):
            # if token.pos_ not in possible_noun:    
                continue
            for j in range(i + 1,len(doc)):
                if doc[j].pos_ == 'ADJ':
                # if doc[j].tag_ in possible_adj:    
                    noun_adj_pairs.append((str(doc),str(token),str(doc[j])))
                    break
                    
    f_list = list(set([item[1] for item in noun_adj_pairs]))
    o_list = list(set([item[2] for item in noun_adj_pairs]))
    return (f_list + o_list) , f_list