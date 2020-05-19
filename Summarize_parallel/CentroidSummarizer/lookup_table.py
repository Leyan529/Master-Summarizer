import gensim
import numpy as np
import os

class LookupTable:
    def __init__(self, model_path):
        
        # self.model = gensim.models.Word2Vec.load_word2vec_format(os.path.abspath(model_path), binary=True,
        #                                                          unicode_errors='ignore')
        # print(model_path)
        self.model  = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False, encoding='utf-8')                                                                 

    def vec(self, word):
        try:
            return self.model[word]
        except KeyError:
            return np.array([0])

    def unseen(self, word):
        try:
            self.model[word]
            return False
        except KeyError:
            return True
