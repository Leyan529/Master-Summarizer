import re
import os
import torch


PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# special_tokens  = [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]
special_tokens  = [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]

class Vocab:
    def __init__(self, vocab_file, vocab_size):
        self._word2id, self._id2word = make_vocab(vocab_file, vocab_size)

    def word2id(self, word):
        return self._word2id[UNKNOWN_TOKEN] if word not in self._word2id else self._word2id[word]

    def id2word(self, idx):
        return UNKNOWN_TOKEN if idx >= self.size else self._id2word[idx]

    @property
    def size(self):
        return len(self._word2id)

def make_vocab(vocab_file, vocab_size):
    # Read the vocab file and add words up to max_size
    word2id, id2word = {}, {}
    for i, t in enumerate(special_tokens):
        word2id[t], id2word[i] = i, t

    with open(vocab_file, 'r') as vocab_f:
        for i,line in enumerate(vocab_f):
            pieces = line.split()
            if len(pieces) != 2:
                # print ('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                continue
            w = pieces[0]
            word2id[w], id2word[i+4] = (i+4), w
            if len(word2id) == vocab_size: break
    return word2id, id2word               

def article2ids(words, vocab):
    ids = []
    oovs = []
    for w in words:
        i = vocab.word2id(w)
        if i == vocab.word2id(UNKNOWN_TOKEN):
            if w not in oovs:
                oovs.append(w)
            ids.append(vocab.size + oovs.index(w))
        else: ids.append(i)

    return ids, oovs

def abstract2ids(words, vocab, article_oovs):
    ids = []
    for w in words:
        i = vocab.word2id(w)
        if i == vocab.word2id(UNKNOWN_TOKEN):
            if w in article_oovs:
                ids.append(vocab.size + article_oovs.index(w))
            else: ids.append(i)
        else: ids.append(i)
    return ids

# def output2words(ids, vocab, art_oovs):
#     words = []
#     for i in ids:
#         w = vocab.id2word(i) if i < vocab.size else art_oovs[i - vocab.size]
#         words.append(w)
#     return words

def output2words(ids, vocab, art_oovs):
    words = []
    for i in ids:
        if i < vocab.size:
            w = vocab.id2word(i) 
        elif len(art_oovs) > (i - vocab.size):
            w = art_oovs[i - vocab.size]
        else:
            w = UNKNOWN_TOKEN

        words.append(w)
    return words

def show_art_oovs(article, vocab):
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w)==vocab.word2id(UNKNOWN_TOKEN) else w for w in words]
    out_str = ' '.join(words)
    return out_str