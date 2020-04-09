import re
import os
import torch
from transformers import BertTokenizer , TransfoXLTokenizer
from utils import config
import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# special_tokens  = [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]
special_tokens  = [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]


BERT_CLS = '[CLS]'
BERT_SEP = '[SEP]'

# Bert
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)  
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_tokenizer.add_special_tokens({'bos_token':START_DECODING,'eos_token':STOP_DECODING,'unk_token':UNKNOWN_TOKEN})
num_added_bert_toks = bert_tokenizer.add_tokens([UNKNOWN_TOKEN, START_DECODING, STOP_DECODING])
# [START] 30524 [START] 
# [STOP] 30525 [STOP]
# [UNK] 100 [UNK]
# print('We have added', num_added_bert_toks, 'bert tokens')
print('We have ', len(bert_tokenizer), 'bert tokens now')

# Transformers-XL
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

num_added_xl_toks = tokenizer.add_tokens([UNKNOWN_TOKEN, START_DECODING, STOP_DECODING])
# [UNK] 267735 [UNK]
# [START] 267736 [START] 
# [STOP] 267737 [STOP]
print('We have added', num_added_xl_toks, 'XL tokens')

class Vocab:
    def __init__(self, vocab_size):
        self.tokenizer = bert_tokenizer    
        # self.tokenizer = tokenizer
        self.tokenizer.max_len = config.max_enc_steps 
        self._word2id, self._id2word = self.make_vocab(vocab_size)

    def word2id(self, word):
        return self._word2id[UNKNOWN_TOKEN] if word not in self._word2id else self._word2id[word]
        # return self.tokenizer.convert_tokens_to_ids(word)

    def id2word(self, idx):
        return UNKNOWN_TOKEN if idx >= self.size else self._id2word[idx]
        # return self.tokenizer._convert_id_to_token(idx)

    @property
    def size(self):
        return len(self._word2id)

    def make_vocab(self,vocab_size):    
        # word2id, id2word = {}, {}
        # bert_tokenizer.add_special_tokens({'bos_token':START_DECODING,'eos_token':STOP_DECODING,'unk_token':UNKNOWN_TOKEN})
        # num_added_bert_toks = bert_tokenizer.add_tokens([UNKNOWN_TOKEN, START_DECODING, STOP_DECODING])
        id2word = { k : v for v , k in self.tokenizer.vocab.items()}
        word2id = { k : v for k , v in self.tokenizer.vocab.items()}
        for idx,w in enumerate([START_DECODING,STOP_DECODING]):
            word2id[w] = len(word2id)
            id2word[len(word2id)-1] = w

        return word2id, id2word


def article2ids(words, vocab):
    # print('words',len(words))
    ids = []
    oovs = []
    for w in words:
        i = vocab.word2id(w)
        if i == vocab.word2id(UNKNOWN_TOKEN):
            if w not in oovs:
                oovs.append(w)
            ids.append(vocab.size + oovs.index(w))
        else: ids.append(i)
    # print('ids',len(ids))
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

def output2words(ids, vocab, art_oovs):
    words = []
    for i in ids:
        w = vocab.id2word(i) if i < vocab.size else art_oovs[i - vocab.size]
        words.append(w)
    return words

def show_art_oovs(article, vocab):
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w)==vocab.word2id(UNKNOWN_TOKEN) else w for w in words]
    out_str = ' '.join(words)
    return out_str