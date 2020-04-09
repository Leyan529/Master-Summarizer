#Most of this file is copied form https://github.com/abisee/pointer-generator/blob/master/data.py

import glob
import random
import struct
import csv
from . import config
from tensorflow.core.example import example_pb2

import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# <s> and </s> are used in the data files to segment the summarys into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
# START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
# STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
# STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences
START_DECODING = '[unused0]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[unused1]' # This has a vocab id, which is used at the end of untruncated target sequences

BERT_CLS = '[CLS]'
BERT_SEP = '[SEP]'

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.
# from pytorch_pretrained_bert import BertTokenizer, TransfoXLTokenizer
from transformers import BertModel, BertTokenizer 
from transformers import TransfoXLTokenizer, TransfoXLModel

# Bert
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)  

num_added_bert_toks = bert_tokenizer.add_tokens([UNKNOWN_TOKEN, 
START_DECODING, STOP_DECODING])
print('We have added', num_added_bert_toks, 'tokens')

# Transformers-XL
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

num_added_xl_toks = tokenizer.add_tokens([UNKNOWN_TOKEN, 
START_DECODING, STOP_DECODING])

print('We have added', num_added_xl_toks, 'tokens')

import torchsnooper

class Vocab(object):
  def __init__(self, vocab_file, max_size):
    
    self.tokenizer = bert_tokenizer
    # self.tokenizer = tokenizer
    self.tokenizer.max_len = config.max_enc_steps

    vocab_dict = {}
    vocab_dict = { k : v for v , k in bert_tokenizer.vocab.items()}
    self.vocab = vocab_dict

  # def word2id(self, word):
  #   # if isinstance(word, (bytes, bytearray)):
  #   #   word= word.decode('utf-8')
  #   if word not in self._word_to_id:
  #     return self._word_to_id[UNKNOWN_TOKEN]
  #   return self._word_to_id[word]
  
  def word2id(self, review_words):
    # if isinstance(word, (bytes, bytearray)):
    #   word= word.decode('utf-8')  
    if type(review_words) == str: 
      review_words = [review_words]
      return self.tokenizer.convert_tokens_to_ids(review_words)[0]
    return self.tokenizer.convert_tokens_to_ids(review_words)

  def exist_Keyword(self,key_word_str):
    return self.tokenizer.tokenize(key_word_str)

  def key2id(self, word):
    # if isinstance(word, (bytes, bytearray)):
    #   word= word.decode('utf-8')
    if type(review_words) == str: review_words = [review_words]
    return self.tokenizer.convert_tokens_to_ids(review_words)

  def id2word(self, word_id):
    # print(word_id,type(word_id))
    # print(self.vocab)
    # if word_id not in self.vocab:
    #   raise ValueError('Id not found in vocab: %d' % word_id)
    # return self.vocab[word_id]
    return self.tokenizer._convert_id_to_token(word_id)

  def size(self):
    return len(self.vocab)

  def write_metadata(self, fpath):
    print("Writing word embedding metadata file to %s..." % (fpath))
    with open(fpath, "w",encoding="utf-8") as f:
      fieldnames = ['word']
      writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
      for i in range(self.size()):
        writer.writerow({"word": self.vocab[i]})


def example_generator(data_path, single_pass):
  while True:
    filelist = glob.glob(data_path) # get the list of datafiles
    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    if single_pass:
      filelist = sorted(filelist)
    else:
      random.shuffle(filelist)
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
#         print(example_str)
        try:
            yield example_pb2.Example.FromString(example_str)
        except:
            continue
    if single_pass:
      # print ("example_generator completed reading all datafiles. No more data.")
      break

# @torchsnooper.snoop()
def review2ids(review_words, vocab):
  # review_words = [b'purchase', b'this', b'on', b'december', b'and...b'etc', b'be', b'very', b'happy', b'with', b'it']
  ids = []
  oovs = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in review_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is OOV
      if w not in oovs: # Add to list of OOVs  (OOVs 不在原始字典中的word)
        oovs.append(w) # oovs = [b'purchase', b'this']
      oov_num = oovs.index(w) # This is 0 for the first review OOV, 1 for the second review OOV...
      ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first review OOV, 50001 for the second...
    else:
      ids.append(i)
  return ids, oovs

# @torchsnooper.snoop()
def summary2ids(summary_words, vocab, review_oovs):
  ids = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in summary_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is an OOV word
      if w in review_oovs: # If w is an in-review OOV
        vocab_idx = vocab.size() + review_oovs.index(w) # Map to its temporary review OOV number
        ids.append(vocab_idx)
      else: # If w is an out-of-review OOV
        ids.append(unk_id) # Map to the UNK token id
    else:
      ids.append(i)
  return ids


def outputids2words(id_list, vocab, review_oovs):
  words = []
  # print(id_list)
  # print(review_oovs)
  # print('-------------')
  for i in id_list:
    # try:
    #   w = vocab.id2word(i) # might be [UNK]
    # except ValueError as e: # w is OOV
    #   assert review_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
    #   review_oov_idx = i - vocab.size()
    #   try:
    #     if review_oov_idx >= len(review_oovs): continue
    #     w = review_oovs[review_oov_idx]
    #   except ValueError as e: # i doesn't correspond to an review oov
    #     raise ValueError('Error: model produced word ID %i which corresponds to review OOV %i but this example only has %i review OOVs' % (i, review_oov_idx, len(review_oovs)))
    w = vocab.id2word(i) # might be [UNK]
    words.append(w)
  # print(words)
  return words


def summary2sents(summary):
  cur = 0
  sents = []
  while True:
    try:
      start_p = summary.index(SENTENCE_START.encode(), cur)
      end_p = summary.index(SENTENCE_END.encode(), start_p + 1)
      cur = end_p + len(SENTENCE_END)
      sents.append(summary[start_p+len(SENTENCE_START):end_p])
      if len(sents) > 1: print(sents)
    except ValueError as e: # no more sentences
      return sents


def show_art_oovs(review, vocab):
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = review.split(' ')
  words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
  out_str = ' '.join(words)
  return out_str


def show_abs_oovs(summary, vocab, review_oovs):
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = summary.split(' ')
  new_words = []
  for w in words:
    if vocab.word2id(w) == unk_token: # w is oov
      if review_oovs is None: # baseline mode
        new_words.append("__%s__" % w)
      else: # pointer-generator mode
        if w in review_oovs:
          new_words.append("__%s__" % w)
        else:
          new_words.append("!!__%s__!!" % w)
    else: # w is in-vocab word
      new_words.append(w)
  out_str = ' '.join(new_words)
  return out_str
