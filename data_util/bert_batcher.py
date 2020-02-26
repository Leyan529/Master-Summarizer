#Most of this file is copied form https://github.com/abisee/pointer-generator/blob/master/batcher.py

import queue as Queue
import time
from random import shuffle
from threading import Thread

import numpy as np
import tensorflow as tf

from . import config
from . import bert_data

import random
random.seed(1234)

import torchsnooper

import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

from copy import deepcopy
tokenizer = bert_data.bert_tokenizer
# tokenizer = bert_data.tokenizer


# tokenizer.add_tokens([bert_data.UNKNOWN_TOKEN, bert_data.START_DECODING, bert_data.STOP_DECODING])

class Example(object):

  # @torchsnooper.snoop()
  def __init__(self, review, summary_sentences, keywords, vocab):
  # def __init__(self, review, summary_sentences,orign_review_text, vocab):
    # Get ids of special tokens
    start_decoding = vocab.word2id(bert_data.START_DECODING) # start_decoding = 30524
    stop_decoding = vocab.word2id(bert_data.STOP_DECODING) # stop_decoding = 30525 
    # print(start_decoding)
    # print(stop_decoding)
  
    review_ids = tokenizer.encode(review, max_length = config.max_enc_steps, add_special_tokens=True)
    review_words = tokenizer.convert_ids_to_tokens(review_ids)
    self.enc_len = len(review_ids)
    self.enc_input = review_ids

    key_ids = tokenizer.encode(keywords,max_length=config.max_key_num, add_special_tokens=False)
    key_words = " ".join(tokenizer.convert_ids_to_tokens(key_ids))
    self.enc_key_len = len(key_ids)  # store the length after truncation but before padding
    self.enc_key_input = key_ids   # list of keyword ids; NO UNK token

    summary = ' '.join(summary_sentences)  # string # summary = 'solid value performer'
    summary_words = summary.split()  # list of strings # summary_words = ['solid', 'value', 'performer']
    # sum_ids = [202, 5, 681, 5301, 6343] # summary_words = ['sony', 'be', 'fm', 'rec', 'dh']
    sum_ids = tokenizer.encode(summary_words, add_special_tokens=False) # list of word ids; OOVs are represented by the id for UNK token  # Get the decoder input sequence and target sequence

    # 將label summary拆為dec_input序列，並加上start_decoding符號
    self.dec_input, _ = self.get_dec_inp_targ_seqs(sum_ids, config.max_dec_steps, start_decoding, stop_decoding)
    self.dec_len = len(self.dec_input)
 
    # If using pointer-generator mode, we need to store some extra info
    # Store a version of the enc_input where in-review OOVs are represented by their temporary OOV id; also store the in-review OOVs words themselves
    # 將 review_words拆解成 enc_input_extend_vocab (在字典裡的字) , review_oovs (不在字典裡的字)
    self.enc_input_extend_vocab, self.review_oovs = bert_data.review2ids(review_words, vocab)
   
    # Get a verison of the reference summary where in-review OOVs are represented by their temporary review OOV id
    # 將 summary_words只表示成 vocab 和 review_oovs出現的單詞 => sum_ids_extend_vocab
    sum_ids_extend_vocab = bert_data.summary2ids(summary_words, vocab, self.review_oovs)


    # 將label summary拆為序列，並加上start_decoding符號 # Get decoder target sequence 
    _, self.target = self.get_dec_inp_targ_seqs(sum_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)

  # Store the original strings
    self.original_review = review
    self.original_summary = summary
    self.original_summary_sents = summary_sentences
    self.key_words = key_words


  # @torchsnooper.snoop()
  # call by class Example (__init__)
  def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
    # sequence = [430, 371, 2782]
    # max_len = 100
    # start_id = 2
    # stop_id = 3
    inp = [start_id] + sequence[:] # inp = [2, 430, 371, 2782]
    target = sequence[:] # target = [430, 371, 2782]
    if len(inp) > max_len: # truncate
      inp = inp[:max_len]
      target = target[:max_len] # no end_token
    else: # no truncation
      target.append(stop_id) # end token # target = [430, 371, 2782, 3]
    assert len(inp) == len(target)
    return inp, target # Return value:.. ([2, 430, 371, 2782], [430, 371, 2782, 3])

  # @torchsnooper.snoop()
  # call by class Batch (init_decoder_seq)
  def pad_decoder_inp_targ(self, max_len, pad_id):
    while len(self.dec_input) < max_len:
      self.dec_input.append(pad_id)
    while len(self.target) < max_len:
      self.target.append(pad_id)

  # @torchsnooper.snoop()
  # call by class Batch (init_encoder_seq)
  def pad_encoder_input(self, max_len, pad_id):
    while len(self.enc_input) < max_len:
      self.enc_input.append(pad_id)
    while len(self.enc_input_extend_vocab) < max_len:
      self.enc_input_extend_vocab.append(pad_id)


  # @torchsnooper.snoop()
  # call by class Batch (init_encoder_seq)
  def pad_encoder_key_input(self, max_len, pad_id):
    while len(self.enc_key_input) < max_len:
      self.enc_key_input.append(pad_id)
    # while len(self.enc_input_extend_vocab) < max_len:
    #   self.enc_input_extend_vocab.append(pad_id)


class Batch(object):  # call by class Batcher (fill_batch_queue)
  # @torchsnooper.snoop()
  def __init__(self, example_list, vocab, batch_size):
    self.batch_size = batch_size
    self.pad_id = vocab.word2id(bert_data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.example_list = example_list
    self.init_encoder_seq(example_list) # initialize the input to the encoder
    self.init_decoder_seq(example_list) # initialize the input and targets for the decoder
    self.store_orig_strings(example_list) # store the original strings

  # @torchsnooper.snoop()
  def init_encoder_seq(self, example_list):
      # Determine the maximum length of the encoder input sequence in this batch
      max_enc_seq_len = max([ex.enc_len for ex in example_list])
      # Determine the maximum length of the encoder keyword input sequence in this batch
      max_enc_key_len = max([ex.enc_key_len for ex in example_list])
      # print(max_enc_key_len,'\n')

      # Pad the encoder input sequences up to the length of the longest sequence
      for i,ex in  enumerate(example_list):
        ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
        # print(i,ex.enc_input,len(ex.enc_input))
        # print('-------------------------\n')

      # Pad the encoder keyword input sequences up to the length of the longest keyword sequence
      for i,ex in  enumerate(example_list):
        ex.pad_encoder_key_input(max_enc_key_len, self.pad_id) # topical keywords 不pad到最大長度
        # print(i,ex.enc_key_input,len(ex.enc_key_input))
        # print('-------------------------\n')
      # Initialize the numpy arrays
      # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
      self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
      self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
      self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)
  
      # Initialize the numpy arrays
      # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
      self.enc_key_batch = np.zeros((self.batch_size, max_enc_key_len), dtype=np.int32)
      self.enc_key_lens = np.zeros((self.batch_size), dtype=np.int32)
      self.enc_key_padding_mask = np.zeros((self.batch_size, max_enc_key_len), dtype=np.float32)

      # Fill in the numpy arrays
      for i, ex in enumerate(example_list):
        self.enc_batch[i, :] = ex.enc_input[:]
        # print(ex.enc_input[:])
        self.enc_lens[i] = ex.enc_len
        # print(max_enc_seq_len,len(ex.enc_input[:]))
        for j in range(ex.enc_len):
          self.enc_padding_mask[i][j] = 1 # 對非空token word進行遮罩處理 (1:ignore ,0: mask)

      # Fill in the numpy arrays
      for i, ex in enumerate(example_list):
        self.enc_key_batch[i, :] = ex.enc_key_input[:]
        # print(ex.enc_key_input[:])
        self.enc_key_lens[i] = ex.enc_key_len
        for j in range(ex.enc_key_len):
          self.enc_key_padding_mask[i][j] = 1  # 對非空token word進行遮罩處理 (1:ignore ,0: mask)

      # For pointer-generator mode, need to store some extra info
      # Determine the max number of in-review OOVs in this batch
      self.max_rev_oovs = max([len(ex.review_oovs) for ex in example_list])
      # Store the in-review OOVs themselves
      self.rev_oovs = [ex.review_oovs for ex in example_list]

      # Store the version of the enc_batch that uses the review OOV ids
      # 初始化每個batch的擴充字典dim(batch_size, max_enc_seq_len)
      self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
      for i, ex in enumerate(example_list):
        # 紀錄batch review中的每個統計單詞表
        self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

  def init_decoder_seq(self, example_list):
    # Pad the inputs and targets
    for ex in example_list:
      ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

    # Initialize the numpy arrays.
    self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    # self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
    self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.dec_batch[i, :] = ex.dec_input[:]
      self.target_batch[i, :] = ex.target[:]
      self.dec_lens[i] = ex.dec_len
      # for j in range(ex.dec_len):
      #   self.dec_padding_mask[i][j] = 1

  def store_orig_strings(self, example_list):
    self.original_reviews = [ex.original_review for ex in example_list] # list of lists
    self.original_summarys = [ex.original_summary for ex in example_list] # list of lists
    self.key_words = [ex.key_words for ex in example_list] # list of lists
    self.original_summarys_sents = [ex.original_summary_sents for ex in example_list] # list of list of lists


class Batcher(object):
  BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

  # @torchsnooper.snoop()
  def __init__(self, data_path, vocab, mode, batch_size, single_pass):
    self._data_path = data_path
    self._vocab = vocab
    self._single_pass = single_pass
    self.mode = mode
    self.batch_size = batch_size
    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

    # Different settings depending on whether we're in single_pass mode or not
    if single_pass:
      self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
      self._num_batch_q_threads = 1  # just one thread to batch examples
      self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
      self._finished_reading = False # this will tell us when we're finished reading the dataset
    else:
      self._num_example_q_threads = 1 #16 # num threads to fill example queue
      self._num_batch_q_threads = 1 #4  # num threads to fill batch queue
      self._bucketing_cache_size = 1 #100 # how many batches-worth of examples to load into cache before bucketing

    # Start the threads that load the queues
    self._example_q_threads = []
    for _ in range(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue)) # thread call fill_example_queue
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in range(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue)) # thread call fill_batch_queue
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    # Start a thread that watches the other threads and restarts them if they're dead
    if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
      self._watch_thread = Thread(target=self.watch_threads)
      self._watch_thread.daemon = True
      self._watch_thread.start()

    # print("Init Batcher finished")

  # @torchsnooper.snoop()
  def next_batch(self):
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
      # tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
      if self._single_pass and self._finished_reading:
        tf.logging.info("Finished reading dataset in single_pass mode.")
        return None

    batch = self._batch_queue.get() # get the next Batch
    return batch

  # @torchsnooper.snoop()
  # call by Batcher.__init__ (execute by thread)
  def fill_example_queue(self):
    # input_gen = <generator object Batcher.text_generator at 0x000001A2EE447200>
    input_gen = self.text_generator(bert_data.example_generator(self._data_path, self._single_pass))

    while True:
      try:
        (review, summary,keywords) = input_gen.__next__() # read the next example from file. review and summary are both strings.
        # (review, summary,orign_review_text) = input_gen.__next__() # read the next example from file. review and summary are both strings.

      except StopIteration: # if there are no more examples:
        tf.logging.info("The example generator for this example queue filling thread has exhausted bert_data.")
        if self._single_pass:
          tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of bert_data; error.")

      # summary_sentences = [b'sony be fm rec dh']
      # summary_sentences = [sent.strip() for sent in bert_data.summary2sents(summary)] # Use the <s> and </s> tags in summary to get a list of sentences.
      summary_sentences = [summary.strip()]
      example = Example(review, summary_sentences, keywords, self._vocab)  # Process into an Example.
      self._example_queue.put(example) # place the Example in the example queue.

  # @torchsnooper.snoop()
  def fill_batch_queue(self):
    while True:
      if self.mode == 'decode':
        # beam search decode mode single example repeated in the batch
        ex = self._example_queue.get()
        b = [ex for _ in range(self.batch_size)]
        self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
      else:
        # Get bucketing_cache_size-many batches of Examples into a list, then sort
        inputs = []
        for _ in range(self.batch_size * self._bucketing_cache_size):
          inputs.append(self._example_queue.get())
        inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True) # sort by length of encoder sequence

        # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
        batches = []
        for i in range(0, len(inputs), self.batch_size):
          batches.append(inputs[i:i + self.batch_size])
        if not self._single_pass:
          shuffle(batches)
        for b in batches:  # each b is a list of Example objects
          self._batch_queue.put(Batch(b, self._vocab, self.batch_size))

  # @torchsnooper.snoop()
  def watch_threads(self):
    while True:
        # 加快Trainig
#       tf.logging.info(
#         'Bucket queue size: %i, Input queue size: %i',
#         self._batch_queue.qsize(), self._example_queue.qsize())

      time.sleep(60) # 絕對不可以拿掉
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()

  # @torchsnooper.snoop()
  # call by Batcher.fill_example_queue (fill_example_queue)
  def text_generator(self, example_generator):
    while True:
      e = next(example_generator) # e is a tf.Example
      # e = example_generator.__next__()  # e is a tf.Example
      try:
        # review_text = b'purchase this on december and get prime shippi...care of ipod tv and cd etc be very happy with it'
        review_text = e.features.feature['review'].bytes_list.value[0]  # the review text was saved under the key 'review' in the bert_data files
        # summary_text = b' <s> solid value performer </s> '
        summary_text = e.features.feature['summary'].bytes_list.value[0]  # the summary text was saved under the key 'summary' in the bert_data files
#         tf.logging.error(e.features.feature['summary'].bytes_list.value)
#         tf.logging.error(e.features.feature[config.keywords].bytes_list.value[0])
#         tf.logging.error('---------------------------------------------------')
        try:
            # print(config.keywords)
            keywords_text = e.features.feature[config.keywords].bytes_list.value[0]  # the keywords text was saved under the key 'keywords' in the bert_data files
        except Exception as e:
            # print(e)
            continue
        review_text = review_text.decode()
        summary_text = summary_text.decode()
        keywords_text = keywords_text.decode()
        if len(review_text.split(" ")) > 512 : continue

      except ValueError:
        tf.logging.error('Failed to get review or summary from example')
        continue
      if len(review_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
        #tf.logging.warning('Found an example with empty review text. Skipping it.')
        continue
      else:
        # Return value:.. (b'purchase this on december and get prime shipp...py with it', b' <s> solid value performer </s> ')
        yield (review_text, summary_text, keywords_text)
        # yield (review_text, summary_text,keywords_text)
