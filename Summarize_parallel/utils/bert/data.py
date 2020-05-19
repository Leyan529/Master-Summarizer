from utils.bert.tokenization import BertTokenizer

# from tokenization import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir='../temp')

# symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
#             'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
# symbols {'BOS': 1, 'EOS': 2, 'PAD': 0, 'EOQ': 3}

UNKNOWN_TOKEN = '[UNK]'
PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
SEP_TOKEN = '[SEP]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[unused0]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[unused1]' # This has a vocab id, which is used at the end of untruncated target sequences

CLS_TOKEN = '[CLS]'

class BertData():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir='../temp')

    def gettokenizer(self):
        return self.tokenizer


class Vocab:
    def __init__(self):
        self._word2id, self._id2word, self.tokenizer = make_vocab()

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.unk_token = '[UNK]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token] 
        self.unk_vid = self.tokenizer.vocab[self.unk_token] 

        symbols = {'BOS': self._word2id['[unused0]'], 'EOS': self._word2id['[unused1]'],
            'PAD': self._word2id['[PAD]'], 'EOQ': self._word2id['[unused2]'],
            'SEP': self._word2id['[SEP]'], 'CLS': self._word2id['[CLS]'],
            'UNK': self._word2id['[UNK]']}
        print(symbols)
        # {'BOS': 1, 'EOS': 2, 'PAD': 0, 'EOQ': 3, 'SEP': 102, 'CLS': 101, 'UNK': 100}

    def word2id(self, word):
        return self._word2id[UNKNOWN_TOKEN] if word not in self._word2id else self._word2id[word]

    def id2word(self, idx):
        return UNKNOWN_TOKEN if idx >= self.size else self._id2word[idx]

    @property
    def size(self):
        return len(self._word2id)

def make_vocab():
    # Read the vocab file and add words up to max_size
    word2id, id2word = {}, {}
    bert = BertData()
    for i,w in enumerate(bert.tokenizer.vocab):        
        word2id[w], id2word[i] = i, w
    return word2id, id2word, bert.gettokenizer()

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


