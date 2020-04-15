from utils import config, bert_data
from utils.bert_data import *
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

 
# START = bert_data.special_tokens.index(bert_data.START_DECODING)
# END = bert_data.special_tokens.index(bert_data.STOP_DECODING)
# PAD = bert_data.special_tokens.index(bert_data.PAD_TOKEN)
# UNKNOWN_TOKEN = bert_data.special_tokens.index(bert_data.UNKNOWN_TOKEN)

vocab = Vocab(config.vocab_size)
# print(list(vocab._word2id.items())[-2:])
# print(list(vocab._id2word.items())[-2:])

START = vocab.word2id(bert_data.START_DECODING) # start_decoding = 30524
END = vocab.word2id(bert_data.STOP_DECODING) # stop_decoding = 30525 
UNKNOWN_TOKEN = vocab.word2id(bert_data.UNKNOWN_TOKEN) # unk_decoding = 100
PAD = vocab.word2id(bert_data.PAD_TOKEN) # unk_decoding = 100

# print(bert_data.START_DECODING,START,vocab.id2word(START))
# print(bert_data.STOP_DECODING,END,vocab.id2word(END))
# print(bert_data.UNKNOWN_TOKEN,UNKNOWN_TOKEN,vocab.id2word(UNKNOWN_TOKEN))
# print(bert_data.PAD_TOKEN,PAD,vocab.id2word(PAD))




def pad_sequence(data, padding_idx=0, length = 0):
    """
        Padder 
        输入：list状的 参差不齐的东东
        输出：list状的 整齐的矩阵
    """
    if length==0: length = max(len(entry) for entry in data)
    return [d + [padding_idx] * (length - len(d)) for d in data]        

def pad_sequence2(data, padding_idx=0, length = 0):
    """
        Padder 
        输入：list状的 参差不齐的东东
        输出：list状的 整齐的矩阵
    """
    if length==0: length = 60
    return [d + [padding_idx] * (length - len(d)) for d in data]

class Example:
    def __init__(self, config, vocab, data):        

        article = data['review']
        abstract = data['summary'].replace("<s>","").replace("</s>","")
#         keywords = data['POS_FOP_keywords']       
        keywords = data[config.keywords]

        # src_words = article.split()[:config.max_enc_steps]
        src_words = vocab.tokenizer.tokenize(article)[:config.max_enc_steps]
        # print('max_enc_steps',config.max_enc_steps)
        # self.enc_inp = [vocab.word2id(w) for w in src_words]        
        self.enc_inp = vocab.tokenizer.encode(article,add_special_tokens=False)[:config.max_enc_steps]
        # print(vocab.tokenizer.convert_ids_to_tokens(self.enc_inp))

        abstract_words = [w for w in abstract.split() if w != ""]
        abs_ids = [vocab.word2id(w) for w in abstract_words]
        self.dec_inp, self.dec_tgt = self.get_dec_inp_tgt(abs_ids, config.max_dec_steps)

        self.art_extend_vocab, self.art_oovs = article2ids(src_words, vocab)
        # print('self.enc_inp',len(self.enc_inp))
        # print('self.art_extend_vocab',len(self.art_extend_vocab))
        abs_extend_vocab = abstract2ids(abstract_words, vocab, self.art_oovs)

        if True:      
            # 改写目标输出 反映COPY OOV
            _, self.dec_tgt = self.get_dec_inp_tgt(abs_extend_vocab, config.max_dec_steps)

        self.original_article = article
        self.original_abstract = abstract
# -----------------------------------------------------------------------------------------------------        
        key_words = keywords.split()
        key_words = [word for word in key_words if word in vocab._word2id.keys()] # 過濾不在vocabulary 的 keyword
        if len(key_words) > config.max_key_num:
            key_words = key_words[:config.max_key_num]  # 限定key_words數量    
        self.enc_key_len = len(key_words)  # store the length after truncation but before padding
        self.key_inp = [vocab.word2id(w) for w in key_words]  # list of keyword ids; NO UNK token
        self.key_words = key_words
# -----------------------------------------------------------------------------------------------------        

    def get_dec_inp_tgt(self, sequence, max_len, start_id = START, stop_id = END):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            # 如果需要截断，就不保留End Token
            inp = inp[:max_len]
            target = target[:max_len]
        else: # no truncation
            target.append(stop_id) # end token
        assert len(inp) == len(target)  
        return inp, target

class Batch:
    def __init__(self, batch):
        # 过滤掉他们（我有特别的filter技巧~）
        batch = list(filter(lambda poi: len(poi.enc_inp)>0, batch))
        batch = sorted(batch, key=lambda poi: len(poi.enc_inp), reverse=True) # sort by length of encoder sequence

        dec_inp = [poi.dec_inp for poi in batch]
        dec_tgt = [poi.dec_tgt for poi in batch]
        enc_inp = [poi.enc_inp for poi in batch]
        key_inp = [poi.key_inp for poi in batch]
        #print(dec_inp)
        
        art_extend_vocab = [poi.art_extend_vocab for poi in batch]
        # ----------------------------------------------------------------
        self.enc_lens = [len(src) for src in enc_inp]
        self.dec_lens = [len(tgt) for tgt in dec_inp]
        self.art_oovs = [poi.art_oovs for poi in batch]
        self.key_lens = [len(keys) for keys in key_inp]

        a = pad_sequence2(dec_inp, PAD)
        b = pad_sequence2(dec_tgt, PAD)
        self.dec_inp = torch.tensor(a)
        self.dec_tgt = torch.tensor(b)
        # print('enc_inp',[len(i) for i in enc_inp])
        # print('art_extend_vocab',[len(i) for i in art_extend_vocab])
        self.enc_inp = torch.tensor(pad_sequence(enc_inp, PAD))
        self.key_inp = torch.tensor(pad_sequence(key_inp, PAD))

        c = pad_sequence(art_extend_vocab, PAD)
        self.art_batch_extend_vocab = torch.tensor(c)
        self.max_art_oovs = max([len(oovs)for oovs in self.art_oovs])

        self.enc_pad_mask = self.enc_inp.eq(PAD)
        self.dec_pad_mask = self.dec_inp.eq(PAD)
        self.key_pad_mask = self.dec_inp.eq(PAD)

        self.original_abstract = [poi.original_abstract for poi in batch]
        self.original_article = [poi.original_article for poi in batch]   
        self.key_words = [poi.key_words for poi in batch]   
        # print('--------------------------------------------------------------------------')

class Collate():
    def __init__(self, beam_size = 1):
        self.beam = beam_size

    def _collate(self, batch):
        return Batch(batch * self.beam)

    def __call__(self, batch):
        return self._collate(batch)    

class ReadDataset(Dataset):
    def __init__(self, df, config, vocab):
        self._n_data = len(df)
        self.Data = df
        self.config = config
        self.vocab = vocab

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, index):
#         print(self.Data.iloc[index])
        return Example(self.config, self.vocab, self.Data.iloc[index])

def getDataLoader(logger, config):
    # 新的資料包裝方式
    vocab = Vocab(config.vocab_size)

    train_df, val_df = train_test_split(pd.read_excel(config.xls_path),test_size=0.1, 
                                        random_state=0, shuffle=True)
    logger.info('train : %s, test : %s'%(len(train_df), len(val_df)))
    train_df = train_df.sort_values(by=['lemm_review_len'])
    val_df = val_df.sort_values(by=['lemm_review_len'])
    train_data = ReadDataset(train_df, config, vocab)
    validate_data = ReadDataset(val_df, config, vocab)


    # class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=Collate())      
    validate_loader = DataLoader(validate_data, batch_size=config.batch_size, shuffle=False, collate_fn=Collate())
    return train_loader, validate_loader, vocab        