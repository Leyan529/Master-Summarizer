from utils import config, data
from utils.data import *
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

 
START = data.special_tokens.index(data.START_DECODING)
END = data.special_tokens.index(data.STOP_DECODING)
PAD = data.special_tokens.index(data.PAD_TOKEN)
UNKNOWN_TOKEN = data.special_tokens.index(data.UNKNOWN_TOKEN)


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
    if length==0: length = 50
    return [d + [padding_idx] * (length - len(d)) for d in data]  

def pad_sequence3(data, padding_idx=0, length = 0):
    """
        Padder 
        输入：list状的 参差不齐的东东
        输出：list状的 整齐的矩阵
    """
    if length==0: length = config.max_key_num
    return [d + [padding_idx] * (length - len(d)) for d in data][:config.max_key_num]     
    
class Example:
    def __init__(self, config, vocab, data):        

        article = data['review'].strip()
        abstract = data['summary'].strip().replace("<s>","").replace("</s>","")
#         keywords = data['POS_FOP_keywords']       
        keywords = data[config.keywords]

        src_words = article.split()[:config.max_enc_steps]
        self.enc_inp = [vocab.word2id(w) for w in src_words]

        abstract_words = [w for w in abstract.split() if w != ""]
        abs_ids = [vocab.word2id(w) for w in abstract_words]
        self.dec_inp, self.dec_tgt = self.get_dec_inp_tgt(config, abs_ids, config.max_dec_steps)

        self.art_extend_vocab, self.art_oovs = article2ids(src_words, vocab)
        abs_extend_vocab = abstract2ids(abstract_words, vocab, self.art_oovs)

        if config.copy:      
            # 改写目标输出 反映COPY OOV
            _, self.dec_tgt = self.get_dec_inp_tgt(config, abs_extend_vocab, config.max_dec_steps)

        self.original_article = article.strip()
        self.original_abstract = abstract.strip()
# -----------------------------------------------------------------------------------------------------        
        key_words = keywords.split()
        key_words = [word for word in key_words if word in vocab._word2id.keys()] # 過濾不在vocabulary 的 keyword
        if len(key_words) > config.max_key_num:
            key_words = key_words[:config.max_key_num]  # 限定key_words數量    
        self.enc_key_len = len(key_words)  # store the length after truncation but before padding
        self.key_inp = [vocab.word2id(w) for w in key_words]  # list of keyword ids; NO UNK token
        self.key_words = key_words
# -----------------------------------------------------------------------------------------------------        

    def get_dec_inp_tgt(self, config, sequence, max_len, start_id = START, stop_id = END):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            # 如果需要截断，就不保留End Token
            inp = inp[:max_len]
            target = target[:max_len]
        else: # no truncation
            target.append(stop_id) # end token
        # if not config.transformer:
        #     inp = inp + [0] * (max_len - len(inp))
        #     target = target + [0] * (max_len - len(target))
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

        # original (pointer-generator)
        # self.dec_inp = torch.tensor(pad_sequence2(dec_inp, PAD))
        # self.dec_tgt = torch.tensor(pad_sequence2(dec_tgt, PAD))
        # self.enc_inp = torch.tensor(pad_sequence(enc_inp, PAD))
        # self.key_inp = torch.tensor(pad_sequence3(key_inp, PAD))

        # new (transformer)
        self.dec_inp = torch.tensor(pad_sequence(dec_inp, PAD))
        self.dec_tgt = torch.tensor(pad_sequence(dec_tgt, PAD))
        self.enc_inp = torch.tensor(pad_sequence(enc_inp, PAD))
        self.key_inp = torch.tensor(pad_sequence(key_inp, PAD))

        self.art_batch_extend_vocab = torch.tensor(pad_sequence(art_extend_vocab, PAD))
        self.max_art_oovs = max([len(oovs)for oovs in self.art_oovs])

        # print('max dec_inp',max([len(b) for b in self.dec_inp]))
        # print('max dec_tgt',max([len(b) for b in self.dec_tgt]))
        # true為mask掉 false則沒有,按照慣例後面為true        
        
        # 新架構PreSum版本
        self.enc_pad_mask = ~self.enc_inp.eq(PAD)
        self.dec_pad_mask = ~self.dec_inp.eq(PAD)
        self.key_pad_mask = ~self.key_inp.eq(PAD)
        # print(self.enc_pad_mask[0][-10:-1])
        # print(self.dec_pad_mask[0][:])
        # print(self.key_pad_mask[0][:])


        self.original_abstract = [poi.original_abstract for poi in batch]
        self.original_article = [poi.original_article for poi in batch]   
        self.key_words = [poi.key_words for poi in batch]   


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
    vocab = Vocab(config.vocab_path, config.vocab_size)

    total_df = pd.read_excel(config.xls_path)
    # total_df = total_df.sort_values(by=['lemm_review_len','overlap'], ascending = False)
    train_df, val_df = train_test_split(total_df, test_size=0.1, 
                                        random_state=0, shuffle=True)
    logger.info('train : %s, test : %s'%(len(train_df), len(val_df)))
    train_df = train_df.sort_values(by=['lemm_review_len'])
    val_df = val_df.sort_values(by=['lemm_review_len'])
    train_data = ReadDataset(train_df, config, vocab)
    validate_data = ReadDataset(val_df, config, vocab)


    # class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=Collate(), drop_last=True)      
    validate_loader = DataLoader(validate_data, batch_size=config.batch_size, shuffle=False, collate_fn=Collate(), drop_last=True)
    logger.info('train batches : %s, test batches : %s'%(len(iter(train_loader)), len(iter(validate_loader))))
    return train_loader, validate_loader, vocab