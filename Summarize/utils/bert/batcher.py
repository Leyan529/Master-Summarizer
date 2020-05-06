# from utils.bert import config, data
from utils import config
from utils.bert.data import *

# import config, data
# from data import *

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch

# PAD': 0, 'EOQ': 3, 'SEP': 102, 'CLS': 101, 'UNK': 100}
PAD = 0
UNKNOWN_TOKEN = 100

def pad_sequence(data, padding_idx=0, length = 0):
    """
        Padder 
        输入：list状的 参差不齐的东东
        输出：list状的 整齐的矩阵
    """
    if length==0: length = max(len(entry) for entry in data)
    return [d + [padding_idx] * (length - len(d)) for d in data] 



class Example:
    def __init__(self, config, vocab, data):        
        # print(config.tokenizer)

        article = data['review']
        abstract = data['summary'].replace("<s>","").replace("</s>","")
#         keywords = data['POS_FOP_keywords']       
        keywords = data[config.keywords]

        src_txt = [sent + " ." if idx < len(article.split(" . "))-1 else sent for idx, sent in enumerate(article.split(" . "))]
        text = ' {} {} '.format(config.sep_token, config.cls_token).join(src_txt)

        src_subtokens = config.tokenizer.tokenize(text)

        src_subtokens = [config.cls_token] + src_subtokens[:config.max_enc_steps-2] + [config.sep_token]
        # src_subtoken_idxs = config.tokenizer.convert_tokens_to_ids(src_subtokens)
        self.enc_inp = config.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(self.enc_inp) if t == config.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_id = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_id += s * [0]
            else:
                segments_id += s * [1]
        cls_id = [i for i, t in enumerate(self.enc_inp) if t == config.cls_vid]
        self.enc_seg = segments_id
        self.enc_cls = cls_id

        tgt_txt = [sent + " ." if idx < len(abstract.split(" . "))-1 else sent for idx, sent in enumerate(abstract.split(" . "))]
        tgt_subtokens_str = '[unused0] ' + ' '.join(config.tokenizer.tokenize(' '.join(tgt_txt), use_bert_basic_tokenizer=config.use_bert_basic_tokenizer)) + ' [unused2]'
        tgt_subtoken = tgt_subtokens_str.split()[:config.max_dec_steps]

        tgt_subtoken_idxs = config.tokenizer.convert_tokens_to_ids(tgt_subtoken)
        abs_ids = tgt_subtoken_idxs
        if config.model_type == 'seq2seq':
            self.dec_inp, self.dec_tgt = abs_ids[:-1], abs_ids[1:]
        else:
            self.dec_inp, self.dec_tgt = abs_ids, abs_ids

        self.art_extend_vocab, self.art_oovs = article2ids(src_subtokens, config.vocab)
        abs_extend_vocab = abstract2ids(tgt_subtoken, vocab, self.art_oovs)

        if config.model_type == 'seq2seq' and config.copy:      
            # 改写目标输出 反映COPY OOV
            _, self.dec_tgt = self.get_dec_inp_tgt(config, abs_extend_vocab, config.max_dec_steps)

        self.original_article = article
        self.original_abstract = abstract

        # src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        # b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
        #                "src_sent_labels": sent_labels, "segs": segments_ids, 
        #                'clss': cls_ids, 'src_txt': src_txt, "tgt_txt": tgt_txt}
# -----------------------------------------------------------------------------------------------------        
        key_words = keywords.split()
        key_words = [word for word in key_words if word in vocab._word2id.keys()] # 過濾不在vocabulary 的 keyword
        if len(key_words) > config.max_key_num:
            key_words = key_words[:config.max_key_num]  # 限定key_words數量    
        self.enc_key_len = len(key_words)  # store the length after truncation but before padding
        self.key_inp = [vocab.word2id(w) for w in key_words]  # list of keyword ids; NO UNK token
        self.key_words = key_words
# -----------------------------------------------------------------------------------------------------        

    def get_dec_inp_tgt(self, config, sequence, max_len, start_id = 1, stop_id = 2):
        '''
        {'BOS(start_id)': 1, 'EOS(stop_id)': 2, 
        'PAD': 0, 'EOQ': 3, 'SEP': 102, 'CLS': 101, 'UNK': 100}
        '''
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

        enc_seg = [poi.enc_seg for poi in batch]
        enc_cls = [poi.enc_cls for poi in batch]
        #print(dec_inp)
        
        art_extend_vocab = [poi.art_extend_vocab for poi in batch]
        # ----------------------------------------------------------------
        self.enc_lens = [len(src) for src in enc_inp]
        self.dec_lens = [len(tgt) for tgt in dec_inp]
        self.art_oovs = [poi.art_oovs for poi in batch]
        self.key_lens = [len(keys) for keys in key_inp]


        self.dec_inp = torch.tensor(pad_sequence(dec_inp, PAD))
        self.dec_tgt = torch.tensor(pad_sequence(dec_tgt, PAD))
        self.enc_inp = torch.tensor(pad_sequence(enc_inp, PAD))
        self.enc_seg = torch.tensor(pad_sequence(enc_seg, PAD))
        self.key_inp = torch.tensor(pad_sequence(key_inp, PAD))
        self.enc_cls = torch.tensor(pad_sequence(enc_cls, -1))

        self.art_batch_extend_vocab = torch.tensor(pad_sequence(art_extend_vocab, PAD))
        self.max_art_oovs = max([len(oovs)for oovs in self.art_oovs])

        self.enc_pad_mask = ~self.enc_inp.eq(PAD) # mask_src
        self.dec_pad_mask = ~self.dec_inp.eq(PAD) # mask_tgt
        self.key_pad_mask = ~self.key_inp.eq(PAD) # mask_key
        self.enc_cls_mask = ~self.enc_cls.eq(-1) # mask_cls
        self.enc_cls[self.enc_cls == -1] = 0

        self.original_abstract = [poi.original_abstract for poi in batch]
        self.original_article = [poi.original_article for poi in batch]   
        self.key_words = [poi.key_words for poi in batch] 


        # if (is_test):
        #         src_str = [x[-2] for x in data]
        #         setattr(self, 'src_str', src_str)
        #         tgt_str = [x[-1] for x in data]
        #         setattr(self, 'tgt_str', tgt_str)  

        # outputs, scores = self.model(enc_inp, dec_tgt, enc_seg, enc_cls,  enc_pad_mask, dec_pad_mask, enc_cls_mask)


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
        return Example(self.config, self.vocab, self.Data.iloc[index])

def getDataLoader(logger, config):
    # 新的資料包裝方式
    vocab = Vocab()
    setattr(config, 'tokenizer', vocab.tokenizer)
    setattr(config, 'vocab', vocab)
    setattr(config, 'cls_token', vocab.cls_token)
    setattr(config, 'sep_token', vocab.sep_token)

    setattr(config, 'sep_vid', vocab.sep_vid)
    setattr(config, 'cls_vid', vocab.cls_vid)
    setattr(config, 'pad_vid', vocab.pad_vid)
    setattr(config, 'unk_vid', vocab.unk_vid)  
    setattr(config, 'use_bert_basic_tokenizer',True)
 
    # 由於 train_test_split 的random state故每次切割的內容皆相同
    total_df = pd.read_excel(config.xls_path)
    total_df = total_df.sort_values(by=['lemm_review_len','overlap'], ascending = False)
    train_df, val_df = train_test_split(total_df, test_size=0.1, 
                                        random_state=0, shuffle=True)

    logger.info('train : %s, test : %s'%(len(train_df), len(val_df)))
    train_df = train_df.sort_values(by=['lemm_review_len'])
    val_df = val_df.sort_values(by=['lemm_review_len'])
    train_data = ReadDataset(train_df, config, vocab)
    validate_data = ReadDataset(val_df, config, vocab)


    # # class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, \
    collate_fn=Collate(), drop_last=True)     

    validate_loader = DataLoader(validate_data, batch_size=config.batch_size, shuffle=False, \
    collate_fn=Collate(), drop_last=True)
    
    logger.info('train batches : %s, test batches : %s'%(len(iter(train_loader)), len(iter(validate_loader))))

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': vocab.tokenizer.vocab['[unused0]'], 'EOS': vocab.tokenizer.vocab['[unused1]'],
                'PAD': vocab.tokenizer.vocab['[PAD]'], 'EOQ': vocab.tokenizer.vocab['[unused2]']}

    return train_loader, validate_loader, vocab, symbols   


 

