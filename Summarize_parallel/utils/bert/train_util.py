import numpy as np
import torch as T
# from utils.bert import config
from utils import config
import logging
import os
from datetime import datetime as dt
# from utils.seq2seq.batcher import PAD

# import logging
# logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("pytorch_pretrained_bert").setLevel(logging.ERROR)



def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
        return tensor  

def re_config(opt):
    for k,v in vars(opt).items():
        setattr(config, k, v)

    config.rl_weight = 1 - opt.mle_weight
    if not config.use_bert_emb:
        config.word_emb_path = config.Data_path + "Embedding/%s/%s.%sd.txt"%(config.word_emb_type,config.word_emb_type,config.emb_dim)
        config.vocab_path = config.Data_path + 'Embedding/%s/word.vocab'%(config.word_emb_type)  
        if hasattr(opt,'word_emb_type'):
            setattr(config, 'word_emb_path', config.Data_path + "Embedding/%s/%s.%sd.txt"%(opt.word_emb_type, opt.word_emb_type, opt.emb_dim))
            setattr(config, 'vocab_path',config.Data_path + 'Embedding/%s/word.vocab'%(opt.word_emb_type))
    
    setattr(config, 'rl_weight',opt.mle_weight)
    
    return config

def getName(config):
    if config.model_type == 'seq2seq':
        loggerName = 'Pointer_generator_%s' % (config.word_emb_type)

        if (config.intra_encoder and config.intra_decoder) and True :
            loggerName = loggerName + '_Intra_Atten'
        if config.key_attention:
            loggerName = loggerName + '_Key_Atten'

    else:
        loggerName = 'Transformer_%s' % (config.word_emb_type)
        if config.encoder == 'bert':
            loggerName = 'BertEnc_' + loggerName
        if config.sep_optim:
            loggerName = 'Sep_' + loggerName
        if config.copy:
            loggerName = 'Pointer_' + loggerName

    model_name = ''
    if config.model_type == 'seq2seq':
        model_name = model_name + 'Pointer-Generator'
        if (config.intra_encoder and config.intra_decoder) and True :
            model_name = model_name + '_Intra_Atten'
        if config.key_attention:
            model_name = model_name + '_Key_Atten'
    else:
        model_name = model_name + 'Transformer'
        if config.encoder == 'bert':
            model_name = 'BertEnc_' + model_name
        if config.sep_optim:
            model_name = 'Sep_' + model_name
        if config.copy:
            model_name = 'Pointer_' + model_name
    if config.train_rl:
        loggerName = loggerName + '_RL'
        model_name = model_name + '_RL'

    if config.encoder == 'bert' and config.use_bert_emb :
        loggerName = loggerName.replace(config.word_emb_type,'BertEmb')
        writerPath = 'runs/%s/%s/%s/exp'% (config.data_type, model_name,'BertEmb')
    else:
        loggerName = loggerName.replace(config.word_emb_type,'NoPretrain')
        writerPath = 'runs/%s/%s/%s/exp'% (config.data_type, model_name,'NoPretrain')

    print('loggerName',loggerName)
    print('writerPath',writerPath)
    return  loggerName, writerPath     

def getLogger(loggerName):
    # 取得日期
    # today = dt.now()
    # loggerPath = "LOG/%s-(%s_%s_%s)-(%s:%s:%s)"%(config.word_emb_type,
    # today.year,today.month,today.day,today.hour,today.minute,today.second)
    
    # 設置logger
    logger = logging.getLogger(loggerName)  # 不加名稱設置root logger
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.Filter(loggerName)

    # # 使用FileHandler輸出到文件
    # directory = os.path.dirname(loggerPath)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # fh = logging.FileHandler(loggerPath)

    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(formatter)

    # 使用StreamHandler輸出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # 添加兩個Handler
    logger.addHandler(ch)
    # logger.addHandler(fh)
    # Handler只啟動一次
    # 設置logger
    logger.info(u'logger已啟動')
    return logger

def removeLogger(logger):
    logger.info(u'logger已關閉')
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

def get_input_from_batch(batch, config, batch_first = False):
    """
        returns: enc_batch, enc_pad_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage
        如果config没有启用pointer 和cov 则相应的项返回None
    """
    enc_batch = get_cuda(batch.enc_inp)
    enc_pad_mask = get_cuda(batch.enc_pad_mask)
    # enc_key_mask = get_cuda(batch.key_pad_mask)

    if config.model_type == 'seq2seq':
        # 舊制格式
        # 1為保留  0則清除,按照慣例後面為0
        enc_pad_mask = enc_pad_mask.float()
        # enc_key_mask = enc_key_mask.float()
        # print('poiner-generaator mode')


    # enc_key = get_cuda(batch.key_inp)
    batch_size, seqlen = enc_batch.size()

    enc_lens = np.array(batch.enc_lens)
    # key_lens = np.array(batch.key_lens)
    coverage_1 = None
    extra_zeros = None
    enc_batch_extend_vocab = None
    coverage_1 = None

    enc_batch_extend_vocab = get_cuda(batch.art_batch_extend_vocab)
    # max_art_oovs is the max over all the article oov list in the batch
    if batch.max_art_oovs > 0:
        if config.model_type == 'seq2seq':
            extra_zeros = get_cuda(T.zeros((batch_size,batch.max_art_oovs)))
        else:
            extra_zeros = get_cuda(T.zeros((batch_size, 1, batch.max_art_oovs), requires_grad=False))
    
    # if config.coverage:
    #     coverage_1 = get_cuda(T.zeros((batch_size, seqlen), requires_grad=False))

    enc_seg = get_cuda(batch.enc_seg)
    enc_cls = get_cuda(batch.enc_cls)
    enc_cls_mask = get_cuda(batch.enc_cls_mask)

    return enc_batch, enc_pad_mask, enc_lens, enc_batch_extend_vocab, extra_zeros,\
     None, None, None, None, None, enc_seg, enc_cls, enc_cls_mask


def get_output_from_batch(batch, config, batch_first = False):
    """ returns: dec_batch, dec_pad_mask, max_dec_len, dec_lens_var, tgt_batch """
    dec_lens = batch.dec_lens
    max_dec_len = np.max(np.array(batch.dec_lens))

    dec_lens = T.tensor(dec_lens).float()

    dec_batch = get_cuda(batch.dec_inp)
    dec_pad_mask = get_cuda(batch.dec_pad_mask)
    tgt_batch = get_cuda(batch.dec_tgt)
    dec_lens = get_cuda(dec_lens)   

    return dec_batch, dec_pad_mask, dec_lens, max_dec_len, tgt_batch

        