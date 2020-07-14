import numpy as np
import torch as T
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

    # config.rl_weight = 1 - opt.mle_weight
    
    config.word_emb_path = config.Data_path + "Embedding/%s/%s.%sd.txt"%(config.word_emb_type,config.word_emb_type,config.emb_dim)
    config.vocab_path = config.Data_path + 'Embedding/%s/word.vocab'%(config.word_emb_type)   
    
    setattr(config, 'rl_weight', 1-opt.mle_weight)
    setattr(config, 'word_emb_path', config.Data_path + "Embedding/%s/%s.%sd.txt"%(opt.word_emb_type,opt.word_emb_type,opt.emb_dim))
    setattr(config, 'vocab_path',config.Data_path + 'Embedding/%s/word.vocab'%(opt.word_emb_type))
    return config

def getName(config):
    if config.model_type == 'seq2seq':
        loggerName = 'Pointer_generator_%s' % (config.word_emb_type)

        if (config.intra_encoder and config.intra_decoder) and True :
            loggerName = loggerName + '_Intra_Atten'
        if config.key_attention:
            key_name = config.keywords.replace('_keywords','')
            key_name = key_name.replace('_FOP','')
            loggerName = loggerName + '_Key_Atten(%s)'%(key_name)

        if not config.pre_train_emb:
            loggerName = loggerName.replace(config.word_emb_type,'no_pretrain')


    model_name = ''
    if config.model_type == 'seq2seq':
        model_name = model_name + 'Pointer-Generator'
        if (config.intra_encoder and config.intra_decoder) and True :
            model_name = model_name + '_Intra_Atten'
        if config.key_attention:
            key_name = config.keywords.replace('_keywords','')
            key_name = key_name.replace('_FOP','')
            model_name = model_name + '_Key_Atten(%s)'%(key_name)
    else:
        model_name = model_name + 'Transformer'

    if config.train_rl:
        loggerName = loggerName + '_RL'
        model_name = model_name + '_RL'

    if not config.pre_train_emb:
        writerPath = 'runs/%s/%s/%s/exp'% (config.data_type, model_name,'NoPretrain')
    else:
        writerPath = 'runs/%s/%s/%s/exp'% (config.data_type, model_name,config.word_emb_type)

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
    # enc_batch = get_cuda(batch.enc_inp)
    # enc_pad_mask = get_cuda(batch.enc_pad_mask)
    # enc_key_mask = get_cuda(batch.key_pad_mask)

    enc_batch = batch.enc_inp.cuda()
    enc_pad_mask = batch.enc_pad_mask.cuda()
    enc_key_mask = batch.key_pad_mask.cuda()

    if config.model_type == 'seq2seq':
        # 舊制格式
        # 1為保留  0則清除,按照慣例後面為0        
        enc_pad_mask = enc_pad_mask.float()
        enc_key_mask = enc_key_mask.float()
        # print('poiner-generaator mode')

    # enc_key = get_cuda(batch.key_inp)
    enc_key = batch.key_inp.cuda()
    batch_size, seqlen = enc_batch.size()

    enc_lens = T.tensor(np.array(batch.enc_lens))
    key_lens = T.tensor(np.array(batch.key_lens))
    extra_zeros = None
    enc_batch_extend_vocab = None
    coverage_1 = None

    # enc_batch_extend_vocab = get_cuda(batch.art_batch_extend_vocab)
    enc_batch_extend_vocab = batch.art_batch_extend_vocab.cuda()
    # max_art_oovs is the max over all the article oov list in the batch
    if batch.max_art_oovs > 0:
        if config.model_type == 'seq2seq':
            # extra_zeros = get_cuda(T.zeros((batch_size,batch.max_art_oovs)))
            extra_zeros = T.zeros((batch_size,batch.max_art_oovs)).cuda()
        else:
            # extra_zeros = get_cuda(T.zeros((batch_size, 1, batch.max_art_oovs), requires_grad=False))
            extra_zeros = T.zeros((batch_size, 1, batch.max_art_oovs), requires_grad=False).cuda()

    # extra_zeros = get_cuda(T.zeros((batch_size, 1, 10),requires_grad=False))
    if config.coverage:
        # coverage_1 = get_cuda(T.zeros((batch_size, seqlen), requires_grad=False))
        coverage_1 = T.zeros((batch_size, seqlen), requires_grad=False).cuda()


    ct_e = T.zeros(batch_size, 2*config.hidden_dim).cuda() # context vector
    # ct_e = get_cuda(ct_e)

    # print('enc_batch_extend_vocab',enc_batch_extend_vocab.shape)
    # print('extra_zeros',extra_zeros.shape)
    # print('ct_e',ct_e.shape)
    # print('-------------------')
    # print('enc_batch',enc_batch[-1])
    # print('batch.max_rev_oovs',batch.max_art_oovs)
    # print('extra_zeros',extra_zeros)
    return enc_batch, enc_pad_mask, enc_lens, enc_batch_extend_vocab, extra_zeros,\
     coverage_1, ct_e, enc_key, enc_key_mask, key_lens

def get_output_from_batch(batch, config, batch_first = False):
    """ returns: dec_batch, dec_pad_mask, max_dec_len, dec_lens_var, tgt_batch """
    dec_lens = batch.dec_lens
    max_dec_len = np.max(np.array(batch.dec_lens))

    dec_lens = T.tensor(dec_lens).float()
    # 这个东东是用来规范化batch loss用的
    # 每一句的总loss除以它的词数


    # dec_batch = get_cuda(batch.dec_inp)
    # dec_pad_mask = get_cuda(batch.dec_pad_mask)
    # tgt_batch = get_cuda(batch.dec_tgt)
    # dec_lens = get_cuda(dec_lens)

    dec_batch = batch.dec_inp.cuda()
    dec_pad_mask = batch.dec_pad_mask.cuda()
    tgt_batch = batch.dec_tgt.cuda()
    dec_lens = dec_lens.cuda()

        
    if config.model_type == 'seq2seq':
        dec_pad_mask = dec_pad_mask.eq(0).float()      
        
    # print('max dec_batch',max([len(b) for b in dec_batch]))
    # print('max tgt_batch',max([len(b) for b in tgt_batch]))
    # print('dec_batch',dec_batch[-1])
    # print('tgt_batch',tgt_batch[-1])
    # print('len',len(tgt_batch[-1]))

    # print('dec_lens',dec_lens)
    # print('max_dec_len',max_dec_len)
    # return dec_batch, dec_pad_mask, dec_lens, max_dec_len, dec_lens_var, tgt_batch
    return dec_batch, dec_pad_mask, dec_lens, max_dec_len, tgt_batch

