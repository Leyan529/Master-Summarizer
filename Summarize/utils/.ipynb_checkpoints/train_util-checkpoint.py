import numpy as np
import torch as T
from utils import config
import logging
import os
from datetime import datetime as dt

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
        return tensor  
    
def re_config(opt):
    config.key_attention = opt.key_attention
    config.intra_encoder = opt.intra_encoder
    config.intra_decoder = opt.intra_decoder

    config.lr = opt.lr
    config.max_dec_steps = opt.max_dec_steps
    config.max_enc_steps = opt.max_enc_steps
    config.min_dec_steps = opt.min_dec_steps
    config.pre_train_emb = opt.pre_train_emb

    config.max_epochs = opt.max_epochs
    config.rand_unif_init_mag = opt.rand_unif_init_mag
    config.trunc_norm_init_std = opt.trunc_norm_init_std
    config.vocab_size = opt.vocab_size
    config.word_emb_type = opt.word_emb_type
    config.mle_weight = opt.mle_weight
    config.rl_weight = 1 - opt.mle_weight
    config.load_ckpt = opt.load_ckpt
    
    config.word_emb_path = config.Data_path + "Embedding/%s/%s.300d.txt"%(config.word_emb_type,config.word_emb_type)
    config.vocab_path = config.Data_path + 'Embedding/%s/word.vocab'%(config.word_emb_type)
    
    config.gound_truth_prob = opt.gound_truth_prob
    config.hidden_dim = opt.hidden_dim
    config.emb_dim = opt.emb_dim
    config.keywords = opt.keywords
    config.gradient_accum = opt.gradient_accum
    config.beam_size = opt.beam_size
    return config

def loadCheckpoint(logger, load_model_path, model, optimizer):    
    checkpoint = T.load(load_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    step = checkpoint['step']
    vocab = checkpoint['vocab']
    loss = checkpoint['loss']
    r_loss = checkpoint['r_loss']
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info("Loaded model at " + load_model_path)
    logger.info("Loaded model step = %s, loss = %.2f, r_loss = %.2f " %(step, loss, r_loss))
    return model, optimizer, step
    
def getLogger(loggerName):
    # 取得日期
    today = dt.now()
    loggerPath = "LOG/%s-(%s_%s_%s)-(%s:%s:%s)"%(config.word_emb_type,
    today.year,today.month,today.day,today.hour,today.minute,today.second)
    
    # 設置logger
    logger = logging.getLogger(loggerName)  # 不加名稱設置root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.Filter(loggerName)

    # 使用FileHandler輸出到文件
    directory = os.path.dirname(loggerPath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fh = logging.FileHandler(loggerPath)

    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # 使用StreamHandler輸出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # 添加兩個Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
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
    enc_key = get_cuda(batch.key_inp)
    batch_size, seqlen = enc_batch.size()

    enc_lens = np.array(batch.enc_lens)
    key_lens = np.array(batch.key_lens)
    coverage_1 = None
    extra_zeros = None
    enc_batch_extend_vocab = None

#     if config['copy']:
    if True:
        enc_batch_extend_vocab = get_cuda(batch.art_batch_extend_vocab)
        # max_art_oovs is the max over all the article oov list in the batch
#         if batch.max_art_oovs > 0:
#             extra_zeros = get_cuda(T.zeros((batch_size, 1, batch.max_art_oovs),requires_grad=False))
    
    extra_zeros = get_cuda(T.zeros((batch_size, 1, 10),requires_grad=False))
    if True:
        coverage_1 = get_cuda(T.zeros((batch_size, seqlen), requires_grad=False))

    if not batch_first:
        enc_batch.transpose_(0, 1)
        enc_pad_mask.transpose_(0, 1)
        if extra_zeros is not None:
            extra_zeros.transpose_(0, 1)
    ct_e = T.zeros(batch_size, 2*config.hidden_dim) # context vector
    ct_e = get_cuda(ct_e)
    
    return enc_batch, enc_pad_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage_1, ct_e, enc_key, key_lens

# 我的 # return enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e    
# 多了 ct_e
# 我的 # return enc_batch, enc_lens, enc_padding_mask, enc_key_batch, enc_key_lens, enc_key_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e


def get_output_from_batch(batch, batch_first = False):
    """ returns: dec_batch, dec_pad_mask, max_dec_len, dec_lens_var, tgt_batch """
    dec_lens = np.array(batch.dec_lens)
    dec_lens_var = T.tensor(dec_lens).float()
    # 这个东东是用来规范化batch loss用的
    # 每一句的总loss除以它的词数
    max_dec_len = max(dec_lens)

    dec_batch = get_cuda(batch.dec_inp)
    dec_pad_mask = get_cuda(batch.dec_pad_mask)
    tgt_batch = get_cuda(batch.dec_tgt)
    dec_lens_var = get_cuda(dec_lens_var)

    if not batch_first:
        dec_batch.transpose_(0, 1)
        tgt_batch.transpose_(0, 1)
        dec_pad_mask.transpose_(0, 1)

    return dec_batch, dec_pad_mask, dec_lens, max_dec_len, dec_lens_var, tgt_batch

def save_model(logger, model, optimizer, step, vocab, loss, r_loss=0):
    file_path = "/%07d.tar" % (step)
    save_path = config.save_model_path + '/%s' % (config.word_emb_type)
    if not os.path.exists(save_path): os.makedirs(save_path)
    save_path = save_path + file_path
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'vocab': vocab,
        'loss':loss,
        'r_loss':r_loss
    }
    logger.info('Saving model step %d to %s...'%(step, save_path))
    T.save(state, save_path)