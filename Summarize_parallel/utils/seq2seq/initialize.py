import os
import torch
import torch as T
from torch import nn
import random
import numpy as np
# import config

# 引入 word2vec
import gensim
from gensim.models import word2vec
# 引入 glove
from glove import Glove
from glove import Corpus
# 引入 bert
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertModel, BertTokenizer 
T.manual_seed(0)
from utils.seq2seq import data

seed = 114514   # 24岁, 是魔法带学生

def init_seeds():
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_lstm_weight(lstm):
    for param in lstm.parameters():
        if len(param.shape) >= 2: # weights
            init_ortho_weight(param.data)
        else: # bias
            init_bias(param.data)

def init_gru_weight(gru):
    for param in gru.parameters():
        if len(param.shape) >= 2: # weights
            init_ortho_weight(param.data)
        else: # bias
            init_bias(param.data)

def init_linear_weight(linear):
    init_xavier_weight(linear.weight)
    if linear.bias is not None:
        init_bias(linear.bias)

def init_normal_weight(w):
    nn.init.normal_(w, mean=0, std=0.01)

def init_uniform_weight(w):
    nn.init.uniform_(w, -0.1, 0.1)

def init_ortho_weight(w):
    nn.init.orthogonal_(w)

def init_xavier_weight(w):
    nn.init.xavier_normal_(w)

def init_bias(b):
    nn.init.constant_(b, 0.)

def init_wt_normal(wt):
    wt.data.normal_(std=1e-4)

def init_lstm_wt(lstm):
    rand_unif_init_mag = 0.02   
    for name, _ in lstm.named_parameters():
        if 'weight' in name:
            wt = getattr(lstm, name)
            wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)
        elif 'bias' in name:
            # set forget bias to 1
            bias = getattr(lstm, name)
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data.fill_(0.)
            bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    trunc_norm_init_std = 1e-4
    linear.weight.data.normal_(std=trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=trunc_norm_init_std)

def get_Word2Vec_weight(vocab, config): # problem
    path = os.path.abspath(config.word_emb_path).replace('Summarize_parallel/','/')
    print(path)
    w2vec = gensim.models.KeyedVectors.load_word2vec_format(
    path, binary=False, encoding='utf-8')
    weight = T.zeros(config.vocab_size, config.emb_dim)
    for i in range(4):
        weight[i, :] = T.from_numpy(np.random.normal(0, 1e-4, config.emb_dim))
    # print('pad token emb : ',weight[1][:5])
    for i in range(len(vocab._id2word.keys())):
        try:
            vocab_word = vocab._id2word[i+4]
            w2vec_word = w2vec.wv.index2entity[i]
            # vocab_word = w2vec_word
        except Exception as e :
            continue
        if i + 4 >= config.vocab_size: break
        weight[i+4, :] = T.from_numpy(w2vec.wv.vectors[i])

    return weight

# def get_glove_weight(vocab, config):
#     config.word_emb_path = config.Data_path_ +"Embedding/glove/glove.6B.300d.txt"
#     # print(config.word_emb_path)
#     weight = T.zeros(config.vocab_size, config.emb_dim)
#     for i in range(4):
#         weight[i, :] = T.from_numpy(np.random.normal(0, 1e-4, 300))

#     with open(config.word_emb_path, 'r',encoding='utf-8') as f :
#         for line in f.readlines():
#             values = line.split()
#             word = values[0]
#             if word not in vocab._word2id.keys(): continue
#             vector = np.asarray(values[1:], "float32")
#             wid = vocab.word2id(word)          
#             weight[wid, :] = T.from_numpy(vector)

#     return weight

def get_glove_weight(vocab, config):
    glove = Glove.load(config.Data_path + 'Embedding/glove/glove%s.model'%(config.emb_dim))
    weight = T.zeros(config.vocab_size, config.emb_dim)
    for i in range(4):
        weight[i, :] = T.from_numpy(np.random.normal(0, 1e-4, config.emb_dim))

    for word, wid in glove.dictionary.items():
        try:
            vector = np.asarray(glove.word_vectors[wid], "float32")
            if wid + 4 >= len(weight): break
            weight[wid + 4, :] = T.from_numpy(vector)
        except Exception:
            print("no indexes %s" % wid)
            continue
    return weight

def get_bert_embed_matrix():
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat
def get_bert_weight(vocab, config):
    embedding_matrix = get_bert_embed_matrix() # Bert word embedding weights
    # weight = T.zeros(config.vocab_size + 5, config.emb_dim)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,do_basic_tokenize=True)
    bert_vocab = tokenizer.vocab # word_to_id

    # for i in range(config.vocab_size + 5):
    #     weight[i, :] = T.from_numpy(np.random.normal(0, 1e-4, 768))

    init_emb = nn.Embedding(config.vocab_size, config.emb_dim)
    init_wt_normal(init_emb.weight)
    weight = init_emb.weight

    for bert_id,word in enumerate(bert_vocab):
        # if word not in vocab._word2id.keys(): continue
        try:
            vector = embedding_matrix[bert_id]
            wid = vocab.word2id(word)
            weight[wid + 4, :] = T.from_numpy(vector)
        except Exception:
            # print("no indexes %s" % wid)
            continue
    return weight

def get_init_embedding(config, vocab):
    if config.pre_train_emb:
        word_embed = nn.Embedding(config.vocab_size, config.emb_dim)
        # FastText & word2Vec same format
        if config.word_emb_type == 'word2Vec' or config.word_emb_type == 'FastText':    
            weight = get_Word2Vec_weight(vocab, config);# print('weight',len(weight))
        elif config.word_emb_type == 'glove':
            weight = get_glove_weight(vocab, config)                
        elif config.word_emb_type == 'bert':    
            weight = get_bert_weight(vocab, config)
            
        word_embed = T.nn.Embedding.from_pretrained(weight)
        word_embed.weight.requires_grad = config.emb_grad               
    else:
        word_embed = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(word_embed.weight)
        # requires_grad指定是否在训练过程中对词向量的权重进行微调
        word_embed.weight.requires_grad = True
    return word_embed

# def save_model(f, model, optimizer):
#     torch.save({"model_state_dict" : model.state_dict(),
#             "optimizer_state_dict" : optimizer.state_dict()},
#             f)
 
# def load_model(f, model, optimizer):
#     checkpoint = torch.load(f)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     return model, optimizer

def loadCheckpoint(logger, load_model_path, model, optimizer):    
    # checkpoint = T.load(load_model_path, map_location='cpu')
    T.backends.cudnn.benchmark = True 
    print(load_model_path)
    checkpoint = T.load(load_model_path)
    model.load_state_dict(checkpoint['model'])
    step = checkpoint['step']
    vocab = checkpoint['vocab']
    loss = checkpoint['loss']
    r_loss = checkpoint['r_loss']
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info("Loaded model at " + load_model_path)
    logger.info("Loaded model step = %s, loss = %.2f, r_loss = %.2f " %(step, loss, r_loss))
    return model, optimizer, step

def save_model(config, logger, model, optimizer, step, vocab, loss, r_loss=0, title = ''):
    file_path = "/%07d.tar" % (step)
    save_path = config.save_model_path + '/%s' % (title)
    if not os.path.exists(save_path): os.makedirs(save_path)
    save_path = save_path + file_path
    state = {
        # 'model': model.state_dict(),
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'vocab': vocab,
        'loss':loss,
        'r_loss':r_loss
    }
    logger.info('Saving model step %d to %s...'%(step, save_path))
    T.save(state, save_path)