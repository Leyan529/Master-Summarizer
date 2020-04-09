import torch as T
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_util import config
import torch.nn.functional as F
from train_util import get_cuda
import torchsnooper
import numpy as np
import sys

import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# 引入 word2vec
import gensim
from gensim.models import word2vec
# 引入 glove
from glove import Glove
from glove import Corpus
# 引入 bert
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
T.manual_seed(0)

def get_Word2Vec_weight(vocab): # problem
    w2vec = gensim.models.KeyedVectors.load_word2vec_format(
    config.word_emb_path, binary=False, encoding='utf-8')
    weight = T.zeros(config.vocab_size, config.emb_dim)
    for i in range(4):
        weight[i, :] = T.from_numpy(np.random.normal(0, 1e-4, 300))
    # print('pad token emb : ',weight[1][:5])
    for i in range(len(vocab._id_to_word.keys())):
        try:
            vocab_word = vocab._id_to_word[i+4]
            w2vec_word = w2vec.wv.index2entity[i]
            # vocab_word = w2vec_word
        except Exception as e :
            continue
        if i + 4 > config.vocab_size: break
        weight[i+4, :] = T.from_numpy(w2vec.wv.vectors[i])

    return weight


def get_glove_weight(vocab):
    config.word_emb_path = config.Data_path_ +"Embedding/glove/glove.6B.300d.txt"

    weight = T.zeros(config.vocab_size, config.emb_dim)
    for i in range(4):
        weight[i, :] = T.from_numpy(np.random.normal(0, 1e-4, 300))

    with open(config.word_emb_path, 'r',encoding='utf-8') as f :
        for line in f.readlines():
            values = line.split()
            word = values[0]
            if word not in vocab._word_to_id.keys(): continue
            vector = np.asarray(values[1:], "float32")
            wid = vocab.word2id(word)          
            weight[wid, :] = T.from_numpy(vector)

    return weight

def get_glove_weight2(vocab):
    glove = Glove.load(config.Data_path + 'Embedding/glove/glove.model')
    weight = T.zeros(config.vocab_size, config.emb_dim)
    for i in range(4):
        weight[i, :] = T.from_numpy(np.random.normal(0, 1e-4, 300))

    for word,idx in glove.dictionary.items():
        try:
            wid = vocab.word2id(word) 
            vector = np.asarray(glove.word_vectors[glove.dictionary[word]], "float32")
            weight[wid + 4, :] = T.from_numpy(vector)
        except KeyError:
            print("no indexes %s" % wid)
            continue
    return weight

def get_bert_embed_matrix():
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat


def get_bert_weight(vocab):
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
        # if word not in vocab._word_to_id.keys(): continue
        try:
            vector = embedding_matrix[bert_id]
            wid = vocab.word2id(word)
            weight[wid + 4, :] = T.from_numpy(vector)
        except Exception:
            # print("no indexes %s" % wid)
            continue
    return weight



def init_lstm_wt(lstm):
    for name, _ in lstm.named_parameters():
        if 'weight' in name:
            wt = getattr(lstm, name)
            wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
        elif 'bias' in name:
            # set forget bias to 1
            bias = getattr(lstm, name)
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data.fill_(0.)
            bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

# def init_smbol_normal(vectors):
#     vectors[4:, :] = vectors[:-4, :]  # 預留4位空格給特殊符號
#     for i in range(4):
#         vectors[i, :] = np.random.normal(0, 1e-4, 300)
#     return vectors

# @torchsnooper.snoop()
# def from_pretrained(embeddings):
#     assert embeddings.dim() == 2, \
#          'Embeddings parameter is expected to be 2-dimensional'
#     # rows, cols = embeddings.shape
#     embedding = T.nn.Embedding(num_embeddings=config.vocab_size+1, embedding_dim=config.emb_dim)
#     embedding.weight = T.nn.Parameter(embeddings)
# 	# requires_grad指定是否在训练过程中对词向量的权重进行微调
#     embedding.weight.requires_grad = False
#     return embedding

# def from_pretrained(embeddings, freeze=False):
#     assert embeddings.dim() == 2, \
#          'Embeddings parameter is expected to be 2-dimensional'
#     rows, cols = embeddings.shape
#     embedding = T.nn.Embedding(num_embeddings=config.vocab_size+1, embedding_dim = config.emb_dim)
#     embedding.weight = T.nn.Parameter(embeddings)
#     embedding.weight.requires_grad = not freeze
#     return embedding

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm) # 初始话LSTM的隐层和细胞状态

        # 同样考虑向前层和向后层
        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim) # 隐层
        init_linear_wt(self.reduce_h) 
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim) # 细胞状态
        init_linear_wt(self.reduce_c)

#     @torchsnooper.snoop()
    def forward(self, x, seq_lens):
#         Starting var:.. x = tensor<(1, 27, 256), float32, cuda:0, grad>
#         Starting var:.. seq_lens = ndarray<(1,), int32>
#         x.shape torch.Size([4, 55, 256])
#         seq_lens.shape (4,)
        # print(x.shape, seq_lens)
        packed = pack_padded_sequence(x, seq_lens, batch_first=True)
#         packed type <class 'torch.nn.utils.rnn.PackedSequence'>
        enc_out, enc_hid = self.lstm(packed)
        # print('enc_out',enc_out)
        # print('h_hid',enc_hid[0].shape)
        # print('c_hid',enc_hid[1].shape)
        enc_out,_ = pad_packed_sequence(enc_out, batch_first=True)  # x:[batch_size,max_enc_steps,emb_dim] y:[seq_lens]
        enc_out = enc_out.contiguous()                              #batch_size, b_seq_len, 2*hid_size
        # print('enc_out',enc_out.shape)
        # print('enc_hid',enc_hid.shape) # tuple
        h, c = enc_hid                                              #shape of h: 2, batch_size, hid_size
        # print('list(h)',h[0].shape)
        # print('list(c)',c[0].shape)
        h = T.cat(list(h), dim=1)                                   #batch_size, 2*hid_size
        c = T.cat(list(c), dim=1)
        # print('h',h.shape)
        # print('c',c.shape)
        h_reduced = F.relu(self.reduce_h(h))                        #batch_size,hid_size
        c_reduced = F.relu(self.reduce_c(c))
        # print('enc_out',enc_out.shape)
        # print('h_reduced',h_reduced.shape)
        # print('c_reduced',c_reduced.shape)
        # enc_out :[batch_size, b_seq_len, hid_size*2=1024]
        # h_reduced:[batch_size, hid_size=512]
        # c_reduced:[batch_size, hid_size=512]
        return enc_out, (h_reduced, c_reduced)

class encoder_attention(nn.Module):

    def __init__(self):
        super(encoder_attention, self).__init__()
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.W_s = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.W_t = nn.Linear(config.emb_dim , config.hidden_dim*2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)


    def forward(self, st_hat, h, enc_padding_mask, sum_temporal_srcs, sum_k_emb):
        ''' Perform attention over encoder hidden states
        :param st_hat: decoder hidden state at current time step
        :param h: encoder hidden states
        :param enc_padding_mask:
        :param sum_temporal_srcs: if using intra-temporal attention, contains summation of attention weights from previous decoder time steps
        Self Attention也经常被称为intra Attention（内部Attention）
        Source内部元素之间或者Target内部元素之间发生的Attention机制，也可以理解为Target=Source这种特殊情况下的注意力计算机制。
        其具体计算过程是一样的，只是计算对象发生了变化而已
        '''

        # Standard attention technique (eq 1 in h Pointer-Generator Networks - https://arxiv.org/pdf/1704.04368.pdf)
        # et = tanh ( W_h(h) + W_s(st_hat) )
        # print(h.shape);print(config.hidden_dim * 2, config.hidden_dim * 2)
        # print('h',h.shape)
        et = self.W_h(h)                        # batch_size,n_seq,2*hid_size
        # print('et1',et.shape)
        dec_fea = self.W_s(st_hat).unsqueeze(1) # batch_size,1,2*hid_size
        # print('dec_fea',dec_fea.shape)
        # print('h',h.shape)
        # print('st_hat',st_hat.shape)
        et = et + dec_fea                       # et => incorporate h_td (hidden decoder state) & h_te (hidden encoder state)
        # print('et2',et.shape)
        if config.key_attention: 
            k_t = self.W_t(sum_k_emb).unsqueeze(1)
            if k_t.shape[0] == et.shape[0] : et = et + k_t
        et = T.tanh(et)                         # batch_size,b_seq_len,2*hid_size
        et = self.v(et).squeeze(2)              # batch_size,b_seq_len
        # print('et3',et.shape)
        # intra-temporal attention     (eq 3 in DEEP REINFORCED MODEL - https://arxiv.org/pdf/1705.04304.pdf)
        if config.intra_encoder:
            exp_et = T.exp(et)
            if sum_temporal_srcs is None:
                et1 = exp_et # eq 3 if t = 1 condition
                sum_temporal_srcs  = get_cuda(T.FloatTensor(et.size()).fill_(1e-10)) + exp_et
            else:
                et1 = exp_et/sum_temporal_srcs  # eq 3 otherwise condition   #batch_size, b_seq_len
                sum_temporal_srcs = sum_temporal_srcs + exp_et # 針對自己過去所有的 source attention score 加總 (self-attention)
        else:
            # (eq 2 in h Pointer-Generator Networks - https://arxiv.org/pdf/1704.04368.pdf)
            et1 = F.softmax(et, dim=1)  # et = softmax(et)
        # et1 最後加權的attention score
        # assign 0 probability for padded elements
        # print('et1',et1.shape)
        # print('enc_padding_mask',enc_padding_mask.shape)
        # print('----------------------------')
        at = et1 * enc_padding_mask
        # torch.sum(input, dim, keepdim=False, out=None) → Tensor 返回新的张量，其中包括输入张量input中指定维度dim中每行的和。
        # 若keepdim值为True，则在输出张量中，除了被操作的dim维度值降为1，其它维度与输入张量input相同
        normalization_factor = at.sum(1, keepdim=True)
        at = at / normalization_factor  # 做 normalization 得 context vector

        at = at.unsqueeze(1)                    #batch_size,1,b_seq_len          # torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度
        # Compute encoder context vector
        ct_e = T.bmm(at, h)                     #batch_size, 1, 2*hid_size      #  將 encoder hidden states 與 attention distribution 做矩阵乘法得 context vector
        ct_e = ct_e.squeeze(1)
        at = at.squeeze(1)                                           # torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度
        # print('ct_e',ct_e.shape)
        # print('at',at.shape)
        # print('h',h.shape)
        # print('sum_temporal_srcs',sum_temporal_srcs.shape)
        # print('-------------------------------------------------')
        return ct_e, at, sum_temporal_srcs  # context vector , attention score , sum_temporal_srcs (value != None if self attention )

class decoder_attention(nn.Module):
    def __init__(self):
        super(decoder_attention, self).__init__()
        if config.intra_decoder:
            self.W_prev = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            self.W_s = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.W_t = nn.Linear(config.emb_dim , config.hidden_dim)
            self.v = nn.Linear(config.hidden_dim, 1, bias=False)

    def forward(self, s_t, prev_s, sum_k_emb):
        '''Perform intra_decoder attention
        Args
        :param s_t: hidden state of decoder at current time step
        :param prev_s: If intra_decoder attention, contains list of previous decoder hidden states
        '''
        if config.intra_decoder is False:
            ct_d = get_cuda(T.zeros(s_t.size())) # set c1_d to vector of zeros
        elif prev_s is None:
            ct_d = get_cuda(T.zeros(s_t.size()))
            prev_s = s_t.unsqueeze(1)               #batch_size, 1, hid_size
        else:
            # Standard attention technique (eq 1 in Pointer-Generator Networks - https://arxiv.org/pdf/1704.04368.pdf)
            # et = tanh ( W_prev(prev_s)  + W_s(st_hat) )
            et = self.W_prev(prev_s)                # batch_size,t-1,hid_size
            dec_fea = self.W_s(s_t).unsqueeze(1)    # batch_size,1,hid_size
            et = et + dec_fea
            
            if config.key_attention: 
                k_t = self.W_t(sum_k_emb).unsqueeze(1)
                if k_t.shape[0] == et.shape[0] : et = et + k_t

            et = T.tanh(et)                         # batch_size,t-1,hid_size
            et = self.v(et).squeeze(2)              # batch_size,t-1
            # intra-decoder attention     (eq 7 & 8 in DEEP REINFORCED MODEL - https://arxiv.org/pdf/1705.04304.pdf)
            at = F.softmax(et, dim=1).unsqueeze(1)  #batch_size, 1, t-1
            ct_d = T.bmm(at, prev_s).squeeze(1)     #batch_size, hid_size    #  將 previous decoder hidden states 與 attention distribution 做矩阵乘法得 decoder context vector
            prev_s = T.cat([prev_s, s_t.unsqueeze(1)], dim=1)    #batch_size, t, hid_size  # 將目前計算的decoder state 合併到 previous decoder hidden states

        return ct_d, prev_s


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.enc_attention = encoder_attention()
        self.dec_attention = decoder_attention()
        self.x_context = nn.Linear(config.hidden_dim*2 + config.emb_dim, config.emb_dim)
        self.x_key_context = nn.Linear(config.hidden_dim*2 + config.emb_dim*2, config.emb_dim)


        self.lstm = nn.LSTMCell(config.emb_dim, config.hidden_dim) # LSTMCell 就是單一時刻下的LSTM， seq_len = 1
        init_lstm_wt(self.lstm)

        self.p_gen_linear = nn.Linear(config.hidden_dim * 5 + config.emb_dim, 1)

        #p_vocab
        self.V = nn.Linear(config.hidden_dim*4, config.hidden_dim)
        self.V1 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.V1)

    def forward(self, x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, key_x, key_mask):
        if type(key_mask) == T.Tensor:
            key_mask = T.unsqueeze(key_mask,2)
            key_x = T.mul(key_x,key_mask)
        # print('ct_e',ct_e.shape);
        sum_k_emb = T.sum(key_x,1)  # sum of word embeddings for d keywords
        # print('decoder_cat',T.cat([x_t, ct_e], dim=1).shape); print('x_context',(config.hidden_dim*2 + config.emb_dim, config.emb_dim))
        x = self.x_context(T.cat([x_t, ct_e], dim=1)) # 将x_t, ct_e（tensor）拼接在一起，形成新的輸入向量 x包含embedding向量以及context vector
        # print('x',x.shape)
        # print('s_t',s_t[0].shape)
        s_t = self.lstm(x, s_t) # LSTMCell 吐出 s_t        
        # print('s_t',s_t[0].shape)
        dec_h, dec_c = s_t # s_t拆分為隱藏層狀態 dec_h 以及 序列語意向量 dec_c
        st_hat = T.cat([dec_h, dec_c], dim=1) # st_hat 由 隱藏層狀態dec_h 以及 序列語意向量 dec_c 拼接 => 2 * config.hidden_dim
        # print('st_hat',st_hat.shape)
        # 將拼接的 st_hat(dec_h & dec_c) 與 enc_out 計算出一個新的 attn_dist 並根據 attn_dist 計算出一個新的encoder語意向量 ct_e
        ct_e, attn_dist, sum_temporal_srcs = self.enc_attention(st_hat, enc_out, enc_padding_mask, sum_temporal_srcs, sum_k_emb)
        # print('enc_ct_e',ct_e.shape);
        # 根據 當下的decoder hidden state 以及過去所有的 previous decoder hidden states 計算出一個新的decoder語意向量 ct_d
        ct_d, prev_s = self.dec_attention(dec_h, prev_s, sum_k_emb)        #intra-decoder attention
        # print('ct_d',ct_d.shape);print('prev_s',prev_s)
        # Token generation and pointer              # (eq 9 in DEEP REINFORCED MODEL - https://arxiv.org/pdf/1705.04304.pdf) #按维数1拼接（横着拼）
        #  計算各個decoing step 使用pointer mechanism 做預測單詞的 prob distribution
        p_gen = T.cat([ct_e, ct_d, st_hat, x], 1)
        p_gen = self.p_gen_linear(p_gen)            # batch_size,1
        p_gen = T.sigmoid(p_gen)                    # batch_size,1

        # (eq 4 in Pointer - Generator Networks - https: // arxiv.org / pdf / 1704.04368.pdf)
        #  P_vocab = softmax(V'(V[st,ht*]))         # st => dec_h ; ht* => [ct_e | ct_d]
        out = T.cat([dec_h, ct_e, ct_d], dim=1)     # [ ht* , st ]   # batch_size, 4*hid_size
        out = self.V(out)                           # batch_size,hid_size
        out = self.V1(out)                          # batch_size, n_vocab
        vocab_dist = F.softmax(out, dim=1)          #  P_vocab
        vocab_dist = p_gen * vocab_dist             #  P_gen *P_vocab(w) # (generate mode) select word from vocab distribution
        attn_dist_ = (1 - p_gen) * attn_dist        #  (1 - p_gen) *P_vocab(w) # (copy mode) select word from source attention distribution => (word not appear in the source document) => (OOV words)

        # pointer mechanism (as suggested in eq 9 Pointer-Generator Networks - https://arxiv.org/pdf/1704.04368.pdf)
        # extra_zeros : 裝載不在詞彙字典的詞分布
        if extra_zeros is not None:
            vocab_dist = T.cat([vocab_dist, extra_zeros], dim=1) # 更新詞彙表分布
        '''
        target = self.scatter_add_(dim, index, other) → Tensor
        将张量other所有值加到index张量中指定的index处的self中. 对于中的每个值other ，
        它被添加到索引中的self其通过它的索引中指定的other用于dimension != dim ，并通过在相应的值index为dimension = dim
        
        使用CUDA后端时，此操作可能会导致不确定的行为，不容易关闭. 请参阅有关可重现性的说明作为背景.

        REPRODUCIBILITY
        在PyTorch发行版，单独的提交或不同的平台上，不能保证完全可重复的结果. 此外，即使使用相同的种子，结果也不必在CPU和GPU执行之间再现.
        https://s0pytorch0org.icopy.site/docs/stable/notes/randomness.html
        '''
        # print(enc_batch_extend_vocab.shape)
        # print(attn_dist_.shape)
        final_dist = vocab_dist.scatter_add(1, enc_batch_extend_vocab, attn_dist_) # 已確定不是scatter_add問題
        # print('vocab_dist',vocab_dist.shape)
        # print('attn_dist_',attn_dist_.shape)
        # print(s_t.shape)
        # print('final ct_e',ct_e.shape);
        # print(sum_temporal_srcs.shape)
        # print(prev_s.shape)
        # print('--------------------------------------------------------------------------')
        return final_dist, s_t, ct_e, sum_temporal_srcs, prev_s



class Model(nn.Module):
    def __init__(self,pre_train_emb,word_emb_type,vocab):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if pre_train_emb:
            self.embeds = nn.Embedding(config.vocab_size, config.emb_dim)
            # FastText & word2Vec same format
            if word_emb_type == 'word2Vec' or word_emb_type == 'FastText':    
                weight = get_Word2Vec_weight(vocab);# print('weight',len(weight))
            elif word_emb_type == 'glove':
                weight = get_glove_weight(vocab)
#                weight = get_glove_weight2(vocab)                
            elif word_emb_type == 'bert':    
                weight = get_bert_weight(vocab)
                
            self.embeds = T.nn.Embedding.from_pretrained(weight)
            self.embeds.weight.requires_grad = config.emb_grad
            
            

        else:
            self.embeds = nn.Embedding(config.vocab_size, config.emb_dim)
            init_wt_normal(self.embeds.weight)
            # requires_grad指定是否在训练过程中对词向量的权重进行微调
            self.embeds.weight.requires_grad = True

        self.encoder = get_cuda(self.encoder)
        self.decoder = get_cuda(self.decoder)
        self.embeds = get_cuda(self.embeds)

