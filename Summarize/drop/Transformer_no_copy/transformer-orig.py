import torch as T
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Parameter
from utils import config
import torch.nn.functional as F
from utils.train_util import get_cuda
import torchsnooper
import numpy as np
import sys
import math
# from utils.initialize import init_linear_weight, init_wt_normal, get_init_embedding
# from utils.initialize import get_Word2Vec_weight, get_glove_weight, get_glove_weight2
# from utils.initialize import get_bert_weight
from utils.initialize import *

# import logging
# logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("pytorch_pretrained_bert").setLevel(logging.ERROR)

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


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        # in_proj: | q | k | v |, 这些参数给F.linear用 
        self.in_proj_weight = Parameter(T.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(T.Tensor(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=False):
        """            
            key_padding_mask: batch  x seqlen
            attn_mask:  tgt_len x src_len
            mask 1 为忽略项
            returns: attn[b_sz * tgtlen * srclen]
        """
        # print('query',query.shape)
        # print('key',key.shape)
        # print('value',value.shape)
        
        # 通过数据指针判断是自注意力还是... # data_ptr返回tensor首元素的内存地址
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()   # py支持连等号的奥
        kv_same = key.data_ptr() == value.data_ptr()

        # tgt_len, b_sz, embed_dim = query.size()
        b_sz, tgt_len, embed_dim = query.size()
        assert key.size() == value.size()

        if qkv_same: # 合在一起是能加快速度么...
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling
        
        q = q.contiguous().view(tgt_len, b_sz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, b_sz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, b_sz * self.num_heads, self.head_dim).transpose(0, 1)

        # q = q.contiguous().view(tgt_len, b_sz * self.num_heads, self.head_dim)
        # k = k.contiguous().view(-1, b_sz * self.num_heads, self.head_dim)
        # v = v.contiguous().view(-1, b_sz * self.num_heads, self.head_dim)

        src_len = k.size(1)
        # k,v: b_sz*heads x src_len x dim
        # q: b_sz*heads x tgt_len x dim 

        attn_weights = T.bmm(q, k.transpose(1, 2))      # Q * K^T
        assert list(attn_weights.size()) == [b_sz * self.num_heads, tgt_len, src_len]        
        if attn_mask is not None:   # tgt self-att mask (triu)
            # print('attn_mask',attn_mask.shape)
            attn_weights.masked_fill_(
                # attn_mask.unsqueeze(0).bool(), # masked_fill expects num of dim tobe same
                attn_mask.unsqueeze(0).bool(), # masked_fill expects num of dim tobe same
                float('-inf')
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(b_sz, self.num_heads, tgt_len, src_len) # extends            
            attn_weights.masked_fill_(
                # mask: b_sz, 1, 1, src_len
                # key_padding_mask.transpose(0, 1).unsqueeze(1).unsqueeze(2).bool(),
                key_padding_mask.unsqueeze(1).unsqueeze(2).bool(),
                float('-inf')
            )
            attn_weights = attn_weights.view(b_sz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        
        if self.weights_dropout:    # !!! attention 的 dropout...?
            attn_weights = self.dropout(attn_weights)
        
        attn = T.bmm(attn_weights, v)
        if not self.weights_dropout:
            attn = self.dropout(attn)

        assert list(attn.size()) == [b_sz * self.num_heads, tgt_len, self.head_dim]
        
        # attn = attn.transpose(0, 1).contiguous().view(tgt_len, b_sz, embed_dim)
        # attn = attn.contiguous().view(tgt_len, b_sz, embed_dim)
        attn = attn.contiguous().view(b_sz, tgt_len, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.view(b_sz, self.num_heads, tgt_len, src_len)
            #attn_weights, _ = attn_weights.max(dim=1)  # max pooling
            #attn_weights = attn_weights[:, 0, :, :]    # 只拿第k个head > <
            attn_weights = attn_weights.mean(dim=1)    # mean pooling
            attn_weights = attn_weights.transpose(0, 1)
            # attn_weights = attn_weights
        else:
            attn_weights = None
        # print('MultiheadAttention',attn.shape)
        # print('MultiheadAttention',attn)
        return attn, attn_weights

    def in_proj_qkv(self, query):
        # chunk: splits a tensor into a specific number of chunks.
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, inputs, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]

        return F.linear(inputs, weight, bias)

class SelfAttentionMask(nn.Module):
    def __init__(self, init_size = 100):
        super(SelfAttentionMask, self).__init__()
        self.weights = SelfAttentionMask.get_mask(init_size)
    
    @staticmethod
    def get_mask(size):
        weights = T.triu(T.ones((size, size), dtype = T.uint8), 1)
        return weights

    def forward(self, size):
        if self.weights is None or size > self.weights.size(0):
            self.weights = SelfAttentionMask.get_mask(size)
        res = get_cuda(self.weights[:size,:size]).detach()
        return res

class SinusoidalPositionalEncoding(nn.Module):
    """
        Attention is All You Need ver.
        Positional Encoding 的计算!
        PE(pos, 2i) = sin(pos / (10000 ^ (2 * i / d_model)))
    """
    def __init__(self, d_model, max_size = 512):
        super(SinusoidalPositionalEncoding, self).__init__()

        pe = T.zeros(max_size, d_model)
        position = T.arange(0, max_size).unsqueeze(1)
        div = T.exp(T.arange(0, d_model, 2, dtype=T.float) *
                        - (math.log(10000.0) / d_model))
        pe[:, 0::2] = T.sin(position.float() * div)
        pe[:, 1::2] = T.cos(position.float() * div)
        pe.unsqueeze_(1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len, b_sz = x.size()
        return get_cuda(self.pe[:seq_len,:,:].expand(-1, b_sz, -1)).detach()

class LearnedPositionalEmbedding(nn.Module):
    """
        This module produces LearnedPositionalEmbedding.
    """
    def __init__(self, embedding_dim, init_size=405):
        super(LearnedPositionalEmbedding, self).__init__()
        self.weights = nn.Embedding(init_size, embedding_dim)   # nn.embedding 默认finetune
        self.reset_parameters()
    
    def reset_parameters(self):
        """ 跟词向量采用了相同的初始化方式...! """
        nn.init.normal_(self.weights.weight, std=0.02)

    def forward(self, inputs, offset=0):
        """Input is expected to be of size [seq_len x b_sz]."""
        seq_len, b_sz = inputs.size()
        positions = get_cuda((offset + T.arange(seq_len)))
        res = self.weights(positions).unsqueeze(1).expand(-1, b_sz, -1)
        return res

class LayerNorm(nn.Module):
    """
        LayerNorm的原型函数... 
        说的那么麻烦...其实就是沿最后一维作标准化
        为了不让取值集中在0附近(失去激活函数的非线性性质), 它还非常贴心地添加了平移和缩放功能...!
    """
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(T.Tensor(hidden_size))
        self.bias = nn.Parameter(T.Tensor(hidden_size))
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 1.)
        nn.init.constant_(self.bias, 0.)
    
    def forward(self, x):
        # print('LayerNorm_x',x.shape)
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / T.sqrt(s + self.eps)
        # print('LayerNorm_o',(self.weight * x + self.bias).shape)
        return self.weight * x + self.bias

class WordProbLayer(nn.Module):
    def __init__(self, hidden_size, dict_size, dropout, copy=False, coverage=False):
        super(WordProbLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dict_size = dict_size
        self.copy = copy
        self.coverage = coverage
        self.dropout = dropout
    
        if self.copy:
            # add an extra single headed att-layer to implement copy-mechanism ヽ(･ω･´ﾒ)
            self.external_attn = MultiheadAttention(self.hidden_size, 1, self.dropout, weights_dropout=False)
            self.proj = nn.Linear(self.hidden_size * 3, self.dict_size)
            self.prob_copy = nn.Linear(self.hidden_size * 3, 1, bias=True)
        else:
            self.proj = nn.Linear(self.hidden_size, self.dict_size)
        
        self.init_weights()

    def init_weights(self):
        init_linear_weight(self.proj)
        if self.copy: init_linear_weight(self.prob_copy)

    def forward(self, h, emb=None, memory=None, src_mask=None, tokens=None, extra_zeros=None):
        """
            h: final hidden layer output by decoder [ seqlen * b_sz * hidden ]
            memory: output by encoder               
            emb: word embedd for current token...
            src_mask: padding mask
            tokens: indices of words from source text [include extended vocabs]
            max_ext_len: max len of extended vocab
            returns: softmaxed probabilities, copy attention distribs
        """
        if self.copy:
            # dists: seqlen * b_sz * seqlen
            # pred: seqlen * b_sz * vocab_size
            atts, dists = self.external_attn(query=h, key=memory, value=memory, key_padding_mask=src_mask, need_weights = True)
            pred = T.softmax(self.proj(T.cat([h, emb, atts], -1)), dim=-1)        #原词典上的概率分布
            if extra_zeros is not None:
                pred = T.cat((pred, extra_zeros.repeat(pred.size(0),1,1)), -1)
            g = T.sigmoid(self.prob_copy(T.cat([h, emb, atts], -1)))              #计算生成概率g
            # tokens应与dists的大小保持一致, 并仅在最后一维大小与pred不同
            tokens = tokens.unsqueeze(0).repeat(pred.size(0), 1, 1)
            # 在最后一维(即预测概率分布)上scatter
            pred = (g * pred).scatter_add(2, tokens, (1 - g) * dists)
        else:
            pred = T.softmax(self.proj(h), dim=-1)
            dists = None
        return pred, dists

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, label_smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        self.size = size

        self.smoothing_value = label_smoothing / (size - 2)    # not padding idx & gold
        self.one_hot = get_cuda(T.full((1, size), self.smoothing_value))
        self.one_hot[0, self.padding_idx] = 0
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
            支持扩展词典, 比如copy机制使用的src词典
            input size: b_sz*seq_en, vocab
            return: 0维tensor
        """
        real_size = output.size(1)  
        if real_size > self.size:
            real_size -= self.size  # real size即扩展词典的大小
        else:
            real_size = 0   
        model_prob = self.one_hot.repeat(target.size(0), 1) # -1 * vocab 
        # print('target',target.shape)
        # print('model_prob',model_prob.shape)
        # print('real_size',real_size)
        if real_size > 0: 
            ext_zeros = get_cuda(T.full((model_prob.size(0), real_size), self.smoothing_value))
            model_prob = T.cat((model_prob, ext_zeros), -1)
            # print('model_prob2',model_prob.shape)
        # @scatter 的正确使用方法
        # 只有被声明的那一维拥有与src和index不同的维数
        # scatter_(input, dim, index, src)
        # 将src中数据根据index中的索引按照dim的方向填进input中
        # print('model_prob',model_prob.shape)
        # print('target',target.shape)
        # print('confidence',self.confidence)
        model_prob.scatter_(1, target, self.confidence)
        model_prob.masked_fill_((target == self.padding_idx), 0.)

        return F.kl_div(output, model_prob, reduction='sum')

class TransformerLayer(nn.Module):
    
    def __init__(self, embed_dim, ff_embed_dim, num_heads, dropout, with_external=False, weights_dropout = True):
        """
            external: 外部注意力(target to source)
            Feed Forward : fc1, fc2
            SubLayer1 : self_attn, attn_layer_norm
            SubLayer2 : fc1, fc2, ff_layer_norm
        """
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.attn_layer_norm = LayerNorm(embed_dim, eps = 1e-12)
        self.ff_layer_norm = LayerNorm(embed_dim, eps = 1e-12)
        self.with_external = with_external
        self.dropout = nn.Dropout(p=dropout)
        if self.with_external:
            self.external_attn = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
            self.external_layer_norm = LayerNorm(embed_dim, eps = 1e-12)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, 
                self_padding_mask = None, self_attn_mask = None,
                external_memories = None, external_padding_mask=None,
                need_weights=False):
        """ returns: x, self_att or src_att """
        # x: seq_len x b_sz x embed_dim
        
        # print('x',x.shape)
        residual = x
        x, self_attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask, need_weights = need_weights)
        x = self.dropout(x)
        # print('self_attn',x.shape)
        # print('self_attn',x)
        # print('residual',residual.shape)
        # print('x',x.shape)
        # print('residual + x',(residual + x).shape)
        x = self.attn_layer_norm(residual + x)  # norm前都接dropout嗷

        if self.with_external:
            residual = x            
            x, external_attn = self.external_attn(query=x, key=external_memories, value=external_memories, key_padding_mask=external_padding_mask, need_weights = need_weights)
            x = self.dropout(x)
            x = self.external_layer_norm(residual + x)
        else:
            external_attn = None

        # Position-wise FF        
        residual = x
        #x = self.dropout(gelu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        x = self.ff_layer_norm(residual + x)
        return x, self_attn, external_attn

class Model(nn.Module):
    def __init__(self,pre_train_emb,word_emb_type,vocab,config):
        super(Model, self).__init__()
        dropout = 0.2
        num_layers = 4
        d_ff = 1024
        self.padding_idx = 0 # pad token id 
        smoothing = 0.1
        # self.copy = True
        self.copy = config.copy
        num_heads = 8
        self.is_predicting = False

        self.word_embed = get_init_embedding(config, vocab)
        # padding_idx指定用以padding的索引位置
        self.word_embed = nn.Embedding(config.vocab_size, config.emb_dim, self.padding_idx)

        self.dropout = nn.Dropout(p=dropout)

        self.attn_mask = SelfAttentionMask()
        # self.pos_embed = SinusoidalPositionalEmbedding(config.emb_dim)
        # self.pos_embed = SinusoidalPositionalEncoding(config.emb_dim)
        self.pos_embed = LearnedPositionalEmbedding(config.emb_dim)
        self.enc_layers = nn.ModuleList()
        self.dec_layers = nn.ModuleList()
        self.emb_layer_norm = LayerNorm(config.emb_dim, eps = 1e-12)    # copy & coverage not implemented...
        self.word_prob = WordProbLayer(config.hidden_dim, config.vocab_size, dropout, copy=self.copy)
        self.label_smoothing = LabelSmoothing(config.vocab_size, self.padding_idx, smoothing)

        for _ in range(num_layers):
            self.enc_layers.append(TransformerLayer(config.hidden_dim, d_ff, num_heads,
            dropout))
            # self.dec_layers.append(TransformerLayer(config.hidden_dim, d_ff, num_heads,
            # dropout, with_external=True))
            self.dec_layers.append(TransformerLayer(config.hidden_dim, d_ff, num_heads,
            dropout, with_external=False))
    
    def reset_parameters(self):
        init_uniform_weight(self.word_embed.weight)

    def label_smoothing_loss(self, pred, gold, mask = None):
        """
            mask 0 表示忽略 
            gold: seqlen, b_sz
        """
        if mask is None: mask = gold.ne(self.padding_idx)
        seq_len, b_sz = gold.size()
        # KL散度需要预测概率过log...
        pred = T.log(pred.clamp(min=1e-8))  # 方便实用的截断函数P 
        # print('log_pred',pred.shape)
        # print('label_smoothing_para1',pred.view(seq_len * b_sz, -1).shape)
        # print('label_smoothing_para2',gold.contiguous().view(seq_len * b_sz, -1).shape)
        # 本损失函数中, 每个词的损失不对seqlen作规范化
        return self.label_smoothing(pred.view(seq_len * b_sz, -1),
                    gold.contiguous().view(seq_len * b_sz, -1)) / mask.sum() # avg loss
        
    def nll_loss(self, pred:T.Tensor, gold, dec_lens):
        """
            nll: 指不自带softmax的loss计算函数
            pred: seqlen, b_sz, vocab
            gold: seqlen, b_sz
        """
        # print('gold',gold.shape)
        # print('gold',gold[0,:])
        # print('pred',pred.shape)
        # pred[T.isnan(pred)] = 0
        pred_gather = pred.gather(dim=2, index=gold.unsqueeze(2))
        # print('pred_gather',pred_gather.shape)
        gold_prob = pred_gather.squeeze(2).clamp(min=1e-8)  # cross entropy
        # print('gold_prob',gold_prob.shape)
        # print('gold_prob',gold_prob.log().masked_fill(gold.eq(self.padding_idx), 0.).sum(dim=0).shape)
        gold_prob = gold_prob.log().masked_fill(gold.eq(self.padding_idx), 0.).sum(dim=1) / dec_lens   # batch内规范化
        # print('gold_prob_mean',-gold_prob.mean())
        # print('-----------------------------------')
        return -gold_prob.mean()
    
    def encode(self, inputs, padding_mask = None):
        if padding_mask is None: 
            padding_mask = inputs.eq(self.padding_idx)
        # inputs = inputs[:,0]
        # print('inputs',inputs.shape)
        # print(config.vocab_size, config.emb_dim, self.padding_idx)
        # print(inputs)
        #test word_embed
        # print('word_embed',self.word_embed(inputs).shape)
        #test pos_embed
        # print('pos_embed',self.pos_embed(inputs).shape)        
        x = self.word_embed(inputs) + self.pos_embed(inputs)
        # x = self.word_embed(inputs)
        # x = self.pos_embed(inputs)
        # print('x',x.shape)
        x = self.dropout(self.emb_layer_norm(x))

        # print('padding_mask',padding_mask.shape,padding_mask)
        for idx ,layer in enumerate(self.enc_layers):
            # print('enc_encode',idx)
            # print(idx,'x',x)
            x, _, _ = layer(x, self_padding_mask=padding_mask)        
            # print('x',x)

        return x, padding_mask

    def decode(self, inputs, src, src_padding_mask, tgt_padding_mask=None,
                src_extend_vocab = None, extra_zeros = None):      # if copy enabled
        """ copy not implemented """
        _ , seqlen = inputs.size()
        if not self.is_predicting and tgt_padding_mask is None:
            tgt_padding_mask = inputs.eq(self.padding_idx)
        x = self.word_embed(inputs) + self.pos_embed(inputs)
        x = self.dropout(self.emb_layer_norm(x))
        emb = x

        self_attn_mask = self.attn_mask(seqlen)

        for idx, layer in enumerate(self.dec_layers):
            # print('dec_decode',idx)
            # print('input',x.shape)
            # print('self_padding_mask',padding_mask.shape)
            # print('self_attn_mask',self_attn_mask.shape)
            # print('external_memories',src.shape)
            # print('external_padding_mask',src_padding_mask.shape)
            x,_,_ = layer(x, self_padding_mask=tgt_padding_mask, 
            self_attn_mask=self_attn_mask,
            external_memories=src, 
            external_padding_mask=src_padding_mask)
        if self.copy:
            pred, attn = self.word_prob(x, emb, memory=src, src_mask=src_padding_mask,
                        tokens = src_extend_vocab, extra_zeros= extra_zeros)
        else: pred, attn = self.word_prob(x)
        # print('x',x.shape)
        # print('x',x)
        # print('word_prob_pred',pred.shape)
        # print('word_prob_pred',pred)
        # print('word_prob_attn',attn)
        return pred, attn

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None,
            src_extend_vocab = None, extra_zeros = None):       # if copy enabled
        """
            src&tgt: seqlen, b_sz
        """ 
        # encode(self, inputs, padding_mask = None)
        src_enc, src_padding_mask = self.encode(src, src_padding_mask)
        
        # decode(self, inputs, src, src_padding_mask, padding_mask=None,
        #         src_extend_vocab = None, extra_zeros = None)
        pred, attn = self.decode(tgt, src_enc, src_padding_mask, tgt_padding_mask,
        src_extend_vocab, extra_zeros)
        # print('tgt',tgt.shape)
        # print('src_enc',src_enc.shape)
        # print('src_padding_mask',src_padding_mask.shape)
        # print('tgt_padding_mask',tgt_padding_mask.shape)
        # print('src_extend_vocab',src_extend_vocab.shape)
        # tgt torch.Size([6, 1])
        # src_enc torch.Size([64, 1, 512])
        # src_padding_mask torch.Size([64, 1])
        # tgt_padding_mask torch.Size([6, 1])
        # src_extend_vocab torch.Size([1, 64])
        # print('extra_zeros',extra_zeros)
        # extra_zeros tensor([[[0., 0., 0., 0., 0., 0., 0.]]], device='cuda:0')
        return pred
       

