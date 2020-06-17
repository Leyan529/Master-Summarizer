import torch as T
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import config
import torch.nn.functional as F
# from utils.seq2seq.train_util import get_cuda
from torch.distributions import Categorical
from utils.seq2seq.train_util import get_input_from_batch, get_output_from_batch
import torchsnooper
import numpy as np
import sys
from utils.seq2seq.initialize import *
from utils.seq2seq.batcher import START,END, PAD , UNKNOWN_TOKEN
from utils.seq2seq.rl_util import *
import math, copy, time
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True, dropout=0.2)
        init_lstm_wt(self.lstm) # 初始话LSTM的隐层和细胞状态

        # 同样考虑向前层和向后层
        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim) # 隐层
        init_linear_wt(self.reduce_h) 
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim) # 细胞状态
        init_linear_wt(self.reduce_c)

    def forward(self, x, seq_lens, max_enc_len):
        # Starting var:.. x = tensor<(1, 27, 256), float32, cuda:0, grad>
        # Starting var:.. seq_lens = ndarray<(1,), int32>
        # x.shape torch.Size([4, 55, 256])
        # seq_lens.shape (4,)
        # print(x.shape, seq_lens)
        self.lstm.flatten_parameters()
        seq_lens = seq_lens.view(-1).tolist()
        # bsz = x.size(0)
        if x.device.index != 0:
            x = T.cat([x[0].unsqueeze(0), x], dim=0)
            seq_lens.insert(0, max_enc_len)

        packed = pack_padded_sequence(x, seq_lens, batch_first=True)
        enc_out, enc_hid = self.lstm(packed)

        enc_out,_ = pad_packed_sequence(enc_out, batch_first=True)  # x:[batch_size,max_enc_steps,emb_dim] y:[seq_lens]
        enc_out = enc_out.contiguous()                              #batch_size, b_seq_len, 2*hid_size

        h, c = enc_hid                                              #shape of h: 2, batch_size, hid_size

        h = T.cat(list(h), dim=1)                                   #batch_size, 2*hid_size
        c = T.cat(list(c), dim=1)

        h_reduced = F.relu(self.reduce_h(h))                        #batch_size,hid_size
        c_reduced = F.relu(self.reduce_c(c))
        if x.device.index != 0:
            enc_out = enc_out[1:]            
            h_reduced = h_reduced[1:]
            c_reduced = c_reduced[1:]
        return enc_out, (h_reduced, c_reduced)
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "初始化時指定頭數h和模型維度d_model"
        super(MultiHeadedAttention, self).__init__()
        # 二者是一定整除的
        assert d_model % h == 0
        # 按照文中的簡化，我們讓d_v與d_k相等
        self.d_k = d_model // h
        self.h = h
        self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, sum_temporal_srcs=None):
        "實現多頭注意力模型"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        "第二步是將這一批次的數據進行變形 d_model => h x d_k"
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        "第三步，針對所有變量計算scaled dot product attention"
        x, self.attn = self.attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        # torch.Size([bsz, head, 1, 256])                            
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # result of concat : torch.Size([2, 1, 1024])
        ct_e = self.linears[-1](x).squeeze(1)
        return ct_e, self.attn.squeeze(2), sum_temporal_srcs

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def clones(self, module, N):
        "生成n個相同的層"
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class encoder_attention(nn.Module):

    def __init__(self):
        super(encoder_attention, self).__init__()
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.W_s = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.W_t = nn.Linear(config.emb_dim , config.hidden_dim*2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=True)


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
                sum_temporal_srcs  = (T.FloatTensor(et.size()).fill_(1e-10)).cuda(et.device.index) + exp_et
                # sum_temporal_srcs  = get_cuda(T.FloatTensor(et.size()).fill_(1e-10)) + exp_et
            else:
                et1 = exp_et/sum_temporal_srcs  # eq 3 otherwise condition   #batch_size, b_seq_len
                sum_temporal_srcs = sum_temporal_srcs + exp_et # 針對自己過去所有的 source attention score 加總 (self-attention)
        else:
            # (eq 2 in h Pointer-Generator Networks - https://arxiv.org/pdf/1704.04368.pdf)
            et1 = F.softmax(et, dim=1)  # et = softmax(et)
        # et1 最後加權的attention score
        # assign 0 probability for padded elements
        # print('et1',et1)
        # print('enc_padding_mask',enc_padding_mask)
        # print('----------------------------')
        # enc_padding_mask = enc_padding_mask[:,:et1.size(1)]
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
            self.v = nn.Linear(config.hidden_dim, 1, bias=True)

    def forward(self, s_t, prev_s, sum_k_emb):
        '''Perform intra_decoder attention
        Args
        :param s_t: hidden state of decoder at current time step
        :param prev_s: If intra_decoder attention, contains list of previous decoder hidden states
        '''
        at = None
        if config.intra_decoder is False:
            # ct_d = get_cuda(T.zeros(s_t.size())) # set c1_d to vector of zeros
            # ct_d = T.zeros(s_t.size()) # set c1_d to vector of zeros
            ct_d = T.zeros(s_t.size()).cuda(s_t.device.index)     
        elif prev_s is None:
            # ct_d = get_cuda(T.zeros(s_t.size()))
            # ct_d = T.zeros(s_t.size())
            ct_d = T.zeros(s_t.size()).cuda(s_t.device.index) 
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
            at = at.squeeze(1) #batch_size, t-1 # 過去關注的t-個時間點的attention score
        return ct_d, prev_s, at


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.enc_attention = encoder_attention()
        self.enc_attention = MultiHeadedAttention(4, config.hidden_dim*2)
        self.dec_attention = decoder_attention()
        self.x_context = nn.Linear(config.hidden_dim*2 + config.emb_dim, config.emb_dim)
        self.x_key_context = nn.Linear(config.hidden_dim*2 + config.emb_dim*2, config.emb_dim)


        self.lstm = nn.LSTMCell(config.emb_dim, config.hidden_dim) # LSTMCell 就是單一時刻下的LSTM， seq_len = 1
        # self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        self.p_gen_linear = nn.Linear(config.hidden_dim * 5 + config.emb_dim, 1)
        self.p_gen_dropout = nn.Dropout(p=0.2)

        #p_vocab
        self.V = nn.Linear(config.hidden_dim*4, config.hidden_dim, bias=True)
        self.V1 = nn.Linear(config.hidden_dim, config.vocab_size, bias=True)
        init_linear_wt(self.V1)

    def forward(self, x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, key_x, key_mask):
        # self.flatten_parameters()
        if type(key_mask) == T.Tensor:            
            key_mask = T.unsqueeze(key_mask,2)
            key_x = T.mul(key_x,key_mask)
        # print('enc_out',enc_out.shape);
        sum_k_emb = T.sum(key_x,1)  # sum of word embeddings for d keywords
        # print('decoder_cat',T.cat([x_t, ct_e], dim=1).shape); print('x_context',(config.hidden_dim*2 + config.emb_dim, config.emb_dim))
        x = self.x_context(T.cat([x_t, ct_e], dim=1)) # 将x_t, ct_e（tensor）拼接在一起，形成新的輸入向量 x包含embedding向量以及context vector
        # print('x',x.shape)
        # print('s_t',s_t[0].shape)
        s_t = self.lstm(x, s_t) # LSTMCell 吐出 s_t        
        # print('s_t',s_t[0].shape)
        dec_h, dec_c = s_t # s_t拆分為隱藏層狀態 dec_h 以及 序列語意向量 dec_c
        st_hat = T.cat([dec_h, dec_c], dim=1) # st_hat 由 隱藏層狀態dec_h 以及 序列語意向量 dec_c 拼接 => 2 * config.hidden_dim
        # 將拼接的 st_hat(dec_h & dec_c) 與 enc_out 計算出一個新的 attn_dist 並根據 attn_dist 計算出一個新的encoder語意向量 ct_e
        # ct_e, attn_dist, sum_temporal_srcs = self.enc_attention(st_hat, enc_out, enc_padding_mask, sum_temporal_srcs, sum_k_emb)
        ct_e, attn_dist, sum_temporal_srcs = self.enc_attention(st_hat, enc_out, enc_out, enc_padding_mask, sum_temporal_srcs)
        # 指針p_gen和生成器vocab_dist共享第一注意頭attn_dist
        # print('enc_ct_e',ct_e.shape);
        # 根據 當下的decoder hidden state 以及過去所有的 previous decoder hidden states 計算出一個新的decoder語意向量 ct_d
        ct_d, prev_s, dec_attn = self.dec_attention(dec_h, prev_s, sum_k_emb)        #intra-decoder attention
        # print('ct_d',ct_d.shape);print('prev_s',prev_s)
        # Token generation and pointer              # (eq 9 in DEEP REINFORCED MODEL - https://arxiv.org/pdf/1705.04304.pdf) #按维数1拼接（横着拼）
        #  計算各個decoing step 使用pointer mechanism 做預測單詞的 prob distribution
        p_gen = T.cat([ct_e, ct_d, st_hat, x], 1)
        p_gen = self.p_gen_linear(p_gen)            # batch_size,1
        p_gen = self.p_gen_dropout(p_gen)

        p_gen = T.sigmoid(p_gen)                    # batch_size,1

        # (eq 4 in Pointer - Generator Networks - https: // arxiv.org / pdf / 1704.04368.pdf)
        #  P_vocab = softmax(V'(V[st,ht*]))         # st => dec_h ; ht* => [ct_e | ct_d]
        out = T.cat([dec_h, ct_e, ct_d], dim=1)     # [ ht* , st ]   # batch_size, 4*hid_size
        out = self.V(out)                           # batch_size,hid_size
        out = self.V1(out)                          # batch_size, n_vocab
        vocab_dist = F.softmax(out, dim=1)          #  P_vocab
        vocab_dist = p_gen * vocab_dist             #  P_gen *P_vocab(w) # (generate mode) select word from vocab distribution
        # attn_dist_ = (1 - p_gen) * attn_dist        #  (1 - p_gen) *P_vocab(w) # (copy mode) select word from source attention distribution => (word not appear in the source document) => (OOV words)
        attn_dist_ = (1 - p_gen) * attn_dist[:,0,:]        #  (1 - p_gen) *P_vocab(w) # (copy mode) select word from source attention distribution => (word not appear in the source document) => (OOV words)

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
        # print('enc_batch_extend_vocab',type(enc_batch_extend_vocab))
        # print('attn_dist_',type(attn_dist_))
        # print('vocab_dist',type(vocab_dist))
        # enc_batch_extend_vocab dim have error
        # vocab_dist torch.Size([8, 50003])
        # attn_dist_ torch.Size([8, 303])
        # final_dist torch.Size([8, 50003])
        # enc_batch_extend_vocab torch.Size([8, 303])
        final_dist = vocab_dist.scatter_add(1, enc_batch_extend_vocab, attn_dist_) # 已確定不是scatter_add問題
        # print('vocab_dist',vocab_dist.shape)
        # print('attn_dist_',attn_dist_.shape)
        # print('final_dist',final_dist.shape)
        # print('enc_batch_extend_vocab',enc_batch_extend_vocab.shape)
        # print(s_t.shape)
        # print('final ct_e',ct_e.shape);
        # print(sum_temporal_srcs.shape)
        # print(prev_s.shape)
        # print('--------------------------------------------------------------------------')
        enc_attn = attn_dist 
        return final_dist, s_t, ct_e, sum_temporal_srcs, prev_s, enc_attn, dec_attn



class Model(nn.Module):
    def __init__(self,pre_train_emb,word_emb_type,vocab):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.embeds = get_init_embedding(config, vocab)
        # self.encoder = get_cuda(self.encoder)
        # self.decoder = get_cuda(self.decoder)
        # self.embeds = get_cuda(self.embeds)
    
    def MLE(self, config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, \
                extra_zeros, enc_batch_extend_vocab , ct_e, \
                max_dec_len, dec_batch, target_batch):
        'Encoder data'
        # device = next(self.parameters()).device
        # enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, coverage, \
        # ct_e, enc_key_batch, enc_key_mask, enc_key_lens= \
        #     get_input_from_batch(device, batch, config, batch_first = True)
 
        enc_batch = self.embeds(enc_batch)  # Get embeddings for encoder input    
        enc_key_batch = self.embeds(enc_key_batch)  # Get key embeddings for encoder input

        enc_out, enc_hidden = self.encoder(enc_batch, enc_lens, max_enc_len)
        # print('enc_out',enc_out.shape)
        # print('enc_lens',enc_lens)
        'Decoder data'
        # dec_batch, dec_padding_mask, dec_lens, max_dec_len, target_batch = \
        # get_output_from_batch(device, batch, config, batch_first = True) # Get input and target batchs for training decoder
        step_losses = []
        s_t = (enc_hidden[0], enc_hidden[1])  # Decoder hidden states
        # x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(START))  # Input to the decoder
        x_t = T.LongTensor(len(enc_out)).fill_(START).cuda(enc_hidden[1].device.index)    # Input to the decoder
        # x_t = T.LongTensor(len(enc_out)).fill_(START)    # Input to the decoder
        prev_s = None  # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None  # Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        pred_probs = []
        # pred_labels = []
        for t in range(min(max_dec_len, config.max_dec_steps)):
            # use_gound_truth = get_cuda((T.rand(len(enc_out)) > config.gound_truth_prob)).long()  # Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
            
            use_gound_truth = (T.rand(len(enc_out)) > config.gound_truth_prob).long().cuda(dec_batch.device.index)   # Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
            # use_gound_truth = (T.rand(len(enc_out)) > config.gound_truth_prob).long()
            x_t = use_gound_truth * dec_batch[:, t] + (1 - use_gound_truth) * x_t  # Select decoder input based on use_ground_truth probabilities
            x_t = self.embeds(x_t)  
            final_dist, s_t, ct_e, sum_temporal_srcs, prev_s, _, _ = self.decoder(x_t, s_t, enc_out, enc_padding_mask,
                                                                                      ct_e, extra_zeros,
                                                                                      enc_batch_extend_vocab,
                                                                                      sum_temporal_srcs, prev_s, enc_key_batch, enc_key_mask)
            target = target_batch[:, t]
            log_probs = T.log(final_dist + config.eps)
            # step_loss = F.nll_loss(log_probs, target, reduction="none", ignore_index=PAD)
            # step_loss.to('cuda:0')   
            # step_losses.append(step_loss)
            pred_probs.append(log_probs)            
            x_t = T.multinomial(final_dist,1).squeeze()  # Sample words from final distribution which can be used as input in next time step

            is_oov = (x_t >= config.vocab_size).long()  # Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t.detach() + (is_oov) * UNKNOWN_TOKEN  # Replace OOVs with [UNK] token

        # losses = T.sum(T.stack(step_losses, 1), 1)  # unnormalized losses for each example in the batch; (batch_size)
        # batch_avg_loss = losses / dec_lens  # Normalized losses; (batch_size)
        # mle_loss = T.mean(batch_avg_loss)  # Average batch loss
        # return mle_loss
        pred_probs = T.stack(pred_probs, 1)
        return pred_probs

    def RL(self, config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, \
                extra_zeros, enc_batch_extend_vocab , ct_e, \
                max_dec_len, dec_batch, target_batch, greedy):
        # greedy
        enc_batch = self.embeds(enc_batch)  # Get embeddings for encoder input    
        enc_key_batch = self.embeds(enc_key_batch)  # Get key embeddings for encoder input
        enc_out, enc_hidden = self.encoder(enc_batch, enc_lens, max_enc_len)

        s_t = enc_hidden                                                                            #Decoder hidden states
        # x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(START))  # Input to the decoder
        x_t = T.LongTensor(len(enc_out)).fill_(START).cuda(enc_batch.device.index)     
        prev_s = None                                                                               #Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None                                                                    #Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        inds = []                       # Stores sampled indices for each time step
        decoder_padding_mask = []       # Stores padding masks of generated samples
        log_probs = []                                                                              #Stores log probabilites of generated samples
        # mask = get_cuda(T.LongTensor(len(enc_out)).fill_(1))   
        mask = T.LongTensor(len(enc_out)).fill_(1).cuda(enc_batch.device.index)                                          #Values that indicate whether [STOP] token has already been encountered; 1 => Not encountered, 0 otherwise
        # Generate RL tokens and compute rl-log-loss
        # ----------------------------------------------------------------------
        for t in range(config.max_dec_steps):
            x_t = self.embeds(x_t)
            
            probs, s_t, ct_e, sum_temporal_srcs, prev_s, _, _ = self.decoder(x_t, s_t, enc_out, enc_padding_mask,
                                                                                      ct_e, extra_zeros,
                                                                                      enc_batch_extend_vocab,
                                                                                      sum_temporal_srcs, prev_s, enc_key_batch, enc_key_mask)
            
            if greedy is False:
                multi_dist = Categorical(probs) # 建立以參數probs為標準的類別分佈
                # perform multinomial sampling
                x_t = multi_dist.sample()  # 將下一個時間點的x_t，視為下一個action   
                # 使用log_prob实施梯度方法 Policy Gradient，构造一个等价類別分佈的损失函数
                log_prob = multi_dist.log_prob(x_t)  
                log_probs.append(log_prob) #
            else:
                # perform greedy sampling distribution
                _, x_t = T.max(probs, dim=1)  # 因greedy以機率最大進行取樣，視為其中一個action   
            x_t = x_t.detach() # detach返回的 Variable 永远不会需要梯度
            inds.append(x_t)
            # mask_t = get_cuda(T.zeros(len(enc_out)))    #Padding mask of batch for current time step
            mask_t = T.zeros(len(enc_out)).cuda(enc_batch.device.index)                                          #Values that indicate whether [STOP] token has already been encountered; 1 => Not encountered, 0 otherwise

            mask_t[mask == 1] = 1                       #If [STOP] is not encountered till previous time step, mask_t = 1 else mask_t = 0
            mask[(mask == 1) + (x_t == END) == 2] = 0    #If [STOP] is not encountered till previous time step and current word is [STOP], make mask = 0
            decoder_padding_mask.append(mask_t)
            is_oov = (x_t>=config.vocab_size).long()                                                #Mask indicating whether sampled word is OOV
            x_t = (1-is_oov)*x_t + (is_oov)*UNKNOWN_TOKEN                                             #Replace OOVs with [UNK] token
        # -----------------------------------End loop -----------------------------------
        inds = T.stack(inds, dim=1)
        decoder_padding_mask = T.stack(decoder_padding_mask, dim=1)
        if greedy is False:    
            # condition greedy False                                                                     #If multinomial based sampling, compute log probabilites of sampled words
            log_probs = T.stack(log_probs, dim=1) # 在第1个维度上stack, 增加新的维度进行堆叠
            log_probs = log_probs * decoder_padding_mask # 遮罩掉為[END] or [STOP]不計算損失           #Not considering sampled words with padding mask = 0
            lens = T.sum(decoder_padding_mask, dim=1) # 計算每個sample words生成的總長度               #Length of sampled sentence
            log_probs = T.sum(log_probs, dim=1) / lens  # 計算平均的每個句子的log loss # (bs,1)        #compute normalizied log probability of a sentence
        return (inds, log_probs, enc_out)

    def forward(self, config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, \
            extra_zeros, enc_batch_extend_vocab , ct_e, \
            max_dec_len, dec_batch, target_batch, train_rl = False, art_oovs = None, original_abstract=None, vocab=None):
        
        if not train_rl:
            return self.MLE(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, \
                extra_zeros, enc_batch_extend_vocab , ct_e, \
                max_dec_len, dec_batch, target_batch)
        else:
            '''multinomial sampling'''
            sample_inds, RL_log_probs, sample_enc_out = self.RL(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, \
                extra_zeros, enc_batch_extend_vocab , ct_e, \
                max_dec_len, dec_batch, target_batch, greedy=False)

            '''# greedy sampling'''
            with T.autograd.no_grad(): 
                greedy_inds, _, gred_enc_out = self.RL(config, max_enc_len, enc_batch, enc_key_batch, enc_lens, enc_padding_mask, enc_key_mask, \
                extra_zeros, enc_batch_extend_vocab , ct_e, \
                max_dec_len, dec_batch, target_batch, greedy=True)

            # art_oovs = inputs.art_oovs
            sample_sents = to_sents(sample_enc_out, sample_inds, vocab, art_oovs)
            greedy_sents = to_sents(gred_enc_out, greedy_inds, vocab, art_oovs)
            
            sample_reward = reward_function(sample_sents, original_abstract) # r(w^s):通过根据概率来随机sample词生成句子的reward值
            baseline_reward = reward_function(greedy_sents, original_abstract) # r(w^):测试阶段使用greedy decoding取概率最大的词来生成句子的reward值

            batch_reward = T.mean(sample_reward).item()
            #Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
            rl_loss = -(sample_reward - baseline_reward) * RL_log_probs  # SCST梯度計算公式     
            rl_loss = T.mean(rl_loss)  
            return rl_loss, batch_reward
            