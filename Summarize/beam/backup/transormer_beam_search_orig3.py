import numpy as np
import torch as T
# from utils import config
from utils.train_util import *

## Crimson Resolve
alpha = 0.9
beta = 5.

class Beam(object):
    def __init__(self, tokens, log_probs, start_id, end_id, unk_id, coverage=None):
        # # beam_size = batch_size * beam_n
        # self.tokens = T.LongTensor(config.beam_size,1).fill_(start_id)  #(beam_size, t) after t time steps
        # # 初始beam score分數為-30
        # # self.scores = T.FloatTensor(config.beam_size,1).fill_(-30)      #beam_size,1; Initial score of beams = -30
        # self.scores = T.FloatTensor(config.beam_size,1).fill_(0)      #beam_size,1; Initial score of beams = -30
        # self.tokens, self.scores = get_cuda(self.tokens), get_cuda(self.scores)
        # self.scores[0][0] = 0  
        # # self.coverage = coverage
        
        # # 每個batch中欲被decode的元素，將根據beam_size進行複製
        # # if type(coverage) == T.tensor:
        # self.coverage = coverage.unsqueeze(0).repeat(config.beam_size, 1) #beam_size, 2*hid_size
        # # self.sum_temporal_srcs = None
        # # self.prev_s = None
        # self.done = False
        # self.end_id = end_id
        # self.unk_id = unk_id
        # # print('self.tokens',self.tokens.shape)
        # # print('self.coverage',self.coverage.shape)

        self.tokens = tokens
        self.start_id = start_id
        self.coverage = coverage 
        self.log_probs = log_probs
        self.done = False
        self.end_id = end_id
        self.unk_id = unk_id

    def extend(self, token, log_prob, start_id, end_id, unk_id, coverage = None):
        return Beam(tokens = self.tokens + [token],
                        log_probs = self.log_probs + [log_prob],
                        coverage = None if coverage is None else (self.coverage+coverage),
                        start_id = start_id,
                        end_id = end_id,   
                        unk_id = unk_id)

    def get_current_state(self):
        tokens = self.tokens[:,-1].clone()
        for i in range(len(tokens)):
            if tokens[i].item() >= config.vocab_size: # 如果token id大於vocab_size，則將其置換為unk_id
                tokens[i] = self.unk_id
        return tokens

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)

    @property
    def coverage_prob(self):
        return sum(self.log_probs) + self.c_score

    @property
    def decay_prob(self):
        # length penalty with coverage
        penalty = ((5.0+(len(self.tokens) + 1)) / 6.0)**alpha
        return sum(self.log_probs)/ penalty

    
    def advance(self, prob_dist, coverage):
        '''Perform beam search: Considering the probabilites of given n_beam x n_extended_vocab words, select first n_beam words that give high total scores
        :param prob_dist: (beam, n_extended_vocab)
        :param hidden_state: Tuple of (beam, hid_size) tensors
        :param context:   (beam, 2*n_hidden)
        :param sum_temporal_srcs:   (beam, n_seq)
        :param prev_s:  (beam, t, hid_size)
        '''
        n_extended_vocab = prob_dist.size(1)
        # 將機率轉化為log score
        # log_probs = T.log(prob_dist+config.eps)                         #beam_size, n_extended_vocab # log_probs(16,36333)
        log_probs = T.log(prob_dist)                         #beam_size, n_extended_vocab # log_probs(16,36333)
        # print('log_probs',log_probs.shape)
        # 重新計算一個新的分數
        scores = log_probs + self.scores                                #[beam_size, n_extended_vocab=36333]
        # 將每一個分數展開回一維的tensor
        scores = scores.view(-1,1)                                      #beam_size*n_extended_vocab, 1


        # self.tokens使用 sort_hypos 將beam 依優質假設機率排序
        


        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
        # 求tensor中某个dim的前k(beam_size)大的值以及对应的index
        best_scores, best_scores_id = T.topk(input=scores, k=config.beam_size, dim=0)   #will be sorted in descending order of scores

        # best_scores: 16個beam_size的最佳分數
        # 因為best_scores_id總共有beam_size*vocab_size個非重複ID，因此除上字典大小還原
        self.scores = best_scores                                       #(beam,1); sorted
        # beams_order為經過分數排序後的beam id排名索引 
        # ex: [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
        beams_order = best_scores_id.squeeze(1)/n_extended_vocab        #(beam,); sorted
        # best_words為分數索引經餘數轉換後的詞彙表索引
        # ex: [  1996,  38329,  74662, 110995, 147328, 183661, 219994, 256327, 292660, 328993, 365326, 401659, 437992, 474325, 510658, 546991]
        best_words = best_scores_id%n_extended_vocab                    #(beam,1); sorted

        # self.coverage = coverage[beams_order]
        # 加上下一個時間點預測的best token
        self.tokens = self.tokens[beams_order]                          #(beam, t); sorted
        self.tokens = T.cat([self.tokens, best_words], dim=1)           #(beam, t+1); sorted

        #End condition is when top-of-beam is EOS.
        if best_words[0][0] == self.end_id:
            self.done = True

    def get_best(self):
        best_token = self.tokens[0].cpu().numpy().tolist()              #Since beams are always in sorted (descending) order, 1st beam is the best beam
        try:
            end_idx = best_token.index(self.end_id)
        except ValueError:
            end_idx = len(best_token)
        best_token = best_token[1:end_idx]
        return best_token

    def get_all(self):
        all_tokens = []
        for i in range(len(self.tokens)):
            all_tokens.append(self.tokens[i].cpu().numpy())
        return all_tokens

class Beam(object):
    def __init__(self, tokens, log_probs, coverage=None):
        self.tokens = tokens
        self.log_probs = log_probs
        self.coverage = coverage
     

    def extend(self, token, log_prob, state = None, coverage = None):
        return Beam(tokens = self.tokens + [token],
                        log_probs = self.log_probs + [log_prob],
                        coverage = None if coverage is None\
                                    else (self.coverage+coverage))

    @property
    def c_score(self):
        return 0 if self.coverage is None else\
                 -beta*(self.coverage.clamp_min(1.).sum() - self.coverage.size(0))

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)

    @property
    def coverage_prob(self):
        return sum(self.log_probs) + self.c_score

    @property
    def decay_prob(self):
        # length penalty with coverage
        penalty = ((5.0+(len(self.tokens) + 1)) / 6.0)**alpha
        return sum(self.log_probs)/ penalty

def sort_beams(beams):
    return sorted(beams, key=lambda h: h.coverage_prob, reverse=True)

def sort_hypos(beams):
    return sorted(beams, key=lambda h: h.decay_prob, reverse=True)

def beam_search(config, batch, model, start_id, end_id, unk_id):

    batch_size = config.batch_size
    beam_idx = T.LongTensor(list(range(batch_size)))
    # print('batch enc_hid',enc_hid[0][0].shape);    
    # print('batch ct_e',ct_e.shape);  
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, \
    coverage_t_0, _, _, _, _= get_input_from_batch(batch, config, batch_first = True)

    # enc_batch = pad_to_batch_size(enc_batch)
    # enc_padding_mask = pad_to_batch_size(enc_padding_mask)
    # enc_batch_extend_vocab = pad_to_batch_size(enc_batch_extend_vocab)
    # extra_zeros = pad_to_batch_size(extra_zeros)

    encoder_outputs, padding_mask = model.encode(enc_batch, enc_padding_mask)
    coverage=(coverage_t_0[0] if config.coverage else None)

    # decoder batch preparation, it has beam_size example initially everything is repeated
    # beams = [Beam(start_id, end_id, unk_id, coverage) for i in range(batch_size)]   #For each example in batch, create Beam object

# , start_id, end_id, unk_id, coverage=None

    # beams = [Beam(tokens=[start_id],
    #                 log_probs=[0.0] ,
    #                 start_id = start_id, end_id=end_id, unk_id=unk_id,
    #                 coverage=(coverage_t_0[0] if config.coverage else None)) 
    #                 for _ in range(config.beam_size)]

    n_rem = batch_size                                                  #Index of beams that are active, i.e: didn't generate [STOP] yet

    beams = [Beam(tokens=[start_id],
                    log_probs=[0.0] ,
                    # start_id = start_id, end_id=end_id, unk_id=unk_id,
                    coverage=(coverage_t_0[0] if config.coverage else None)) 
                    for _ in range(n_rem)]

    # encoder_outputs = [encoder_outputs] * 

                          
    results = []
    for t in range(config.max_dec_steps):
        # print('time step : %s' %(t))
        # 將batch中每個beam的元素在第一維度疊加
        # hyp_tokens = T.stack(
        #     [beam.get_current_state() for beam in beams if beam.done == False]      #remaining(rem),beam
        # )
        hyp_tokens = get_cuda(T.tensor([h.tokens for h in beams])) # NOT batch first
        # for bid in range(config.batch_size):
        #     hyp_tokens = get_cuda(T.tensor([h.tokens for h in beams])) # NOT batch first
        #     hyp_tokens.masked_fill_(hyp_tokens>=config.vocab_size, unk_id)# convert oov to unk

        #     encoder_outputs = encoder_outputs[bid]
        #     padding_mask = padding_mask[bid]
        #     enc_batch_extend_vocab = enc_batch_extend_vocab[bid]
        #     extra_zeros = extra_zeros[bid]          

        
           
        if t > 0:
            # encoder_outputs = T.stack(
            #     [encoder_outputs for beam in beams if beam.done == False]                    #rem,beam,hid_size
            # ).contiguous()

            # padding_mask = T.stack(
            #     [padding_mask for beam in beams if beam.done == False]                    #rem,beam,hid_size
            # ).contiguous()

            # enc_batch_extend_vocab = T.stack(
            #     [enc_batch_extend_vocab for beam in beams if beam.done == False]                    #rem,beam,hid_size
            # ).contiguous()

            # extra_zeros = T.stack(
            #     [extra_zeros for beam in beams if beam.done == False]                    #rem,beam,hid_size
            # ).contiguous()

            encoder_outputs = T.stack(
                [encoder_outputs[0] for _ in beams]                    #rem,beam,hid_size
            ).contiguous()

            padding_mask = T.stack(
                [padding_mask[0] for _ in beams]                    #rem,beam,hid_size
            ).contiguous()

            enc_batch_extend_vocab = T.stack(
                [enc_batch_extend_vocab[0] for _ in beams]                    #rem,beam,hid_size
            ).contiguous()

            if type(extra_zeros) == T.Tensor:
                extra_zeros = T.stack(
                    [extra_zeros[0] for _ in beams]                    #rem,beam,hid_size
                ).contiguous()
            

        # print('tgt',hyp_tokens.shape)
        # print('src_enc',encoder_outputs.shape)
        # print('src_padding_mask',padding_mask.shape)
        # print('tgt_padding_mask',None)
        # print('src_extend_vocab',enc_batch_extend_vocab.shape)
        # # print('extra_zeros',extra_zeros)
        # print('--------------------------------------------')

        pred, attn = model.decode(hyp_tokens, encoder_outputs, padding_mask, None, \
                                    enc_batch_extend_vocab, extra_zeros)
        # attn = attn[:,-1,:]  # attn: [bsz * src_len]                                    
        # -------------------------------------------------------------------------------------------
        # print('pred',pred.shape)
        log_probs = T.log(pred[:,-1,:])         # get probs for next token
        topk_log_probs, topk_ids = T.topk(log_probs, config.beam_size * 2)  # avoid all <end> tokens in top-k
        # hyp_toks = bsz => at time step 0 , and hyp_toks = candinate hyps in other time step
        # print('topk_log_probs',topk_log_probs.shape)  # [hyp_toks, beam_sz *2] at next time step
        # print('topk_ids',topk_ids.shape)     # [hyp_toks, beam_sz *2] at next time step            
        #--------------------------------------------------------------------------------------------
        all_hyp_beams = []
        num_orig_beams = 1 if t == 0 else len(beams)
        for i in range(num_orig_beams):
            h = beams[i]
            # here save states, context vec and coverage...
            # for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
            for j in range(config.beam_size):  # for each of the top 2*beam_size hyps:
                new_beam = h.extend(token=topk_ids[i, j].item(),
                                log_prob=topk_log_probs[i, j].item(),
                                coverage=attn[i,-1,:] if config.coverage and config.copy else None)
                all_hyp_beams.append(new_beam)
        #--------------------------------------------------------------------------------------------
        new_beams = []
        for h in sort_beams(all_hyp_beams):
            # 如果這個beam目前最後一個token為<end>就將其加入結果列
            if h.latest_token == end_id:
                if t >= config.min_dec_steps:
                    results.append(h)
            # 否則將其加入待測beam 序列 
            else: new_beams.append(h)
            # if len(beams) == config['beam_size'] or len(results) == config['beam_size']:
            #     break
        #--------------------------------------------------------------------------------------------
        new_beams = sort_hypos(new_beams)        
        # beams = beams[:config.beam_size] # 由於待測beam 序列 過大造成cuda memory不足，限制前beam_size hyp
        # new_beams = new_beams[:config.batch_size] # 由於待測beam 序列 過大造成cuda memory不足，限制前beam_size hyp
        # print('new_beams',len(new_beams))
        beams = new_beams[:config.beam_size]

    if len(results) == 0:
        results = beams

    beams_sorted = sort_hypos(results) # 只抓出第0個result

    predicted_words = [b.tokens[1:] for b in beams_sorted][:batch_size] 
    # print('predicted_words',len(predicted_words))
    return predicted_words
