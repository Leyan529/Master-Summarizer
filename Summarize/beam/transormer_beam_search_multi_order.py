import numpy as np
import torch as T
# from utils import config
from utils.train_util import *

## Crimson Resolve
alpha = 0.9
beta = 5.

class Beam(object):
    def __init__(self, start_id, end_id, unk_id, coverage):
        # beam_size = batch_size * beam_n
        self.tokens = T.LongTensor(config.beam_size,1).fill_(start_id)  #(beam_size, t) after t time steps
        self.log_scores = T.FloatTensor(config.beam_size,1).fill_(0)  #(beam_size, t) after t time steps

        # 初始beam score分數為-30
        self.scores = T.FloatTensor(config.beam_size,1).fill_(-30)      #beam_size,1; Initial score of beams = -30
        self.tokens, self.scores, self.log_scores = get_cuda(self.tokens), get_cuda(self.scores), get_cuda(self.log_scores)
        self.scores[0][0] = 0  
        
        # 每個batch中欲被decode的元素，將根據beam_size進行複製
        if type(coverage) == T.tensor:
            self.coverage = coverage.unsqueeze(0).repeat(config.beam_size, 1) #beam_size, 2*hid_size
        else:
            self.coverage = None

        self.done = False
        self.end_id = end_id
        self.unk_id = unk_id
        
        # print('self.tokens',self.tokens.shape)
        # print('self.coverage',self.coverage.shape)

    def c_score(self, coverage):
        return 0 if coverage is None else\
                 -beta*(coverage.clamp_min(1.).sum() - coverage.size(0))          

    def get_current_state(self,step):
        # print('self.tokens',self.tokens.shape)
        tokens = self.tokens[:,:step].clone()
        # print('tokens',tokens.shape)
        for i in range(len(tokens)):
            for j in range(step):
                if tokens[i][j].item() >= config.vocab_size: # 如果token id大於vocab_size，則將其置換為unk_id
                    tokens[i][j] = self.unk_id
        # print('get_current_%s_state'%step,tokens)
        return tokens

    def sort_beams(self, sum_log_scores, converge, beams_order):
            # order_log_probs =  sum_log_scores[beams_order]
            if type(converge) == T.Tensor:
                # order_converge = converge[beams_order]
                order_c_score = get_cuda(T.FloatTensor([self.c_score(v) for v in converge]))
            else:
                order_c_score = get_cuda(T.FloatTensor([self.c_score(None) for _ in sum_log_scores]))
            # print('order_c_score',order_c_score, order_c_score.shape)
            coverage_prob = sum_log_scores + order_c_score
            # print('coverage_prob',coverage_prob,coverage_prob.shape)
            coverage_prob = {idx:score for idx, score in enumerate(coverage_prob)}
            beams_order = [k for k,v in sorted(coverage_prob.items(), key=lambda item: item[1],reverse=True)]
            # print('sort_beams',beams_order)
            return beams_order[:int(len(beams_order)/2)]

   
    def advance(self, n_rem, prob_dist, step, converge):
        '''Perform beam search: Considering the probabilites of given n_beam x n_extended_vocab words, select first n_beam words that give high total scores
        :param prob_dist: (beam, n_extended_vocab)
        :param hidden_state: Tuple of (beam, hid_size) tensors
        :param context:   (beam, 2*n_hidden)
        :param sum_temporal_srcs:   (beam, n_seq)
        :param prev_s:  (beam, t, hid_size)
        '''
        # print('prob_dist',prob_dist.shape)
        # print('converge',converge.shape)
        # print('c_score',self.c_score)
        n_extended_vocab = prob_dist.size(2)
        # print('n_extended_vocab',n_extended_vocab)
        # 將機率轉化為log score
        log_probs = T.log(prob_dist+config.eps)                         #beam_size, n_extended_vocab # log_probs(16,36333)

        
        # log_probs = log_probs.view(1, config.beam_size, prob_dist.shape[-1] )
        log_probs = log_probs.view(config.beam_size, prob_dist.shape[-1])

        # 重新計算一個新的分數
        scores = log_probs + self.scores                                #[beam_size, n_extended_vocab=36333]
        # print('scores',scores)
        scores = scores.view(-1,1) 
        # print('scores',scores.shape)
        # scores = scores.unsqueeze(0)

        # 將每一個分數展開回一維的tensor
        # scores = scores.view(-1,1)                                      #beam_size*n_extended_vocab, 1
        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
        # 求tensor中某个dim的前k(beam_size)大的值以及对应的index
        best_scores, best_scores_id = T.topk(input=scores, k=config.beam_size*2, dim=0, sorted=False)   #will be sorted in descending order of scores
        # print('best_scores',best_scores)
        # print('best_scores_id',best_scores_id)        

        # best_scores: 16個beam_size的最佳分數
        # 因為best_scores_id總共有beam_size*vocab_size個非重複ID，因此除上字典大小還原
        # self.scores = best_scores                                       #(beam,1); sorted
        # beams_order為經過分數排序後的beam id排名索引 
        # ex: [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]

        beams_order = best_scores_id.squeeze(1)/n_extended_vocab   # 先註解   #(beam,); sorted   # sort_beams

        # print('squeeze_best_scores_id',best_scores_id.squeeze(2))
        # best_words為分數索引經餘數轉換後的詞彙表索引
        # ex: [  1996,  38329,  74662, 110995, 147328, 183661, 219994, 256327, 292660, 328993, 365326, 401659, 437992, 474325, 510658, 546991]
        best_words = best_scores_id%n_extended_vocab                    #(beam,1); sorted
        # print('self.tokens',self.tokens)
        # print('best_words',best_words)
        # print('beams_order',beams_order)
        # -----------------------------concate best beam*2 best word-------------------------------------
        # cand_tokens = self.tokens.repeat(2,1)  
        cand_tokens = T.cat([self.tokens.repeat(2,1) , best_words], dim=1)
        cand_log_scores = T.cat([self.log_scores.repeat(2,1), best_scores], dim=1)
        # print('cand_tokens',cand_tokens)
        # print('cand_log_scores',cand_log_scores)
        # -----------------------------sort_beams-------------------------------------  
        sum_log_scores = T.sum(cand_log_scores, dim=1)
        # print('sum_log_scores',sum_log_scores,sum_log_scores.shape)
        if config.copy:
            beams_order_converge = self.sort_beams(sum_log_scores, converge[-1].repeat(2,1), beams_order)
        else:
            beams_order_converge = self.sort_beams(sum_log_scores, None, beams_order)
        # print('beams_order_converge',beams_order_converge)
        # -----------------------------sort_beams-------------------------------------   
        # # sum_log_scores = T.sum(self.log_scores, dim=1)
        # # print('sum_log_scores',sum_log_scores, type(sum_log_scores))  
        # if config.copy:
        #     beams_order_converge = self.sort_beams(sum_log_scores, converge[-1], beams_order)
        # else:
        #     beams_order_converge = self.sort_beams(sum_log_scores, None, beams_order)
        # # print('sum_log_scores',sum_log_scores)
        # # print('beams_order_converge',beams_order_converge)
        # -----------------------------sort_beams-------------------------------------  
        self.tokens = cand_tokens[beams_order_converge]   
        self.log_scores = cand_log_scores[beams_order_converge]  
        
        # print('self.log_scores',self.log_scores)
        # print('----------------------------------------')
        # self.tokens = self.tokens[beams_order]                          #(beam, t); sorted
        # self.tokens = T.cat([self.tokens, best_words], dim=1)           #(beam, t+1); sorted
        # self.tokens = self.tokens.squeeze(0)
        # print('tokens',self.tokens,self.tokens.shape)
        # print('log_scores',self.log_scores.shape)
        # print('scores',best_scores.shape)
        # self.log_scores = T.cat([self.log_scores, best_scores], dim=1)           #(beam, t+1);
        # print(self.log_scores,self.log_scores.shape)

        
        
        # -----------------------------sort_hypos-------------------------------------       
        penalties = ((5.0+(step)) / 6.0)**alpha 
        decay_prob = [(penalties / prob).item() for prob in T.sum(self.log_scores,dim=1)]
        # print('T.sum(self.log_scores,dim=1)',T.sum(self.log_scores,dim=1))
        # print('decay_prob',decay_prob)
        decay_prob = {idx:prob for idx, prob in enumerate(decay_prob)}
        hyp_order = [k for k,v in sorted(decay_prob.items(), key=lambda item: item[1],reverse=True)]
        # print('hyp_order',hyp_order)
        self.tokens = self.tokens[hyp_order]                          #(beam, t); sorted apply hyp_order
        self.log_scores = self.log_scores[hyp_order]                          #(beam, t); sorted apply hyp_order

        # print('self.tokens',self.tokens)
        # print('------------------------------------------------')
        # -------------------------------------------------------------------------------
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

def beam_search(config, batch, model, start_id, end_id, unk_id):

    batch_size = config.batch_size
    beam_idx = T.LongTensor(list(range(batch_size)))

    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, \
    coverage_t_0, _, _, _, _= get_input_from_batch(batch, config, batch_first = True)

    encoder_outputs, padding_mask = model.encode(enc_batch, enc_padding_mask)
    coverage=(coverage_t_0[0] if config.coverage else None)

    # decoder batch preparation, it has beam_size example initially everything is repeated
    beams = [Beam(start_id, end_id, unk_id, coverage) for i in range(batch_size)]   #For each example in batch, create Beam object
    n_rem = batch_size                                                  #Index of beams that are active, i.e: didn't generate [STOP] yet

    # print('encoder_outputs',encoder_outputs.shape)
    for t in range(config.max_dec_steps-1):
        # print(t)
        # print('---------------------------------------')
        # 將batch中每個beam的元素在第一維度疊加
        # print([beam.get_current_state(0) for beam in beams if beam.done == False])
        hyp_tokens = T.stack(
            [beam.get_current_state(t+1) for beam in beams if beam.done == False]      #remaining(rem),beam
        )
        if hyp_tokens.shape[0]!= config.batch_size: break   # no more hyp_tokens -STOP
            
            
        hyp_tokens = hyp_tokens.view(config.batch_size, config.beam_size * ( t + 1 ) )
        
        # hyp_tokens = get_cuda(T.tensor([h.tokens for h in beams])).transpose(0,1) # NOT batch first
        hyp_tokens.masked_fill_(hyp_tokens>=config.vocab_size, unk_id)# convert oov to unk      

        pred, attn = model.decode(hyp_tokens, encoder_outputs, padding_mask, None, \
                                    enc_batch_extend_vocab, extra_zeros)
                                           

        # advance
        pred = pred.view(n_rem, ( t + 1 ), config.beam_size , -1)  
        pred = pred[:, t :  , : , :]

        if config.copy:
            attn = attn.view(n_rem, ( t + 1 ), config.beam_size , -1)
            attn = attn[:, t :  , : , :]
        else:
            attn = None
        # pred torch.Size([2, 1, 16, 50001])
        # attn torch.Size([2, 1, 16, 238])

        # print('pred',pred.shape)   
        # print('attn',attn.shape)

        # For all the active beams, perform beam search
        active = []         #indices of active beams after beam search

        for i in range(n_rem):
            b = beam_idx[i].item()
            beam = beams[b]
            if beam.done:
                continue
            # print('pred',pred[i].shape) 
            beam.advance(n_rem, pred[i], t+1, converge = (attn[i] if config.copy else None))
            if beam.done == False:
                active.append(b)

        if len(active) == 0:
            break

        beam_idx = T.LongTensor(active)
        n_rem = len(beam_idx)

    predicted_words = []
    for beam in beams:
        predicted_words.append(beam.get_best())

    print('predicted_words',predicted_words)
    print('---------------------------')
    return predicted_words