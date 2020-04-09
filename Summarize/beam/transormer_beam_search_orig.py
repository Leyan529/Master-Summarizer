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
        # 初始beam score分數為-30
        self.scores = T.FloatTensor(config.beam_size,1).fill_(-30)      #beam_size,1; Initial score of beams = -30
        self.tokens, self.scores = get_cuda(self.tokens), get_cuda(self.scores)
        self.scores[0][0] = 0  
        
        # 每個batch中欲被decode的元素，將根據beam_size進行複製
        # self.coverage = coverage.unsqueeze(0).repeat(config.beam_size, 1) #beam_size, 2*hid_size
        # self.sum_temporal_srcs = None
        # self.prev_s = None
        self.done = False
        self.end_id = end_id
        self.unk_id = unk_id

    def get_current_state(self):
        tokens = self.tokens[:,-1].clone()
        for i in range(len(tokens)):
            if tokens[i].item() >= config.vocab_size: # 如果token id大於vocab_size，則將其置換為unk_id
                tokens[i] = self.unk_id
        return tokens


    def advance(self, prob_dist):
        '''Perform beam search: Considering the probabilites of given n_beam x n_extended_vocab words, select first n_beam words that give high total scores
        :param prob_dist: (beam, n_extended_vocab)
        :param hidden_state: Tuple of (beam, hid_size) tensors
        :param context:   (beam, 2*n_hidden)
        :param sum_temporal_srcs:   (beam, n_seq)
        :param prev_s:  (beam, t, hid_size)
        '''
        n_extended_vocab = prob_dist.size(1)
        # 將機率轉化為log score
        log_probs = T.log(prob_dist+config.eps)                         #beam_size, n_extended_vocab # log_probs(16,36333)
        # 重新計算一個新的分數
        scores = log_probs + self.scores                                #[beam_size, n_extended_vocab=36333]
        # 將每一個分數展開回一維的tensor
        scores = scores.view(-1,1)                                      #beam_size*n_extended_vocab, 1
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


def beam_search(config, batch, model, start_id, end_id, unk_id):

    batch_size = config.batch_size
    beam_idx = T.LongTensor(list(range(batch_size)))
    # print('batch enc_hid',enc_hid[0][0].shape);    
    # print('batch ct_e',ct_e.shape);  
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, \
    coverage_t_0, _, _, _, _= get_input_from_batch(batch, config, batch_first = False)

    encoder_outputs, padding_mask = model.encode(enc_batch, enc_padding_mask)
    coverage=(coverage_t_0[0] if config.coverage else None)

    # decoder batch preparation, it has beam_size example initially everything is repeated
    beams = [Beam(start_id, end_id, unk_id, coverage) for i in range(batch_size)]   #For each example in batch, create Beam object
    
    n_rem = batch_size                                                  #Index of beams that are active, i.e: didn't generate [STOP] yet

    # print('encoder_outputs',encoder_outputs.shape)
    for t in range(config.max_dec_steps):
        # print('---------------------------------------')
        # 將batch中每個beam的元素在第一維度疊加
        hyp_tokens = T.stack(
            [beam.get_current_state() for beam in beams if beam.done == False]      #remaining(rem),beam
        )
        # print(hyp_tokens.shape)
        # hyp_tokens = get_cuda(T.tensor([h.tokens for h in beams])).transpose(0,1) # NOT batch first
        # hyp_tokens.masked_fill_(hyp_tokens>=config.vocab_size, unk_id)# convert oov to unk
        
        # print('beam_x_t',x_t.shape)
        # x_t = model.embeds(x_t)                                                 #rem*beam, n_emb
        
        # ct_e = T.stack(
        #     [beam.context for beam in beams if beam.done == False]                  #rem,beam,hid_size
        # ).contiguous().view(-1,2*config.hidden_dim)                                 #rem,beam,hid_size



        # s_t = (dec_h, dec_c)
        # enc_out_beam = enc_out[beam_idx].view(n_rem,-1).repeat(1, config.beam_size).view(-1, enc_out.size(1), enc_out.size(2))
        # enc_pad_mask_beam = enc_padding_mask[beam_idx].repeat(1, config.beam_size).view(-1, enc_padding_mask.size(1))

        # extra_zeros_beam = None
        # if extra_zeros is not None:
        #     extra_zeros_beam = extra_zeros[beam_idx].repeat(1, config.beam_size).view(-1, extra_zeros.size(1))
        # enc_extend_vocab_beam = enc_batch_extend_vocab[beam_idx].repeat(1, config.beam_size).view(-1, enc_batch_extend_vocab.size(1))

        pred, attn = model.decode(hyp_tokens, encoder_outputs, padding_mask, None, \
                                    enc_batch_extend_vocab, extra_zeros)

        # beam search 在decode時找出final dist機率最大的beam size個單詞
        # final_dist, (dec_h, dec_c), ct_e, sum_temporal_srcs, prev_s = model.decoder(
        #     x_t, s_t, enc_out_beam, enc_pad_mask_beam, ct_e, 
        #     extra_zeros_beam, enc_extend_vocab_beam, sum_temporal_srcs, prev_s, 
        #     enc_key_batch, enc_key_lens)              #final_dist: rem*beam, n_extended_vocab

        pred = pred.view(n_rem, config.beam_size, -1)                   #final_dist: rem, beam, n_extended_vocab
        # dec_h = dec_h.view(n_rem, config.beam_size, -1)                             #rem, beam, hid_size
        # dec_c = dec_c.view(n_rem, config.beam_size, -1)                             #rem, beam, hid_size
        # ct_e = ct_e.view(n_rem, config.beam_size, -1)                             #rem, beam, 2*hid_size

        # if sum_temporal_srcs is not None:
        #     sum_temporal_srcs = sum_temporal_srcs.view(n_rem, config.beam_size, -1) #rem, beam, n_seq

        # if prev_s is not None:
        #     prev_s = prev_s.view(n_rem, config.beam_size, -1, config.hidden_dim)    #rem, beam, t

        # For all the active beams, perform beam search
        active = []         #indices of active beams after beam search

        for i in range(n_rem):
            b = beam_idx[i].item()
            beam = beams[b]
            if beam.done:
                continue

            # sum_temporal_srcs_i = prev_s_i = None
            # if sum_temporal_srcs is not None:
            #     sum_temporal_srcs_i = sum_temporal_srcs[i]                              #beam_size, n_seq
            # if prev_s is not None:
            #     prev_s_i = prev_s[i]                                                #beam_size, t, hid_size
            beam.advance(pred[i])
            if beam.done == False:
                active.append(b)

        if len(active) == 0:
            break

        beam_idx = T.LongTensor(active)
        n_rem = len(beam_idx)

    predicted_words = []
    for beam in beams:
        predicted_words.append(beam.get_best())

    return predicted_words