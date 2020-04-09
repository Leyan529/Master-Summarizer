import numpy as np
import torch as T
# from utils import config
from utils.train_util import *

## Crimson Resolve
alpha = 0.9
beta = 5.

class Beam(object):
    def __init__(self, tokens, log_probs, state=None, coverage=None):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.coverage = coverage

    def extend(self, token, log_prob, state = None, coverage = None):
        return Beam(tokens = self.tokens + [token],
                        log_probs = self.log_probs + [log_prob],
                        state = state,
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
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, \
    coverage_t_0, _, _, _, _= get_input_from_batch(batch, config, batch_first = False)

    encoder_outputs, padding_mask = model.encode(enc_batch, enc_padding_mask)

    # decoder batch preparation, it has beam_size example initially everything is repeated
    beams = [Beam(tokens=[start_id],
                    log_probs=[0.0],
                    coverage=(coverage_t_0[0] if config.coverage else None)) 
                    for _ in range(config.beam_size)]
    results = []
    steps = 0
    while steps < config.max_dec_steps and len(results) < config.beam_size:
        hyp_tokens = get_cuda(T.tensor([h.tokens for h in beams])).transpose(0,1) # NOT batch first
        hyp_tokens.masked_fill_(hyp_tokens>=config.vocab_size, unk_id)# convert oov to unk
        
        # print('tgt',hyp_tokens.shape)
        # print('src_enc',encoder_outputs.shape)
        # print('src_padding_mask',padding_mask.shape)
        # print('tgt_padding_mask',None)
        # print('src_extend_vocab',enc_batch_extend_vocab.shape)
        # print('extra_zeros',extra_zeros)
        print(hyp_tokens.shape)
        pred, attn = model.decode(hyp_tokens, encoder_outputs, padding_mask, None,
                                    enc_batch_extend_vocab, extra_zeros)
        

        # pred, attn = self.decode(tgt, src_enc, src_padding_mask, tgt_padding_mask,
        # src_extend_vocab, extra_zeros)

        # gather attention at current step
        # attn = attn[-1,:,:]  # attn: [bsz * src_len]
        # print(attn.size())
        log_probs = T.log(pred[-1,:,:])         # get probs for next token
        topk_log_probs, topk_ids = T.topk(log_probs, config.beam_size * 2)  # avoid all <end> tokens in top-k
        # print(topk_ids)
        # print(topk_log_probs)
        all_beams = []
        num_orig_beams = 1 if steps == 0 else len(beams)
        for i in range(num_orig_beams):
            h = beams[i]
            # here save states, context vec and coverage...
            for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                new_beam = h.extend(token=topk_ids[i, j].item(),
                                log_prob=topk_log_probs[i, j].item(),
                                coverage=attn[i] if config.coverage else None)
                all_beams.append(new_beam)

        beams = []
        for h in sort_beams(all_beams):
            if h.latest_token == end_id:
                if steps >= config.min_dec_steps:
                    results.append(h)
            else: beams.append(h)
            # if len(beams) == config['beam_size'] or len(results) == config['beam_size']:
            #     break
        steps += 1

    if len(results) == 0:
        results = beams

    beams_sorted = sort_hypos(results)

    return beams_sorted[0]