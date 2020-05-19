#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math

import torch
import torch.nn as nn
from translate.beam import GNMTGlobalScorer
from utils.bert.train_util import get_cuda


def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')

    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger

        self.args = args
        self.model = model
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        self.block_trigram = args.block_trigram

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        # tensorboard_log_dir = args.model_path

        # self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.dec_tgt.size(0)

        preds, pred_score, gold_score, tgt_str, src =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"],batch.original_abstract, batch.src

        translations = []
        # vocab convert_ids_to_tokens
        for b in range(batch_size):
            pred_sents_list = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
            pred_sents = ' '.join(pred_sents_list).replace(' ##','')
            gold_sent = ' '.join(tgt_str[b].split())
            
            raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
            raw_src = ' '.join(raw_src)
            pred_sents = pred_sents.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()

            translation = (pred_sents, gold_sent, raw_src)
            # print('src :',raw_src)
            # print('gold_sent :',gold_sent)
            # print('pred_sents :',pred_sents)
            # print('--------------------------------------------------')
            translations.append(translation)

        return translations

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch,
                self.max_length,
                min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam
        encoder = nn.DataParallel(self.model.encoder, device_ids=self.args.device_ids)
        

        beam_size = self.beam_size
        
        batch_size = batch.src.size(0)
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src
        extra_zeros = get_cuda(torch.zeros((batch_size, 1, batch.max_art_oovs), requires_grad=False))
        src_extend_vocab = get_cuda(batch.art_batch_extend_vocab)

        # src_features = self.model.encoder(src, segs, mask_src)
        # dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        
        src_features = encoder(src, segs, mask_src)
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        decoder = nn.DataParallel(self.model.decoder, device_ids=self.args.device_ids)

        src_features = get_cuda(src_features)

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(src_features, beam_size, dim=0)
        extra_zeros = tile(extra_zeros, beam_size, dim=0)
        src_extend_vocab = tile(src_extend_vocab, beam_size, dim=0)

        batch_offset = get_cuda(torch.arange(
            batch_size, dtype=torch.long))
        beam_offset = get_cuda(torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long))

        alive_seq = get_cuda(torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long))

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            get_cuda(torch.tensor([0.0] + [float("-inf")] * (beam_size - 1)).repeat(batch_size)))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            # print('step',step)
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0,1)
            # --------------------------------------------------------------------------------
            # dec_out, dec_states, dec_emb, last_dec_mid = self.model.decoder(decoder_input, src_features, dec_states,
            #                                          step=step)
            dec_out, dec_states, dec_emb, last_dec_mid = decoder(decoder_input, src_features, dec_states,
                                                     step=step)                                                     
            # log_probs, check_attn = self.model.word_prob(dec_out.transpose(0,1).squeeze(0),
            #     dec_emb.transpose(0,1).squeeze(0), last_dec_mid, 
            #     src_features, extra_zeros.transpose(0,1).squeeze(0),
            #     src_extend_vocab, is_predict = True
            # )
            word_prob = nn.DataParallel(self.model.word_prob, device_ids=self.args.device_ids)
            log_probs, check_attn = word_prob(dec_out.transpose(0,1).squeeze(0),
                dec_emb.transpose(0,1).squeeze(0), last_dec_mid, 
                src_features, extra_zeros.transpose(0,1).squeeze(0),
                src_extend_vocab, is_predict = True
            )
            # --------------------------------------------------------------------------------
            # Generator forward.
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if(self.args.block_trigram):
                cur_len = alive_seq.size(1)
                if(cur_len>3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = [self.vocab.ids_to_tokens[w] for w in words]
                        words = ' '.join(words).replace(' ##','').split()
                        if(len(words)<=3):
                            continue
                        trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results

