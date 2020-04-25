"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import math
import pickle
from utils.bert.train_util import get_cuda

import torch.distributed


def choose_criterion(config, vocab_size):
    if config.label_smoothing > 0:
        criterion = LabelSmoothingLoss(
                        config.label_smoothing, vocab_size, ignore_index=0
                    )
    else:
        criterion = nn.NLLLoss(ignore_index=0, reduction='sum')
    criterion = get_cuda(criterion)
    return criterion

class LabelSmoothingLoss(nn.Module):
        """
        With label smoothing,
        KL-divergence between q_{smoothed ground truth prob.}(w)
        and p_{prob. computed by model}(w) is minimized.
        """
        def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
            assert 0.0 < label_smoothing <= 1.0
            self.padding_idx = ignore_index
            super(LabelSmoothingLoss, self).__init__()

            smoothing_value = label_smoothing / (tgt_vocab_size - 2)
            one_hot = torch.full((tgt_vocab_size,), smoothing_value)
            one_hot[self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot.unsqueeze(0))
            self.confidence = 1.0 - label_smoothing

        def forward(self, output, target):
            """
            output (FloatTensor): batch_size x n_classes
            target (LongTensor): batch_size
            """
            model_prob = self.one_hot.repeat(target.size(0), 1)
            model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
            model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)
            # print('output',output.shape)
            # print('model_prob',model_prob.shape)
            return F.kl_div(output, model_prob, reduction='sum')

def _bottle(_v):
    return _v.view(-1, _v.size(2))

def compute_correct(scores, target):
    pred = scores.max(1)[1]
    target = target.contiguous().view(-1, 1)
    non_padding = target.ne(0)
    num_correct = pred.eq(target) \
                        .masked_select(non_padding) \
                        .sum() \
                        .item()
    return num_correct

def compute_loss(model, criterion, output, target):
    bottled_output = _bottle(output)
    scores = model.generator(bottled_output)
    gtruth = target.contiguous().view(-1)
    loss = criterion(scores, gtruth) 
    num_correct = compute_correct(scores, target)   
    return loss, num_correct, scores, target


def accuracy(n_correct, n_words):
    """ compute accuracy """
    if n_words == 1: return 0
    return 100 * (n_correct / n_words)

def xent(loss, n_words):
        """ compute cross entropy """
        return loss / n_words

def ppl(loss, n_words):
    """ compute perplexity """
    return math.exp(min(loss / n_words, 100))