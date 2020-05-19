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
# from utils.bert.train_util import get_cuda

import torch.distributed


def choose_criterion(config, vocab_size):
    if config.label_smoothing > 0:
        criterion = LabelSmoothingLoss(
                        config.label_smoothing, vocab_size, ignore_index=0
                    )
    else:
        # criterion = nn.NLLLoss(ignore_index=0, reduction='sum')
        # criterion = nn.NLLLoss(ignore_index=0, reduction='sum',reduce =False)
        criterion = NLLLoss(ignore_index=0)
    # criterion = get_cuda(criterion)
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
            self.size = tgt_vocab_size
            self.smoothing_value = label_smoothing / (tgt_vocab_size - 2)
            one_hot = torch.full((tgt_vocab_size,), self.smoothing_value)
            one_hot[self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot.unsqueeze(0))
            self.confidence = 1.0 - label_smoothing

        def forward(self, output, target):
            """
            output (FloatTensor): batch_size x n_classes
            target (LongTensor): batch_size
            """

            """
            支持扩展词典, 比如copy机制使用的src词典
            input size: b_sz*seq_en, vocab
            return: 0维tensor
            """
            real_size = output.size(2)  
            if real_size > self.size:
                real_size -= self.size  # real size即扩展词典的大小
            else:
                real_size = 0  

            target = target.contiguous().view(-1)
            output = output.view(-1, output.size(2)) 

            model_prob = self.one_hot.repeat(target.size(0), 1)

            if real_size > 0: 
                ext_zeros = get_cuda(torch.full((model_prob.size(0), real_size), self.smoothing_value))
                model_prob = torch.cat((model_prob, ext_zeros), -1)

            model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
            model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)
            # print('output',output.shape)
            # print('model_prob',model_prob.shape)
            return F.kl_div(output, model_prob, reduction='sum')

class NLLLoss(nn.Module):
        """
        With label smoothing,
        KL-divergence between q_{smoothed ground truth prob.}(w)
        and p_{prob. computed by model}(w) is minimized.
        """
        def __init__(self, ignore_index):
            super(NLLLoss, self).__init__()
            self.NLL = nn.NLLLoss(ignore_index=ignore_index, reduction='sum')

        def forward(self, output, target):
            target = target.contiguous().view(-1)
            output = output.contiguous().view(target.size(0),-1)
            return self.NLL(output, target)

def _bottle(_v):
    return _v.view(-1, _v.size(2))

def compute_correct(scores, target, num_tokens, tokenizer):
    # pred_words = scores.max(1)[1] # 從維度1選擇機率最高的
    pred_words = scores.max(2)[1] # 從維度1選擇機率最高的
    pred_words = pred_words.contiguous().view(-1)
    target = target.contiguous().view(-1)
    non_padding = target.ne(0)
    num_correct = pred_words.eq(target) \
                        .masked_select(non_padding) \
                        .sum() \
                        .item()
    # if num_correct == num_tokens:    
    # print('-----------------------------------')
    # print('non_padding',non_padding)
    # print('pred_words',pred_words)
    # print('target',target)
    
    return num_correct

# def compute_loss(model, criterion, pred, target, num_tokens, tokenizer):
#     gtruth = target   
#     loss = criterion(pred, gtruth) 
#     num_correct = compute_correct(pred, target, num_tokens, tokenizer)  
#     return loss, num_correct, target

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