import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from utils.transformer.decoder import TransformerDecoder
from utils.transformer.neural import MultiHeadedAttention

import copy
import random
import torch.nn.functional as F
from utils.bert.train_util import *


torch.manual_seed(666)
random.seed(666)
torch.backends.cudnn.deterministic = True

class WordProbLayer(nn.Module):
    def __init__(self, config, vocab_size, dec_hidden_size, init_embeddings, copy = False, dropout=0.1):
        super(WordProbLayer, self).__init__()
        # self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.config = config
        # self.copy = copy
        # self.coverage = coverage
        self.dropout = dropout        
        self.LogScore = nn.LogSoftmax(dim=-1)
        
        self.copy = copy

        if config.copy:
            self.proj = nn.Linear(dec_hidden_size*3, vocab_size)
            self.prob_copy = nn.Linear(dec_hidden_size * 3, 1, bias=True)
        else:
            self.proj = nn.Linear(dec_hidden_size, vocab_size)
            self.proj.weight = init_embeddings.weight

    def forward(self, x, emb=None, dec_attn_weights=None, src_enc=None, extra_zeros=None, tokens=None, is_predict =False):  
        """
            x: final hidden layer output by decoder [ b_sz * seqlen * hidden ]
            memory: output by encoder               
            emb: word embedd for current token...
            
            returns: softmaxed probabilities, copy attention distribs
        """     
        if self.copy:
            '''make src dist'''
            attn_dist = T.sum(dec_attn_weights, dim=1) # source attn weight
            atts = T.bmm(attn_dist, src_enc) # source context dist (h_t*)
            if is_predict:
                atts = atts.transpose(0,1).squeeze(0)


            '''make target dist'''
            pred = self.LogScore(self.proj(T.cat([x, emb, atts], -1)))        #原词典上的概率分布

            if extra_zeros is not None:
                if not is_predict:
                    cat_extra_zeros = extra_zeros.repeat(1,pred.size(1),1)          # 沿着指定的维度[pred.size(0),1,1]重复张量
                else:
                    cat_extra_zeros = extra_zeros
                pred = T.cat((pred, cat_extra_zeros), -1)                       # 更新詞彙表分布
            # -----------------------------------------------------------------------
            p_gen = T.sigmoid(self.prob_copy(T.cat([x, emb, atts], -1)))
            if not is_predict:
                tokens = tokens.unsqueeze(0).repeat(pred.size(1),1,1).transpose(0,1)
                pass
                pred = (p_gen * pred).scatter_add(2, tokens, attn_dist)
            else:
                attn_dist = attn_dist.transpose(0,1).squeeze(0)
                pred = (p_gen * pred).scatter_add(1, tokens, attn_dist)

            '''
            target = self.scatter_add_(dim, index, other) → Tensor
            将张量other所有值加到index张量中指定的index处的self中. 对于中的每个值other ，
            它被添加到索引中的self其通过它的索引中指定的other用于dimension != dim ，并通过在相应的值index为dimension = dim
            
            使用CUDA后端时，此操作可能会导致不确定的行为，不容易关闭. 请参阅有关可重现性的说明作为背景.

            REPRODUCIBILITY
            在PyTorch发行版，单独的提交或不同的平台上，不能保证完全可重复的结果. 此外，即使使用相同的种子，结果也不必在CPU和GPU执行之间再现.
            https://s0pytorch0org.icopy.site/docs/stable/notes/randomness.html
            '''
            return pred, attn_dist 
                    
        else:
            final_dist = self.LogScore(self.proj(x))
            attn_dist = None
            return final_dist, attn_dist 

    


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            # print('x',x.shape)
            # print('segs',segs.shape)
            # print('mask',mask.shape)
            top_vec, _ = self.model(x, segs, attention_mask=mask)            
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)                
        return top_vec

class AbsSummarizer(nn.Module):
    def __init__(self, config):
        super(AbsSummarizer, self).__init__()
        self.config = config
        self.encoder = Bert(large = False, temp_dir='../temp', 
                            finetune = config.finetune_bert)

        if (config.encoder == 'Transformer'):
            '''
              "attention_probs_dropout_prob": 0.1,
                "finetuning_task": null,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 768,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 512,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "num_labels": 2,
                "output_attentions": false,
                "output_hidden_states": false,
                "pruned_heads": {},
                "torchscript": false,
                "type_vocab_size": 2,
                "vocab_size": 30522
            '''            
            bert_config = BertConfig(self.encoder.model.config.vocab_size, hidden_size=config.enc_hidden_size,
                                     num_hidden_layers=config.enc_layers, num_attention_heads=config.enc_heads,
                                     intermediate_size= config.enc_ff_size,
                                     hidden_dropout_prob=config.enc_dropout,
                                     attention_probs_dropout_prob=config.enc_dropout)
            self.encoder.model = BertModel(bert_config)

        if(config.max_pos>512):
            my_pos_embeddings = nn.Embedding(config.max_pos, self.encoder.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.encoder.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.encoder.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(config.max_pos-512,1)
            self.encoder.model.embeddings.position_embeddings = my_pos_embeddings
            print(my_pos_embeddings.weight.data.shape)
        self.vocab_size = self.encoder.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.model.config.hidden_size, padding_idx=0)
        if (config.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.encoder.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            config.dec_layers,
            config.dec_hidden_size, heads=config.dec_heads,
            d_ff=config.dec_ff_size, dropout=config.dec_dropout, embeddings=tgt_embeddings)

        for module in self.decoder.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        if(config.use_bert_emb):
            tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.model.config.hidden_size, padding_idx=0)
            tgt_embeddings.weight = copy.deepcopy(self.encoder.model.embeddings.word_embeddings.weight)
            self.decoder.embeddings = tgt_embeddings

        self.word_prob = WordProbLayer(config, self.vocab_size, 
                                    config.dec_hidden_size, self.decoder.embeddings,
                                    copy = config.copy)
        for p in self.word_prob.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                p.data.zero_()
        
    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls, extra_zeros, src_extend_vocab):
        device = next(self.parameters()).device
        src = src.cuda(device)
        segs = segs.cuda(device)
        mask_src = mask_src.cuda(device)
        # ---------------------------------------------------------
        src_enc = self.encoder(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, src_enc)
        # ---------------------------------------------------------
        tgt = tgt.cuda(device)
        clss = clss.cuda(device)
        mask_tgt = mask_tgt.cuda(device)
        mask_cls = mask_cls.cuda(device)
        if extra_zeros:
            extra_zeros = extra_zeros.cuda(device)
        src_extend_vocab = src_extend_vocab.cuda(device)
        # ---------------------------------------------------------
        decoder_outputs, state, pure_emb, dec_attn_weights = self.decoder(tgt[:, :-1], src_enc, dec_state)


        '''
        # [bsz*seq_len , hid] -> [bsz*seq_len , vocab_size]
        # [bsz, seq_len , hid] -> [bsz, seq_len , vocab_size]
        '''
        if self.config.copy:
            pred, attn = self.word_prob(decoder_outputs, pure_emb, dec_attn_weights, 
            src_enc, extra_zeros, src_extend_vocab)
        else:
            pred, attn = self.word_prob(decoder_outputs) 
        
        return (pred, attn)




        
