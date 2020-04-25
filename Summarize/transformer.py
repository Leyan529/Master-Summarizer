import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from utils.transformer.decoder import TransformerDecoder

import copy
import random
import torch.nn.functional as F
from utils.bert.train_util import *


torch.manual_seed(666)
random.seed(666)
torch.backends.cudnn.deterministic = True

def get_generator(vocab_size, dec_hidden_size):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator = get_cuda(generator)
    return generator

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
        # self.config = config
        self.encoder = Bert(large = False, temp_dir='../temp', finetune = True)

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

        self.generator = get_generator(self.vocab_size, config.dec_hidden_size)
        self.generator[0].weight = self.decoder.embeddings.weight

        for module in self.decoder.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        for p in self.generator.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                p.data.zero_()
        if(config.use_bert_emb):
            tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.model.config.hidden_size, padding_idx=0)
            tgt_embeddings.weight = copy.deepcopy(self.encoder.model.embeddings.word_embeddings.weight)
            self.decoder.embeddings = tgt_embeddings
            self.generator[0].weight = self.decoder.embeddings.weight

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.encoder(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, state
