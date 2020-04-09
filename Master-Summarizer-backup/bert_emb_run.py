import warnings
warnings.filterwarnings("ignore")
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import time
import re


import torch as T



import torch.nn as nn
import torch.nn.functional as F


from model import Model


from data_util import config
from data_util import bert_data as data
from data_util.bert_batcher import Batcher
from data_util.bert_data import Vocab

from train_util import *
from torch.distributions import Categorical
from rouge import Rouge
from numpy import random
import argparse
import torchsnooper
import logging
transformers_logger = logging.getLogger("transformers.tokenization_utils")
transformers_logger.setLevel(logging.ERROR)
transformers_logger.disabled = True
from write_result import *


# -------- Test Packages -------
from bert_beam_search import *
import shutil
from tensorboardX import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu

# from pytorch_pretrained_bert import BertModel
from transformers import BertModel, BertTokenizer 
from transformers import TransfoXLTokenizer, TransfoXLModel, TransfoXLConfig

def train_action(opt,logger, writer, train_num):
    try:       
        opt.rl_weight = 1 - opt.mle_weight  

        if opt.load_model:
            opt.load_model = "/%s/%s"%(opt.word_emb_type,opt.load_model)    

        logger.info(u'------Training Setting--------')  

        logger.info("Traing Type :%s" %(config.data_type))
        if opt.train_mle == True:
            logger.info("Training mle: %s, mle weight: %.2f"%(opt.train_mle, opt.mle_weight))

        if opt.train_rl == True:
            logger.info("Training rl: %s, rl weight: %.2f \n"%(opt.train_rl, opt.rl_weight))

        # if opt.word_emb_type == 'bert': config.emb_dim = 768
        if opt.pre_train_emb : 
            logger.info('use pre_train_%s vocab_size %s \n'%(opt.word_emb_type,config.vocab_size))

        else:
            logger.info('use %s vocab_size %s \n'%(opt.word_emb_type,config.vocab_size))

        logger.info("intra_encoder: %s intra_decoder: %s \n"%(config.intra_encoder, config.intra_decoder))
        if opt.word_emb_type in ['word2Vec','glove']:
            config.vocab_path = config.Data_path + "Embedding/%s/word.vocab"%(opt.word_emb_type)            
            config.vocab_size = len(open(config.vocab_path).readlines())
        vocab = Vocab(config.vocab_path, config.vocab_size) # only by word pretrain vocab
        train_processor = BertEmbTrain(opt,vocab,logger, writer, train_num)
        train_processor.trainIters()
    except Exception as e:
        print(e)
        traceback = sys.exc_info()[2]
        logger.error(sys.exc_info())
        logger.error(traceback.tb_lineno)
        logger.error(e)
    logger.info(u'------Training END--------')  


class BertEmbTrain(object):
    def __init__(self, opt, vocab, logger, writer, train_num):
        self.vocab = vocab
        self.train_batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        self.test_batcher = Batcher(config.test_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)
        self.opt = opt
        self.start_id = self.vocab.word2id(data.START_DECODING)
        self.end_id = self.vocab.word2id(data.STOP_DECODING)
        self.pad_id = self.vocab.word2id(data.PAD_TOKEN)
        self.unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        self.logger = logger
        self.writer = writer
        self.train_num = train_num
        time.sleep(5)

    def save_model(self, iter, loss, r_loss):
        if not os.path.exists(config.save_model_path):
            os.makedirs(config.save_model_path)
        file_path = "/%07d_%.2f_%.2f.tar" % (iter, loss, r_loss)
        save_path = config.save_model_path + '/%s' % (self.opt.word_emb_type)
        if not os.path.isdir(save_path): os.mkdir(save_path)
        save_path = save_path + file_path
        T.save({
            "iter": iter + 1,
            "model_dict": self.model.state_dict(),
            "trainer_dict": self.trainer.state_dict()
        }, save_path)
        return file_path

    def setup_train(self):
        # BERT
        self.bert_model = get_cuda(BertModel.from_pretrained('bert-base-uncased'))
        #         self.bert_model = get_cuda(TransfoXLModel.from_pretrained('transfo-xl-wt103'))
        #         config = TransfoXLConfig()
        #         config.d_embed = 1025
        #         self.bert_model = get_cuda(TransfoXLModel(config)) # 更改參數以傳入TransfoXLModel
        self.bert_model.resize_token_embeddings(len(bert_data.bert_tokenizer))
        
        self.model = Model(self.opt.pre_train_emb, self.opt.word_emb_type, self.vocab)
        self.logger.info(str(self.model))
        self.model = get_cuda(self.model)
        device = T.device("cuda" if T.cuda.is_available() else "cpu")  # PyTorch v0.4.0
        if self.opt.multi_device:
            if T.cuda.device_count() > 1:
                #                 print("Let's use", T.cuda.device_count(), "GPUs!")
                self.logger.info("Let's use " + str(T.cuda.device_count()) + " GPUs!")
                self.model = nn.DataParallel(self.model, list(range(T.cuda.device_count()))).cuda()

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        self.model.to(device)
        #         self.model.eval()

        self.trainer = T.optim.Adam(self.model.parameters(), lr=config.lr)
        start_iter = 0
        if self.opt.load_model is not None:
            load_model_path = config.save_model_path + self.opt.load_model
            print(load_model_path)
            checkpoint = T.load(load_model_path)
            start_iter = checkpoint["iter"]
            self.model.load_state_dict(checkpoint["model_dict"])
            self.trainer.load_state_dict(checkpoint["trainer_dict"])
            #             print("Loaded model at " + load_model_path)
            self.logger.info("Loaded model at " + load_model_path)
        if self.opt.new_lr is not None:
            self.trainer = T.optim.Adam(self.model.parameters(), lr=self.opt.new_lr)
        return start_iter

    def train_batch_MLE(self, enc_out, enc_hidden, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, enc_key_batch, enc_key_lens, batch):
        ''' Calculate Negative Log Likelihood Loss for the given batch. In order to reduce exposure bias,
                pass the previous generated token as input with a probability of 0.25 instead of ground truth label
        Args:
        :param enc_out: Outputs of the encoder for all time steps (batch_size, length_input_sequence, 2*hidden_size)
        :param enc_hidden: Tuple containing final hidden state & cell state of encoder. Shape of h & c: (batch_size, hidden_size)
        :param enc_padding_mask: Mask for encoder input; Tensor of size (batch_size, length_input_sequence) with values of 0 for pad tokens & 1 for others
        :param ct_e: encoder context vector for time_step=0 (eq 5 in https://arxiv.org/pdf/1705.04304.pdf)
        :param extra_zeros: Tensor used to extend vocab distribution for pointer mechanism
        :param enc_batch_extend_vocab: Input batch that stores OOV ids
        :param batch: batch object
        '''
        dec_batch, max_dec_len, dec_lens, target_batch = get_dec_data(
            batch)  # Get input and target batchs for training decoder
        step_losses = []
        s_t = (enc_hidden[0], enc_hidden[1])  # Decoder hidden states
        x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(self.start_id))  # Input to the decoder
        prev_s = None  # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None  # Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        try:
            for t in range(min(max_dec_len, config.max_dec_steps)):
                use_gound_truth = get_cuda((T.rand(len(enc_out)) > 0.25)).long()  # Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
                x_t = use_gound_truth * dec_batch[:, t] + (1 - use_gound_truth) * x_t  # Select decoder input based on use_ground_truth probabilities
                x_t = self.model.embeds(x_t) 

                final_dist, s_t, ct_e, sum_temporal_srcs, prev_s = self.model.decoder(x_t, s_t, enc_out, enc_padding_mask,
                                                                                          ct_e, extra_zeros,
                                                                                          enc_batch_extend_vocab,
                                                                                          sum_temporal_srcs, prev_s, enc_key_batch, enc_key_lens)
                target = target_batch[:, t]
                log_probs = T.log(final_dist + config.eps)
                step_loss = F.nll_loss(log_probs, target, reduction="none", ignore_index=self.pad_id)
                step_losses.append(step_loss)
                x_t = T.multinomial(final_dist,1).squeeze()  # Sample words from final distribution which can be used as input in next time step

                is_oov = (x_t >= config.vocab_size).long()  # Mask indicating whether sampled word is OOV
                x_t = (1 - is_oov) * x_t.detach() + (is_oov) * self.unk_id  # Replace OOVs with [UNK] token
        except Exception as e:
            self.logger.error('xxxxxxxxxxx')
            traceback = sys.exc_info()[2]
            self.logger.error(sys.exc_info())
            self.logger.error(traceback.tb_lineno)
            self.logger.error(e)
            self.logger.error('xxxxxxxxxxx')

                
        losses = T.sum(T.stack(step_losses, 1), 1)  # unnormalized losses for each example in the batch; (batch_size)
        batch_avg_loss = losses / dec_lens  # Normalized losses; (batch_size)
        mle_loss = T.mean(batch_avg_loss)  # Average batch loss
        return mle_loss

    def train_batch_RL(self, enc_out, enc_hidden, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab,
                       review_oovs, greedy):
        '''Generate sentences from decoder entirely using sampled tokens as input. These sentences are used for ROUGE evaluation
        Args
        :param enc_out: Outputs of the encoder for all time steps (batch_size, length_input_sequence, 2*hidden_size)
        :param enc_hidden: Tuple containing final hidden state & cell state of encoder. Shape of h & c: (batch_size, hidden_size)
        :param enc_padding_mask: Mask for encoder input; Tensor of size (batch_size, length_input_sequence) with values of 0 for pad tokens & 1 for others
        :param ct_e: encoder context vector for time_step=0 (eq 5 in https://arxiv.org/pdf/1705.04304.pdf)
        :param extra_zeros: Tensor used to extend vocab distribution for pointer mechanism
        :param enc_batch_extend_vocab: Input batch that stores OOV ids
        :param review_oovs: Batch containing list of OOVs in each example
        :param greedy: If true, performs greedy based sampling, else performs multinomial sampling
        Returns:
        :decoded_strs: List of decoded sentences
        :log_probs: Log probabilities of sampled words
        '''
        s_t = enc_hidden  # Decoder hidden states
        x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(self.start_id))  # Input to the decoder
        prev_s = None  # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None  # Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        inds = []  # Stores sampled indices for each time step
        decoder_padding_mask = []  # Stores padding masks of generated samples
        log_probs = []  # Stores log probabilites of generated samples
        mask = get_cuda(T.LongTensor(len(enc_out)).fill_(
            1))  # Values that indicate whether [STOP] token has already been encountered; 1 => Not encountered, 0 otherwise

        for t in range(config.max_dec_steps):
            x_t = self.bert_model(x_t)[-2:][0]
            probs, s_t, ct_e, sum_temporal_srcs, prev_s = self.model.decoder(x_t, s_t, enc_out, enc_padding_mask, ct_e,
                                                                             extra_zeros, enc_batch_extend_vocab,
                                                                             sum_temporal_srcs, prev_s)
            if greedy is False:
                multi_dist = Categorical(probs)
                x_t = multi_dist.sample()  # perform multinomial sampling
                log_prob = multi_dist.log_prob(x_t)
                log_probs.append(log_prob)
            else:
                _, x_t = T.max(probs, dim=1)  # perform greedy sampling
            x_t = x_t.detach()
            inds.append(x_t)
            mask_t = get_cuda(T.zeros(len(enc_out)))  # Padding mask of batch for current time step
            mask_t[mask == 1] = 1  # If [STOP] is not encountered till previous time step, mask_t = 1 else mask_t = 0
            mask[(mask == 1) + (
            x_t == self.end_id) == 2] = 0  # If [STOP] is not encountered till previous time step and current word is [STOP], make mask = 0
            decoder_padding_mask.append(mask_t)
            is_oov = (x_t >= config.vocab_size).long()  # Mask indicating whether sampled word is OOV
            x_t = (1 - is_oov) * x_t + (is_oov) * self.unk_id  # Replace OOVs with [UNK] token

        inds = T.stack(inds, dim=1)
        decoder_padding_mask = T.stack(decoder_padding_mask, dim=1)
        if greedy is False:  # If multinomial based sampling, compute log probabilites of sampled words
            log_probs = T.stack(log_probs, dim=1)
            log_probs = log_probs * decoder_padding_mask  # Not considering sampled words with padding mask = 0
            lens = T.sum(decoder_padding_mask, dim=1)  # Length of sampled sentence
            log_probs = T.sum(log_probs,
                              dim=1) / lens  # (bs,)                                     #compute normalizied log probability of a sentence
        decoded_strs = []
        for i in range(len(enc_out)):
            id_list = inds[i].cpu().numpy()
            oovs = review_oovs[i]
            S = data.outputids2words(id_list, self.vocab, oovs)  # Generate sentence corresponding to sampled words
            try:
                end_idx = S.index(data.STOP_DECODING)
                S = S[:end_idx]
            except ValueError:
                S = S
            if len(S) < 2:  # If length of sentence is less than 2 words, replace it with "xxx"; Avoids setences like "." which throws error while calculating ROUGE
                S = ["xxx"]
            S = " ".join(S)
            decoded_strs.append(S)

        return decoded_strs, log_probs 

    def reward_function(self, decoded_sents, original_sents):
        rouge = Rouge()
        try:
            scores = rouge.get_scores(decoded_sents, original_sents)
        except Exception:
            #             print("Rouge failed for multi sentence evaluation.. Finding exact pair")
            self.logger.info("Rouge failed for multi sentence evaluation.. Finding exact pair")
            scores = []
            for i in range(len(decoded_sents)):
                try:
                    score = rouge.get_scores(decoded_sents[i], original_sents[i])
                except Exception:
                    #                     print("Error occured at:")
                    #                     print("decoded_sents:", decoded_sents[i])
                    #                     print("original_sents:", original_sents[i])
                    self.logger.info("Error occured at:")
                    self.logger.info("decoded_sents:", decoded_sents[i])
                    self.logger.info("original_sents:", original_sents[i])
                    score = [{"rouge-l": {"f": 0.0}}]
                scores.append(score[0])
        rouge_l_f1 = [score["rouge-l"]["f"] for score in scores]
        avg_rouge_l_f1 = sum(rouge_l_f1) / len(rouge_l_f1)
        rouge_l_f1 = get_cuda(T.FloatTensor(rouge_l_f1))
        return rouge_l_f1, scores, avg_rouge_l_f1

    # def write_to_file(self, decoded, max, original, sample_r, baseline_r, iter):
    #     with open("temp.txt", "w") as f:
    #         f.write("iter:"+str(iter)+"\n")
    #         for i in range(len(original)):
    #             f.write("dec: "+decoded[i]+"\n")
    #             f.write("max: "+max[i]+"\n")
    #             f.write("org: "+original[i]+"\n")
    #             f.write("Sample_R: %.4f, Baseline_R: %.4f\n\n"%(sample_r[i].item(), baseline_r[i].item()))


    def train_one_batch(self, batch,test_batch, iter):
        ans_list, batch_scores = None, None
        # Train
        enc_batch, enc_lens, enc_padding_mask, \
        enc_key_batch, enc_key_lens, enc_key_padding_mask,\
        enc_batch_extend_vocab, extra_zeros, context = get_enc_data(batch)     
        
        enc_batch = enc_batch.type(T.LongTensor).cuda() #  `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
        enc_key_batch = enc_key_batch.type(T.LongTensor).cuda() #  `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
        enc_padding_mask = enc_padding_mask.type(T.LongTensor).cuda() #  `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
        enc_key_padding_mask = enc_key_padding_mask.type(T.LongTensor).cuda() #  `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
        
        
        # enc_padding_mask  `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length]
        # `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length]

        enc_batch = self.bert_model(enc_batch, attention_mask = enc_padding_mask)[-2:][0] 
        enc_key_batch = self.bert_model(enc_key_batch, attention_mask = enc_key_padding_mask)[-2:][0]  
        
        # Get embeddings for encoder input
        # Get key embeddings for encoder input 
        enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)
        
        # Test
        enc_batch2, enc_lens2, enc_padding_mask2, \
        enc_key_batch2, enc_key_lens2, enc_key_padding_mask2,\
        enc_batch_extend_vocab2, extra_zeros2, context2 = get_enc_data(test_batch)
        
        enc_batch2 = enc_batch2.type(T.LongTensor).cuda() #  `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
        enc_key_batch2 = enc_key_batch2.type(T.LongTensor).cuda() #  `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
        enc_padding_mask2 = enc_padding_mask2.type(T.LongTensor).cuda() #  `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
        enc_key_padding_mask2 = enc_key_padding_mask2.type(T.LongTensor).cuda() #  `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
        
        
        with T.autograd.no_grad():
      
            enc_batch2 = self.bert_model(enc_batch2, attention_mask = enc_padding_mask2)[-2:][0]
            enc_key_batch2 = self.bert_model(enc_key_batch2, attention_mask = enc_key_padding_mask2)[-2:][0]  
                
            enc_out2, enc_hidden2 = self.model.encoder(enc_batch2, enc_lens2)
        # -------------------------------Summarization-----------------------
        if self.opt.train_mle == True:  # perform MLE training
            mle_loss = self.train_batch_MLE(enc_out, enc_hidden, enc_padding_mask, context, extra_zeros,
                                            enc_batch_extend_vocab, enc_key_batch, enc_key_lens, batch)
            mle_loss_2 = self.train_batch_MLE(enc_out2, enc_hidden2, enc_padding_mask2, context2, extra_zeros2,
                                            enc_batch_extend_vocab2, enc_key_batch2, enc_key_lens2, test_batch)
        else:
            mle_loss = get_cuda(T.FloatTensor([0]))
            mle_loss_2 = get_cuda(T.FloatTensor([0]))
            
        # --------------RL training-----------------------------------------------------
        if self.opt.train_rl == True:  # perform reinforcement learning training
            # multinomial sampling
            sample_sents, RL_log_probs = self.train_batch_RL(enc_out, enc_hidden, enc_padding_mask, context,
                                                             extra_zeros, enc_batch_extend_vocab, batch.rev_oovs,
                                                             greedy=False)
            sample_sents2, RL_log_probs2 = self.train_batch_RL(enc_out2, enc_hidden2, enc_padding_mask2, context2,
                                                             extra_zeros2, enc_batch_extend_vocab2, test_batch.rev_oovs,
                                                             greedy=False)
            with T.autograd.no_grad():
                # greedy sampling
                greedy_sents, _ = self.train_batch_RL(enc_out, enc_hidden, enc_padding_mask, context, extra_zeros,
                                                      enc_batch_extend_vocab, batch.rev_oovs, greedy=True)

            sample_reward, _, _ = self.reward_function(sample_sents, batch.original_summarys)
            baseline_reward, _, _ = self.reward_function(greedy_sents, batch.original_summarys)
            # if iter%200 == 0:
            #     self.write_to_file(sample_sents, greedy_sents, batch.original_abstracts, sample_reward, baseline_reward, iter)
            rl_loss = -(sample_reward - baseline_reward) * RL_log_probs  # Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
            rl_loss = T.mean(rl_loss)

            batch_reward = T.mean(sample_reward).item()
            self.writer.add_scalar('Train_RL/RL_log_probs', RL_log_probs, iter)
        else:
            rl_loss = get_cuda(T.FloatTensor([0]))
            batch_reward = 0
        # ------------------------------------------------------------------------------------
        #         if opt.train_mle == True: 
        self.trainer.zero_grad()
        (self.opt.mle_weight * mle_loss + self.opt.rl_weight * rl_loss).backward()
        self.trainer.step()
        #-----------------------Summarization----------------------------------------------------
        if iter % 1000 == 0:
            with T.autograd.no_grad():
                train_rouge_l_f = self.calc_avg_rouge_result(iter,batch,'Train',enc_hidden, enc_out, enc_padding_mask, context, extra_zeros, enc_batch_extend_vocab, enc_key_batch, enc_key_lens)
                test_rouge_l_f = self.calc_avg_rouge_result(iter,test_batch,'Test',enc_hidden2, enc_out2, enc_padding_mask2, context2, extra_zeros2, enc_batch_extend_vocab2, enc_key_batch2, enc_key_lens2)
                self.writer.add_scalars('Compare/rouge-l-f',  
                   {'train_rouge_l_f': train_rouge_l_f,
                    'test_rouge_l_f': test_rouge_l_f
                   }, iter)
                self.logger.info('iter: %s train_rouge_l_f: %.3f test_rouge_l_f: %.3f \n' % (iter, train_rouge_l_f, test_rouge_l_f))
                
        return mle_loss.item(),mle_loss_2.item(), batch_reward

    def calc_avg_rouge_result(self, iter, batch, mode, enc_hidden, enc_out, enc_padding_mask, context, extra_zeros, enc_batch_extend_vocab, enc_key_batch, enc_key_lens):
        pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, context, extra_zeros, enc_batch_extend_vocab, enc_key_batch, enc_key_lens, self.model, self.start_id, self.end_id, self.unk_id)

        article_sents, decoded_sents, keywords_list, \
        ref_sents, long_seq_index = prepare_result(data, self.vocab, batch, mode, pred_ids)
        
        rouge_l = write_rouge(self.writer,iter,mode,article_sents, decoded_sents, \
                    keywords_list, ref_sents, long_seq_index)
        
        write_bleu(self.writer,iter, mode, article_sents, decoded_sents, \
                   keywords_list, ref_sents, long_seq_index)
        
        write_group(self.writer,iter,mode,article_sents, decoded_sents,\
                    keywords_list, ref_sents, long_seq_index)

        return rouge_l

    def get_best_res_score(self, results, scores):
        max_score = float(0)
        _id = 0
        for idx in range(len(results)):
            re_matchData = re.compile(r'\-?\d{1,10}\.?\d{1,10}')
            data = re.findall(re_matchData, str(scores[idx]))
            score = sum([float(d) for d in data])
            if score > max_score:
                _id = idx
        return results[_id], scores[_id]

    def get_lr(self):
        for param_group in self.trainer.param_groups:
            return param_group['lr']

    def get_weight_decay(self):
        for param_group in self.trainer.param_groups:
            return param_group['weight_decay']

    def trainIters(self):
        final_file_path = None
        iter = self.setup_train()
        epoch = 0
        count = test_mle_total = train_mle_total = r_total = 0
        self.logger.info(u'------Training START--------')
        test_batch = self.test_batcher.next_batch()
        #         while iter <= config.max_iterations:
        while epoch <= config.max_epochs:
            train_batch = self.train_batcher.next_batch()
            try:
                train_mle_loss,test_mle_loss, r  = self.train_one_batch(train_batch,test_batch, iter)

                self.writer.add_scalar('RL_Train/reward', r, iter)

                self.writer.add_scalars('Compare/mle_loss' ,  
                   {'train_mle_loss': train_mle_loss,
                    'test_mle_loss': test_mle_loss
                   }, iter)
                
            except KeyboardInterrupt:
                self.logger.info("-------------------Keyboard Interrupt------------------")
                exit(0)
            except Exception as e:                
                self.logger.info("-------------------Ignore error------------------\n%s\n" % e)
                print("Please load final_file_path : %s" % final_file_path)
                traceback = sys.exc_info()[2]
                print(sys.exc_info())
                print(traceback.tb_lineno)
                print(e)
                break
            # if opt.train_mle == False: break
            train_mle_total += train_mle_loss
            r_total += r
            test_mle_total += test_mle_loss
            count += 1
            iter += 1

            if iter % 1000 == 0:
                train_mle_avg = train_mle_total / count
                r_avg = r_total / count
                test_mle_avg = test_mle_total / count
                epoch = int((iter * config.batch_size) / self.train_num) + 1
                # self.logger.info('epoch: %s iter: %s train_mle_loss: %.3f test_mle_loss: %.3f reward: %.3f \n' % (epoch, iter, train_mle_avg, test_mle_avg, r_avg))
                

                count = test_mle_total = train_mle_total = r_total = 0
                self.writer.add_scalar('RL_Train/r_avg', r_avg, iter)
                
                self.writer.add_scalars('Compare/mle_avg_loss' ,  
                   {'train_mle_avg': train_mle_avg,
                    'test_mle_avg': test_mle_avg
                   }, iter)
            # break
            if iter % 5000 == 0:
                final_file_path = self.save_model(iter, test_mle_avg, r_avg)
                # if self.opt.view:
                #     best_res, best_score = self.get_best_res_score(ans_list, batch_scores)
                #     self.logger.info('best_res: %s \n' % (best_res))
                #     self.logger.info('best_score: %s \n' % (best_score))
                #     self.writer.add_text('Train/%s' % (iter), best_res['decoded_str'], iter)
                #     self.writer.add_text('Train/%s' % (iter), best_res['summary'], iter)
                #     self.writer.add_text('Train/%s' % (iter), best_res['review'], iter)                


