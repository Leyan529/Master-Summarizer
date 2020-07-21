from rouge import Rouge
from utils.seq2seq.train_util import get_cuda
import torch as T
from utils.seq2seq.data import output2words
from utils.seq2seq import data

def reward_function(decoded_sents, original_sents):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(decoded_sents, original_sents)
    except Exception:
        # print("Rouge failed for multi sentence evaluation.. Finding exact pair")
        scores = []
        for i in range(len(decoded_sents)):
            try:
                score = rouge.get_scores(decoded_sents[i], original_sents[i])
            except Exception:
                # print("Error occured at:")
                # print("decoded_sents:", decoded_sents[i])
                # print("original_sents:", original_sents[i])
                score = [{"rouge-l":{"r":0.0}}]
            scores.append(score[0])
    rewards = [score["rouge-l"]["r"] for score in scores]
    rewards = get_cuda(T.FloatTensor(rewards))
    return rewards


def to_sents(enc_out, inds, vocab, art_oovs):
    decoded_strs = []
    for i in range(len(enc_out)):
        id_list = inds[i].tolist() # 取出每個sample sentence 的word id list
        S = output2words(id_list, vocab, art_oovs[i]) #Generate sentence corresponding to sampled words
        try:
            end_idx = S.index(data.STOP_DECODING)
            S = S[:end_idx]
        except ValueError:
            S = S
        if len(S) < 2:          #If length of sentence is less than 2 words, replace it with "xxx"; Avoids setences like "." which throws error while calculating ROUGE
            S = ["xxx"]
        S = " ".join(S)
        decoded_strs.append(S)
    return decoded_strs