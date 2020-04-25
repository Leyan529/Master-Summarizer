from rouge import Rouge
from utils.seq2seq.train_util import get_cuda
import torch as T

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
                score = [{"rouge-l":{"f":0.0}}]
            scores.append(score[0])
    rouge_l_f1 = [score["rouge-l"]["f"] for score in scores]
    rouge_l_f1 = get_cuda(T.FloatTensor(rouge_l_f1))
    return rouge_l_f1