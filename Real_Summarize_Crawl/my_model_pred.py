import os
import torch
import torch as T
from torch import nn

def loadModel(load_model_path, model):    
    # # checkpoint = T.load(load_model_path, map_location='cpu')
    # T.backends.cudnn.benchmark = True 
    # print(load_model_path)
    # checkpoint = T.load(load_model_path)
    # model.load_state_dict(checkpoint['model'])
    # step = checkpoint['step']
    # # vocab = checkpoint['vocab']
    # loss = checkpoint['loss']
    # r_loss = checkpoint['r_loss']
    # # optimizer.load_state_dict(checkpoint['optimizer'])
    # print("Loaded model at " + load_model_path)
    # print("Loaded model step = %s, loss = %.2f, r_loss = %.2f " %(step, loss, r_loss))
    
    model = torch.load('Pointer_generator_word2Vec.pkl')
    return model


