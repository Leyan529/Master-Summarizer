import numpy as np
import torch as T
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, config, logger, vocab, title, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.config = config
        self.logger = logger 
        self.vocab = vocab
        self.title = title 
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.step = 0

    def __call__(self, model, optimizer, step, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.save_model(model, optimizer, step, val_loss)
            self.counter = 0

    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), 'checkpoint.pt')
    #     self.val_loss_min = val_loss

    def save_model(self, model, optimizer, step, loss, r_loss=0):
        file_path = "/%07d.tar" % (step)
        save_path = self.config.save_model_path + '/%s/best' % (self.title)
        if not os.path.exists(save_path): os.makedirs(save_path)
        # 保存和加载整个模型
        T.save(model.module, save_path + '/model.pt')
        
        save_path = save_path + file_path
        state = {
            # 'model': model.state_dict(),
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'vocab': self.vocab,
            'loss':loss,
            'r_loss':r_loss
        }
        self.logger.info('Saving best model step %d to %s...'%(step, save_path))
        T.save(state, save_path)

        