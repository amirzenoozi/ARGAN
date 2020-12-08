import numpy as np

class EarlyStopping():
    
    def __init__(self, min_delta = 0, patience = 0):
        
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        self.stop_training = False
    
    def on_epoch_end(self, epoch, current_value):
        if np.less(self.best, (current_value - self.min_delta)):
            self.best = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
        return self.stop_training