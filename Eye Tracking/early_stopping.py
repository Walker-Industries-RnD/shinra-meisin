class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.stop = False
        self.best = float('inf')
    def feed(self, val_loss):
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
    def reset(self):
        self.counter = 0
        self.stop = False
        self.best = float('inf')
