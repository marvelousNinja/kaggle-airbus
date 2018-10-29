from airbus.callbacks.callback import Callback

class LRSchedule(Callback):
    def __init__(self, optimizer, epoch_lr_pairs, logger):
        self.epoch_counter = 0
        self.optimizer = optimizer
        self.epoch_lr_pairs = epoch_lr_pairs
        self.logger = logger

    def on_train_begin(self):
        if len(self.epoch_lr_pairs) == 0: return
        epoch, lr = self.epoch_lr_pairs[0]
        if epoch > self.epoch_counter: return
        self.epoch_lr_pairs.pop(0)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.logger(f'LRSchedule: LR set to {lr:.5f}')

    def on_validation_end(self, logs, outputs, gt):
        self.epoch_counter += 1
