from airbus.callbacks.callback import Callback

class CyclicLR(Callback):
    def __init__(self, step_size, min_lr, max_lr, optimizer, logger):
        self.step_size = step_size
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.counter = 0
        self.optimizer = optimizer
        self.logger = logger

    def on_train_batch_end(self, *_):
        self.counter += 1

        if (self.counter // self.step_size) % 2 == 0:
            new_lr = self.min_lr + (self.max_lr - self.min_lr) * (self.counter % self.step_size) / self.step_size
        else:
            new_lr = self.max_lr - (self.max_lr - self.min_lr) * (self.counter % self.step_size) / self.step_size

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        if (new_lr == self.min_lr) and self.logger:
            self.logger(f'CyclicLR: Learning rate has been reset')
