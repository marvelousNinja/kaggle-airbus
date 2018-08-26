class CyclicLR:
    def __init__(self, cycle_iterations, min_lr, max_lr, optimizer, logger):
        self.cycle_iterations = cycle_iterations
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.counter = 0
        self.optimizer = optimizer
        self.logger = logger

    def step(self):
        self.counter += 1
        new_lr = self.max_lr - (self.max_lr - self.min_lr) * (self.counter % self.cycle_iterations) / self.cycle_iterations
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        if (self.counter % self.cycle_iterations == 0) and self.logger:
            self.logger(f'CyclicLR: Learning rate has been reset')
