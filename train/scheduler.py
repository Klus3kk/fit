class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.current_epoch = 0
        self.initial_lr = optimizer.lr

    def step(self):
        self.current_epoch += 1
        if self.current_epoch % self.step_size == 0:
            self.optimizer.lr = self.optimizer.lr * self.gamma

    def get_lr(self):
        return self.optimizer.lr