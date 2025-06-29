class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.current_epoch = 0
        # Store BOTH the initial and original learning rate
        self.initial_lr = optimizer.lr
        self._original_lr = optimizer.lr

    def step(self):
        """
        Update learning rate based on the current epoch.
        This should be called once per epoch.
        """
        self.current_epoch += 1

        # Only apply gamma when step_size is reached
        if self.current_epoch % self.step_size == 0:
            # Calculate directly from original learning rate
            factor = self.gamma ** (self.current_epoch // self.step_size)
            self.optimizer.lr = self._original_lr * factor

    def get_lr(self):
        return self.optimizer.lr
