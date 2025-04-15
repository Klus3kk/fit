class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def fit(self, x, y, epochs=10, verbose=True):
        for epoch in range(1, epochs + 1):
            # Forward
            preds = self.model(x)
            loss = self.loss_fn(preds, y)

            # Backward
            loss.backward()

            # Step
            self.optimizer.step()
            self.optimizer.zero_grad()

            if verbose:
                print(f"Epoch {epoch:03d} | Loss: {loss.data[0]:.6f}")
