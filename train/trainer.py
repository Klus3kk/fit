import time

class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def _set_training_mode(self, training=True):
        def set_mode(module):
            if hasattr(module, 'training'):
                module.training = training
            if hasattr(module, '_children'):
                for child in module._children:
                    set_mode(child)

        set_mode(self.model)

    def fit(self, x, y, epochs=10, verbose=True):
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Forward
            preds = self.model(x)
            loss = self.loss_fn(preds, y)

            # Backward
            loss.backward()

            # Optimizer
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Accuracy calculation
            acc = None
            if preds.data.ndim == 2 and preds.data.shape[1] > 1:  # classification
                predicted_labels = preds.data.argmax(axis=1)
                correct = (predicted_labels == y.data).sum()
                acc = correct / len(y.data)

            duration = time.time() - start_time

            if verbose:
                acc_str = f"{acc * 100:.2f}%" if acc is not None else "-"
                print("╭" + "─" * 50 + "╮")
                print(f"│ Epoch {epoch:03d} | Loss: {loss.data:.4f} | Acc: {acc_str:>6} | LR: {self.optimizer.lr:.4f} │")
                print("╰" + "─" * 50 + "╯")

    def evaluate(self, x, y):
        self._set_training_mode(False)
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        acc = None
        if preds.data.ndim == 2:
            pred_labels = preds.data.argmax(axis=1)
            correct = (pred_labels == y.data).sum()
            acc = correct / len(y.data)
        return float(loss.data), float(acc)

