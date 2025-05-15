import numpy as np
import pytest

from core.tensor import Tensor
from monitor.tracker import TrainingTracker
from nn.activations import ReLU, Softmax
from nn.linear import Linear
from nn.sequential import Sequential
from train.loss import CrossEntropyLoss, MSELoss
from train.optim import SGD
from train.scheduler import StepLR
from train.trainer import Trainer


class TestLoss:
    def test_mse_loss(self):
        mse = MSELoss()
        predictions = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        targets = Tensor(np.array([[2.0, 2.0], [4.0, 3.0]]))

        loss = mse(predictions, targets)

        # MSE = mean((pred - target)^2)
        # = mean([(1-2)^2, (2-2)^2, (3-4)^2, (4-3)^2])
        # = mean([1, 0, 1, 1])
        # = 0.75
        assert np.isclose(loss.data, 0.75)

    def test_cross_entropy_loss(self):
        ce = CrossEntropyLoss()
        # Logits for 3 samples, 3 classes
        logits = Tensor(
            np.array(
                [
                    [2.0, 1.0, 0.1],  # First sample, most confident about class 0
                    [0.1, 2.0, 1.0],  # Second sample, most confident about class 1
                    [0.1, 1.0, 2.0],  # Third sample, most confident about class 2
                ]
            )
        )
        # True class indices
        targets = Tensor(np.array([0, 1, 2]))

        loss = ce(logits, targets)

        # Manual cross-entropy calculation:
        # 1. Apply softmax to get probabilities
        # 2. Take -log of probability for true class
        # 3. Average

        # Softmax for sample 1: [0.65, 0.24, 0.10] (approximately)
        # Softmax for sample 2: [0.10, 0.65, 0.24]
        # Softmax for sample 3: [0.10, 0.24, 0.65]

        # -log(0.65) + -log(0.65) + -log(0.65) = 3 * -log(0.65) ≈ 1.29

        assert loss.data > 0
        assert np.isclose(loss.data, -np.log(0.65), rtol=0.1)


class TestTrainer:
    def setup_method(self):
        # Create a simple dataset - XOR problem
        self.X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), requires_grad=False)
        self.y = Tensor(np.array([0, 1, 1, 0]), requires_grad=False)

        # Create a simple model
        self.model = Sequential(Linear(2, 4), ReLU(), Linear(4, 2), Softmax())

        # Set fixed weights for deterministic testing
        self.model.layers[0].weight.data = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        )
        self.model.layers[0].bias.data = np.array([0.1, 0.1, 0.1, 0.1])
        self.model.layers[2].weight.data = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        )
        self.model.layers[2].bias.data = np.array([0.1, 0.1])

        # Create loss function and optimizer
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=0.1)

    def test_trainer_init(self):
        trainer = Trainer(self.model, self.loss_fn, self.optimizer)
        assert trainer.model == self.model
        assert trainer.loss_fn == self.loss_fn
        assert trainer.optimizer == self.optimizer
        assert trainer.tracker is None
        assert trainer.scheduler is None
        assert trainer.grad_clip is None

    def test_trainer_fit_one_epoch(self):
        trainer = Trainer(self.model, self.loss_fn, self.optimizer)

        # Save initial weights
        initial_w1 = self.model.layers[0].weight.data.copy()
        initial_b1 = self.model.layers[0].bias.data.copy()
        initial_w2 = self.model.layers[2].weight.data.copy()
        initial_b2 = self.model.layers[2].bias.data.copy()

        # Train for 1 epoch
        trainer.fit(self.X, self.y, epochs=1)

        # Check that weights have changed
        assert not np.array_equal(initial_w1, self.model.layers[0].weight.data)
        assert not np.array_equal(initial_b1, self.model.layers[0].bias.data)
        assert not np.array_equal(initial_w2, self.model.layers[2].weight.data)
        assert not np.array_equal(initial_b2, self.model.layers[2].bias.data)

    def test_trainer_with_scheduler(self):
        scheduler = StepLR(self.optimizer, step_size=1, gamma=0.5)
        trainer = Trainer(self.model, self.loss_fn, self.optimizer, scheduler=scheduler)

        # Initial learning rate
        initial_lr = self.optimizer.lr

        # Train for 2 epochs
        trainer.fit(self.X, self.y, epochs=2)

        # After 2 epochs, learning rate should be reduced once
        assert self.optimizer.lr == initial_lr * 0.5

    def test_trainer_evaluate(self):
        trainer = Trainer(self.model, self.loss_fn, self.optimizer)

        # Evaluate before training
        loss, acc = trainer.evaluate(self.X, self.y)

        # Loss should be a positive float
        assert isinstance(loss, float)
        assert loss > 0

        # Accuracy should be between 0 and 1
        assert 0 <= acc <= 1

    def test_trainer_with_tracker(self):
        tracker = TrainingTracker()
        trainer = Trainer(self.model, self.loss_fn, self.optimizer, tracker=tracker)

        # Train for 1 epoch
        trainer.fit(self.X, self.y, epochs=1)

        # Check that tracker has recorded the data
        assert len(tracker.logs["loss"]) == 1
        assert "accuracy" in tracker.logs
