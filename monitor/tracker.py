from collections import defaultdict
import time


class TrainingTracker:
    def __init__(self):
        self.logs = defaultdict(list)
        self.epoch_start_time = None

    def start_epoch(self):
        self.epoch_start_time = time.time()

    def log(self, loss, acc=None, lr=None):
        duration = time.time() - self.epoch_start_time if self.epoch_start_time else None
        self.logs["loss"].append(float(loss))
        if acc is not None:
            self.logs["accuracy"].append(float(acc))
        if lr is not None:
            self.logs["lr"].append(float(lr))
        if duration is not None:
            self.logs["time"].append(duration)

    def summary(self, last_n=5):
        print("Training Summary (Last Epochs):")
        print("╭" + "─" * 60 + "╮")
        for i in range(1, last_n + 1):
            idx = -i
            try:
                line = f"│ Epoch {-idx:03d} | Loss: {self.logs['loss'][idx]:.4f}"
                if "accuracy" in self.logs:
                    line += f" | Acc: {self.logs['accuracy'][idx] * 100:.2f}%"
                if "lr" in self.logs:
                    line += f" | LR: {self.logs['lr'][idx]:.5f}"
                if "time" in self.logs:
                    line += f" | Time: {self.logs['time'][idx]:.2f}s"
                print(line.ljust(61) + "│")
            except IndexError:
                continue
        print("╰" + "─" * 60 + "╯")

    def export(self, filepath):
        import csv
        keys = self.logs.keys()
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for i in range(len(self.logs["loss"])):
                row = {k: self.logs[k][i] if i < len(self.logs[k]) else '' for k in keys}
                writer.writerow(row)
