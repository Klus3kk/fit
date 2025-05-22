import json
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np


class TrainingTracker:
    def __init__(
        self,
        experiment_name=None,
        log_dir="./logs",
        save_best=True,
        best_metric="loss",
        track_gradients=False,
        early_stopping=None,
    ):
        """
        Enhanced training tracker with better visualization and logging.

        Args:
            experiment_name: Name for this training run (auto-generated if None)
            log_dir: Directory to save logs
            save_best: Whether to track and report best metric values
            best_metric: Metric to track for "best" model ('loss' or 'accuracy')
            track_gradients: Whether to track gradient norms (requires passing gradient info)
            early_stopping: Dict with early stopping settings, e.g., {'patience': 10, 'min_delta': 0.001}
        """
        self.experiment_name = experiment_name
        if self.experiment_name is None:
            self.experiment_name = "exp_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Core tracking attributes
        self.logs = defaultdict(list)
        self.epoch_start_time = None
        self.training_start_time = time.time()
        self.current_epoch = 0

        # Enhanced features
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_values = {"loss": float("inf"), "accuracy": 0.0}
        self.best_epochs = {"loss": 0, "accuracy": 0}

        self.track_gradients = track_gradients
        self.gradient_norms = []

        # Early stopping
        self.early_stopping = early_stopping or {}
        self.patience_counter = 0
        self.should_stop = False

    def start_epoch(self):
        """Start timing a new training epoch."""
        self.current_epoch += 1
        self.epoch_start_time = time.time()
        return self  # For method chaining

    def log(self, loss, acc=None, lr=None, grad_norm=None, custom_metrics=None):
        """
        Log metrics for the current epoch.

        Args:
            loss: Loss value (required)
            acc: Accuracy value (optional)
            lr: Learning rate (optional)
            grad_norm: Gradient norm (optional, only if track_gradients=True)
            custom_metrics: Dict of any additional metrics to track
        """
        # Calculate epoch duration
        duration = (
            time.time() - self.epoch_start_time if self.epoch_start_time else None
        )

        # Log standard metrics
        self.logs["loss"].append(float(loss))

        if acc is not None:
            self.logs["accuracy"].append(float(acc))

        if lr is not None:
            self.logs["lr"].append(float(lr))

        if duration is not None:
            self.logs["time"].append(duration)

        # Track gradient norms if enabled
        if self.track_gradients and grad_norm is not None:
            self.logs["grad_norm"].append(float(grad_norm))

        # Log any custom metrics
        if custom_metrics:
            for name, value in custom_metrics.items():
                self.logs[name].append(float(value))

        # Update best values
        self._update_best_values()

        # Check early stopping criteria
        if self.early_stopping:
            self._check_early_stopping()

        return self  # For method chaining

    def _update_best_values(self):
        """Update tracking of best metric values."""
        # Update best loss (lower is better)
        current_loss = self.logs["loss"][-1]
        if current_loss < self.best_values["loss"]:
            self.best_values["loss"] = current_loss
            self.best_epochs["loss"] = self.current_epoch

        # Update best accuracy (higher is better)
        if "accuracy" in self.logs:
            current_acc = self.logs["accuracy"][-1]
            if current_acc > self.best_values["accuracy"]:
                self.best_values["accuracy"] = current_acc
                self.best_epochs["accuracy"] = self.current_epoch

    def _check_early_stopping(self):
        """Check if early stopping criteria are met."""
        if not self.early_stopping:
            return

        patience = self.early_stopping.get("patience", 10)
        min_delta = self.early_stopping.get("min_delta", 0.001)
        metric = self.early_stopping.get("metric", "loss")

        if metric not in self.logs:
            return

        # Get current and best values
        current_value = self.logs[metric][-1]
        best_value = self.best_values.get(metric)

        if best_value is None:
            return

        # Check if improvement (depending on metric)
        is_improvement = False

        if metric == "loss" or metric.endswith("_loss"):
            # For loss metrics, lower is better
            is_improvement = current_value < (best_value - min_delta)
        else:
            # For other metrics like accuracy, higher is better
            is_improvement = current_value > (best_value + min_delta)

        # Reset or increment patience counter
        if is_improvement:
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Check if we should stop
        if self.patience_counter >= patience:
            self.should_stop = True

    def should_early_stop(self):
        """Return True if early stopping criteria have been met."""
        return self.should_stop

    def summary(self, last_n=5, style="box", show_best=True):
        """
        Display a summary of training progress.

        Args:
            last_n: Number of most recent epochs to display
            style: Output style ("box", "table", "minimal")
            show_best: Whether to show best metric values
        """
        # Determine length of training period
        total_duration = time.time() - self.training_start_time
        hours, remainder = divmod(total_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

        if style == "box":
            self._box_summary(last_n, show_best, time_str)
        elif style == "table":
            self._table_summary(last_n, show_best, time_str)
        else:
            self._minimal_summary(last_n, show_best, time_str)

    def _box_summary(self, last_n, show_best, time_str):
        """Display summary in a nice box format."""
        print("\n" + "═" * 70)
        print(f"Training Summary: {self.experiment_name}")
        print("═" * 70)

        # Determine metrics to show
        metrics = [k for k in self.logs.keys() if k != "time"]

        # Show last N epochs
        print(f"\nLast {min(last_n, len(self.logs['loss']))} Epochs:")
        print("╭" + "─" * 68 + "╮")

        # Header
        header = "│ Epoch"
        for metric in metrics:
            header += f" | {metric.capitalize()}"
        if "time" in self.logs:
            header += " | Time"
        header = header.ljust(69) + "│"
        print(header)
        print("├" + "─" * 68 + "┤")

        # Data rows
        for i in range(1, min(last_n + 1, len(self.logs["loss"]) + 1)):
            idx = -i
            try:
                row = f"│ {self.current_epoch + idx + 1:4d}"

                for metric in metrics:
                    if (
                        metric == "accuracy"
                        and metric in self.logs
                        and idx < len(self.logs[metric])
                    ):
                        row += f" | {self.logs[metric][idx] * 100:6.2f}%"
                    elif (
                        metric == "lr"
                        and metric in self.logs
                        and idx < len(self.logs[metric])
                    ):
                        row += f" | {self.logs[metric][idx]:.6f}"
                    elif idx < len(self.logs[metric]):
                        row += f" | {self.logs[metric][idx]:.6f}"
                    else:
                        row += " | -"

                if "time" in self.logs and idx < len(self.logs["time"]):
                    row += f" | {self.logs['time'][idx]:.2f}s"

                print(row.ljust(69) + "│")
            except IndexError:
                continue

        print("╰" + "─" * 68 + "╯")

        # Show best values if requested
        if show_best:
            print("\nBest Values:")
            for metric in ["loss", "accuracy"]:
                if metric in self.logs and len(self.logs[metric]) > 0:
                    if metric == "accuracy":
                        best_value = self.best_values[metric] * 100
                        best_epoch = self.best_epochs[metric]
                        print(
                            f"* Best {metric}: {best_value:.2f}% (epoch {best_epoch})"
                        )
                    else:
                        best_value = self.best_values[metric]
                        best_epoch = self.best_epochs[metric]
                        print(f"* Best {metric}: {best_value:.6f} (epoch {best_epoch})")

        # Show early stopping status if enabled
        if self.early_stopping:
            metric_name = self.early_stopping.get("metric", "loss")
            print(f"\nEarly Stopping ({metric_name}):")
            print(
                f"• Patience: {self.patience_counter}/{self.early_stopping.get('patience', 10)}"
            )
            if self.should_stop:
                print("• Status: STOP TRAINING (criteria met)")
            else:
                print("• Status: Continue training")

        # Show total training time
        print(f"\nTotal Training Time: {time_str}")

    def _table_summary(self, last_n, show_best, time_str):
        """Display summary in a simple table format."""
        print(f"\nTraining Summary: {self.experiment_name}")
        print("-" * 70)

        # Determine metrics to show
        metrics = [k for k in self.logs.keys() if k != "time"]

        # Header
        header = f"{'Epoch':6s}"
        for metric in metrics:
            header += f" | {metric:10s}"
        if "time" in self.logs:
            header += f" | {'Time':8s}"
        print(header)
        print("-" * 70)

        # Data rows
        for i in range(1, min(last_n + 1, len(self.logs["loss"]) + 1)):
            idx = -i
            try:
                row = f"{self.current_epoch + idx + 1:6d}"

                for metric in metrics:
                    if (
                        metric == "accuracy"
                        and metric in self.logs
                        and idx < len(self.logs[metric])
                    ):
                        row += f" | {self.logs[metric][idx] * 100:8.2f}%"
                    elif idx < len(self.logs[metric]):
                        row += f" | {self.logs[metric][idx]:10.6f}"
                    else:
                        row += f" | {'-':10s}"

                if "time" in self.logs and idx < len(self.logs["time"]):
                    row += f" | {self.logs['time'][idx]:6.2f}s"

                print(row)
            except IndexError:
                continue

        print("-" * 70)

        # Show best values if requested
        if show_best:
            print("\nBest Values:")
            for metric in ["loss", "accuracy"]:
                if metric in self.logs and len(self.logs[metric]) > 0:
                    best_value = self.best_values[metric]
                    best_epoch = self.best_epochs[metric]

                    if metric == "accuracy":
                        # Format accuracy as percentage
                        formatted_value = f"{best_value * 100:.2f}%"
                    else:
                        # Format other metrics with 6 decimal places
                        formatted_value = f"{best_value:.6f}"

                    print(f"- Best {metric}: {formatted_value} (epoch {best_epoch})")

        # Show total training time
        print(f"\nTotal Training Time: {time_str}")

    def _minimal_summary(self, last_n, show_best, time_str):
        """Display summary in a minimal format."""
        print(f"\nTraining Summary ({self.experiment_name}):")

        # Show latest epoch
        if self.logs["loss"]:
            print(f"Latest (Epoch {self.current_epoch}):")
            print(f"- Loss: {self.logs['loss'][-1]:.6f}")

            if "accuracy" in self.logs and self.logs["accuracy"]:
                print(f"- Accuracy: {self.logs['accuracy'][-1] * 100:.2f}%")

        # Show best values if requested
        if show_best:
            print("\nBest Values:")
            for metric in ["loss", "accuracy"]:
                if metric in self.logs and len(self.logs[metric]) > 0:
                    best_value = self.best_values[metric]
                    best_epoch = self.best_epochs[metric]

                    if metric == "accuracy":
                        # Format accuracy as percentage
                        best_value_formatted = f"{best_value * 100:.2f}%"
                    else:
                        # Format other metrics with 6 decimal places
                        best_value_formatted = f"{best_value:.6f}"

                    print(
                        f"- Best {metric}: {best_value_formatted} (epoch {best_epoch})"
                    )

        # Show total training time
        print(f"\nTotal Training Time: {time_str}")

    def plot(self, metrics=None, figsize=(10, 6), save_path=None):
        """
        Plot training metrics.

        Args:
            metrics: List of metrics to plot (None = all available)
            figsize: Figure size as (width, height)
            save_path: Where to save the plot (displayed if None)

        Returns:
            True if successful, False otherwise
        """
        try:
            import matplotlib.pyplot as plt

            if metrics is None:
                # Plot all metrics except time
                metrics = [
                    k for k in self.logs.keys() if k != "time" and len(self.logs[k]) > 0
                ]

            # Create figure
            fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
            if len(metrics) == 1:
                axes = [axes]

            # Plot each metric
            for i, metric in enumerate(metrics):
                if metric not in self.logs or not self.logs[metric]:
                    continue

                epochs = list(range(1, len(self.logs[metric]) + 1))
                axes[i].plot(epochs, self.logs[metric], "b-", linewidth=2)

                # Add best value marking
                if metric == "loss" or metric == "accuracy":
                    best_epoch = self.best_epochs[metric]
                    best_value = self.best_values[metric]
                    if 0 < best_epoch <= len(epochs):
                        axes[i].plot(best_epoch, best_value, "ro", markersize=5)
                        axes[i].text(
                            best_epoch,
                            best_value,
                            f" Best: {best_value:.6f}",
                            verticalalignment="bottom",
                        )

                # Customize plot
                axes[i].set_title(metric.capitalize())
                axes[i].grid(True)

                # Set y-axis label and limits
                axes[i].set_ylabel(metric)
                if metric == "accuracy":
                    axes[i].set_ylim([0, 1])

            # Set x-axis label
            axes[-1].set_xlabel("Epoch")

            plt.tight_layout()

            # Save or display
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

            return True

        except ImportError:
            print("Plotting requires matplotlib. Install with: pip install matplotlib")
            return False

    def export(self, filepath=None, format="csv"):
        """
        Export training logs to a file.

        Args:
            filepath: Path to save the file (auto-generated if None)
            format: Export format ("csv" or "json")

        Returns:
            Path to the exported file
        """
        if filepath is None:
            ext = ".csv" if format.lower() == "csv" else ".json"
            filepath = os.path.join(self.log_dir, f"training_log{ext}")

        if format.lower() == "csv":
            return self._export_csv(filepath)
        else:
            return self._export_json(filepath)

    def _export_csv(self, filepath):
        """Export logs to CSV format."""
        import csv

        keys = self.logs.keys()
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()

            for i in range(len(self.logs["loss"])):
                row = {
                    k: self.logs[k][i] if i < len(self.logs[k]) else "" for k in keys
                }
                writer.writerow(row)

        print(f"Logs exported to {filepath}")
        return filepath

    def _export_json(self, filepath):
        """Export logs to JSON format."""
        export_data = {
            "experiment_name": self.experiment_name,
            "epochs": self.current_epoch,
            "best_values": self.best_values,
            "best_epochs": self.best_epochs,
            "logs": {k: [float(v) for v in vals] for k, vals in self.logs.items()},
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"Logs exported to {filepath}")
        return filepath

    def load(self, filepath):
        """
        Load training logs from file.

        Args:
            filepath: Path to the log file (.json or .csv)

        Returns:
            True if successful, False otherwise
        """
        if filepath.endswith(".json"):
            return self._load_json(filepath)
        elif filepath.endswith(".csv"):
            return self._load_csv(filepath)
        else:
            print(f"Unsupported file format: {filepath}")
            return False

    def _load_json(self, filepath):
        """Load logs from JSON file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Restore logs
            self.logs = defaultdict(list)
            for k, vals in data.get("logs", {}).items():
                self.logs[k] = vals

            # Restore other attributes
            self.experiment_name = data.get("experiment_name", self.experiment_name)
            self.current_epoch = data.get("epochs", len(self.logs.get("loss", [])))
            self.best_values = data.get("best_values", self.best_values)
            self.best_epochs = data.get("best_epochs", self.best_epochs)

            print(f"Loaded training logs from {filepath}")
            return True

        except Exception as e:
            print(f"Error loading logs from {filepath}: {e}")
            return False

    def _load_csv(self, filepath):
        """Load logs from CSV file."""
        try:
            import csv

            self.logs = defaultdict(list)

            with open(filepath, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    for k, v in row.items():
                        if v != "":
                            self.logs[k].append(float(v))

            # Recalculate current epoch and best values
            self.current_epoch = len(self.logs.get("loss", []))
            self._update_best_values_all()

            print(f"Loaded training logs from {filepath}")
            return True

        except Exception as e:
            print(f"Error loading logs from {filepath}: {e}")
            return False

    def _update_best_values_all(self):
        """Update best values based on all logged data."""
        # Reset best values
        self.best_values = {"loss": float("inf"), "accuracy": 0.0}
        self.best_epochs = {"loss": 0, "accuracy": 0}

        # Update loss
        if "loss" in self.logs and self.logs["loss"]:
            best_loss = min(self.logs["loss"])
            best_loss_epoch = self.logs["loss"].index(best_loss) + 1

            self.best_values["loss"] = best_loss
            self.best_epochs["loss"] = best_loss_epoch

        # Update accuracy
        if "accuracy" in self.logs and self.logs["accuracy"]:
            best_acc = max(self.logs["accuracy"])
            best_acc_epoch = self.logs["accuracy"].index(best_acc) + 1

            self.best_values["accuracy"] = best_acc
            self.best_epochs["accuracy"] = best_acc_epoch

    def get_progress_stats(self):
        """
        Get statistics about training progress.

        Returns:
            Dictionary with progress statistics
        """
        stats = {
            "epochs": self.current_epoch,
            "best_values": self.best_values.copy(),
            "best_epochs": self.best_epochs.copy(),
            "early_stopping": {
                "enabled": bool(self.early_stopping),
                "patience_counter": self.patience_counter,
                "should_stop": self.should_stop,
            },
        }

        # Calculate improvement rates (from last 5 epochs)
        for metric in ["loss", "accuracy"]:
            if metric in self.logs and len(self.logs[metric]) >= 5:
                # Calculate absolute improvement rate
                values = self.logs[metric][-5:]
                first, last = values[0], values[-1]
                abs_change = last - first

                # Calculate relative improvement rate (% change per epoch)
                relative_change = abs_change / abs(first) * 100 / 4  # per epoch

                # Add to stats
                if "improvement_rates" not in stats:
                    stats["improvement_rates"] = {}

                stats["improvement_rates"][metric] = {
                    "absolute": abs_change,
                    "relative_percent": relative_change,
                }

        # Calculate metric trends
        if len(self.logs.get("loss", [])) >= 3:
            trends = {}

            for metric in ["loss", "accuracy"]:
                if metric in self.logs and len(self.logs[metric]) >= 3:
                    values = self.logs[metric][-3:]

                    # Determine trend (increasing, decreasing, stable)
                    if all(values[i] < values[i - 1] for i in range(1, len(values))):
                        trend = "decreasing"
                    elif all(values[i] > values[i - 1] for i in range(1, len(values))):
                        trend = "increasing"
                    else:
                        trend = "fluctuating"

                    # Determine if trend is good
                    is_good = (metric == "loss" and trend == "decreasing") or (
                        metric != "loss" and trend == "increasing"
                    )

                    trends[metric] = {"trend": trend, "is_good": is_good}

            stats["trends"] = trends

        return stats
