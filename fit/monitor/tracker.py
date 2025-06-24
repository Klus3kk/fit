"""
Training tracker for monitoring and logging training progress.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np

from fit.core.tensor import Tensor


class TrainingTracker:
    """
    Comprehensive training tracker for monitoring metrics, early stopping, and logging.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        log_dir: str = "./logs",
        early_stopping: Optional[Dict] = None,
        save_best: bool = True,
        verbose: int = 1,
    ):
        """
        Initialize training tracker.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save logs
            early_stopping: Early stopping configuration
            save_best: Whether to save best model state
            verbose: Verbosity level (0=silent, 1=normal, 2=verbose)
        """
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        self.log_dir = log_dir
        self.verbose = verbose
        self.save_best = save_best

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Initialize tracking data
        self.logs = {}
        self.current_epoch = 0
        self.start_time = None
        self.best_values = {}
        self.best_epochs = {}

        # Early stopping
        self.early_stopping = early_stopping
        if early_stopping:
            self.patience = early_stopping.get("patience", 10)
            self.min_delta = early_stopping.get("min_delta", 1e-4)
            self.monitor_metric = early_stopping.get("metric", "val_loss")
            self.mode = early_stopping.get("mode", "min")  # 'min' or 'max'
            self.wait = 0
            self.stopped_epoch = 0
            self.best_value = float("inf") if self.mode == "min" else float("-inf")

        # Metrics history
        self.epoch_times = []
        self.learning_rates = []

    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
        if self.verbose >= 1:
            print(f"Starting training experiment: {self.experiment_name}")
            print(f"Logs will be saved to: {self.log_dir}")

    def update(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Update tracker with metrics for current epoch.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric names and values

        Returns:
            True if training should stop (early stopping), False otherwise
        """
        self.current_epoch = epoch

        # Record metrics
        for metric_name, value in metrics.items():
            if metric_name not in self.logs:
                self.logs[metric_name] = []

            # Convert tensor to float if needed
            if isinstance(value, Tensor):
                value = float(value.data)
            elif isinstance(value, np.ndarray):
                value = float(value)

            self.logs[metric_name].append(value)

            # Track best values
            if metric_name not in self.best_values:
                self.best_values[metric_name] = value
                self.best_epochs[metric_name] = epoch
            else:
                # Update best value (assuming lower is better for loss, higher for accuracy)
                is_better = False
                if "loss" in metric_name.lower() or "error" in metric_name.lower():
                    is_better = value < self.best_values[metric_name]
                else:
                    is_better = value > self.best_values[metric_name]

                if is_better:
                    self.best_values[metric_name] = value
                    self.best_epochs[metric_name] = epoch

        # Check early stopping
        should_stop = False
        if self.early_stopping and self.monitor_metric in metrics:
            should_stop = self._check_early_stopping(metrics[self.monitor_metric])

        # Log progress
        if self.verbose >= 1:
            self._log_epoch(epoch, metrics)

        return should_stop

    def _check_early_stopping(self, current_value: float) -> bool:
        """
        Check if early stopping criteria are met.

        Args:
            current_value: Current value of monitored metric

        Returns:
            True if training should stop
        """
        if self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = self.current_epoch
            if self.verbose >= 1:
                print(f"\nEarly stopping at epoch {self.current_epoch}")
                print(
                    f"Best {self.monitor_metric}: {self.best_value} at epoch {self.current_epoch - self.wait}"
                )
            return True

        return False

    def _log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for current epoch."""
        if epoch == 0:
            # Print header
            header = f"{'Epoch':<6}"
            for metric in metrics.keys():
                header += f"{metric:<12}"
            header += f"{'Time':<8}"
            print(header)
            print("-" * len(header))

        # Print metrics
        log_line = f"{epoch + 1:<6}"
        for metric, value in metrics.items():
            if isinstance(value, float):
                log_line += f"{value:<12.4f}"
            else:
                log_line += f"{value:<12}"

        # Add time if available
        if len(self.epoch_times) > epoch:
            log_line += f"{self.epoch_times[epoch]:<8.2f}s"

        print(log_line)

    def log_epoch_time(self, epoch_time: float):
        """Log time taken for current epoch."""
        self.epoch_times.append(epoch_time)

    def log_learning_rate(self, lr: float):
        """Log current learning rate."""
        self.learning_rates.append(lr)

    def plot_metrics(
        self, metrics: Optional[List[str]] = None, save_path: Optional[str] = None
    ):
        """
        Plot training metrics.

        Args:
            metrics: List of metrics to plot (default: all)
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib")
            return False

        if not self.logs:
            print("No metrics to plot")
            return False

        metrics_to_plot = metrics or list(self.logs.keys())
        metrics_to_plot = [
            m for m in metrics_to_plot if m in self.logs and self.logs[m]
        ]

        if not metrics_to_plot:
            print("No valid metrics found to plot")
            return False

        # Create subplots
        n_metrics = len(metrics_to_plot)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            epochs = range(1, len(self.logs[metric]) + 1)
            ax.plot(epochs, self.logs[metric], "b-", linewidth=2, label=metric)

            # Mark best value
            if metric in self.best_values:
                best_epoch = self.best_epochs[metric]
                best_value = self.best_values[metric]
                ax.plot(
                    best_epoch + 1,
                    best_value,
                    "ro",
                    markersize=8,
                    label=f"Best: {best_value:.4f}",
                )

            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].remove()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            # Save to log directory
            plot_path = os.path.join(
                self.log_dir, f"{self.experiment_name}_metrics.png"
            )
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {plot_path}")

        plt.show()
        return True

    def export(self, filepath: Optional[str] = None, format: str = "json") -> str:
        """
        Export training logs to file.

        Args:
            filepath: Path to save file (auto-generated if None)
            format: Export format ("json" or "csv")

        Returns:
            Path to exported file
        """
        if filepath is None:
            ext = ".json" if format.lower() == "json" else ".csv"
            filepath = os.path.join(self.log_dir, f"{self.experiment_name}_logs{ext}")

        if format.lower() == "json":
            return self._export_json(filepath)
        else:
            return self._export_csv(filepath)

    def _export_json(self, filepath: str) -> str:
        """Export logs to JSON format."""
        export_data = {
            "experiment_name": self.experiment_name,
            "total_epochs": self.current_epoch + 1,
            "training_time": time.time() - self.start_time if self.start_time else None,
            "best_values": self.best_values,
            "best_epochs": self.best_epochs,
            "logs": self.logs,
            "early_stopping": (
                {
                    "stopped": self.stopped_epoch > 0,
                    "stopped_epoch": self.stopped_epoch,
                    "patience": getattr(self, "patience", None),
                    "monitor_metric": getattr(self, "monitor_metric", None),
                }
                if self.early_stopping
                else None
            ),
            "epoch_times": self.epoch_times,
            "learning_rates": self.learning_rates,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"Logs exported to {filepath}")
        return filepath

    def _export_csv(self, filepath: str) -> str:
        """Export logs to CSV format."""
        import csv

        if not self.logs:
            print("No logs to export")
            return filepath

        # Get all metric names
        metrics = list(self.logs.keys())
        max_epochs = max(len(values) for values in self.logs.values())

        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            header = ["epoch"] + metrics
            if self.epoch_times:
                header.append("epoch_time")
            if self.learning_rates:
                header.append("learning_rate")
            writer.writerow(header)

            # Write data
            for epoch in range(max_epochs):
                row = [epoch + 1]

                # Add metric values
                for metric in metrics:
                    if epoch < len(self.logs[metric]):
                        row.append(self.logs[metric][epoch])
                    else:
                        row.append("")

                # Add epoch time if available
                if self.epoch_times and epoch < len(self.epoch_times):
                    row.append(self.epoch_times[epoch])
                elif self.epoch_times:
                    row.append("")

                # Add learning rate if available
                if self.learning_rates and epoch < len(self.learning_rates):
                    row.append(self.learning_rates[epoch])
                elif self.learning_rates:
                    row.append("")

                writer.writerow(row)

        print(f"Logs exported to {filepath}")
        return filepath

    def load(self, filepath: str) -> bool:
        """
        Load training logs from file.

        Args:
            filepath: Path to log file (.json)

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            self.experiment_name = data.get("experiment_name", self.experiment_name)
            self.logs = data.get("logs", {})
            self.best_values = data.get("best_values", {})
            self.best_epochs = data.get("best_epochs", {})
            self.epoch_times = data.get("epoch_times", [])
            self.learning_rates = data.get("learning_rates", [])

            # Update current epoch
            if self.logs:
                self.current_epoch = (
                    max(len(values) for values in self.logs.values()) - 1
                )

            print(f"Logs loaded from {filepath}")
            return True

        except Exception as e:
            print(f"Error loading logs: {e}")
            return False

    def summary(self) -> Dict[str, Any]:
        """
        Get training summary.

        Returns:
            Dictionary with training summary
        """
        total_time = time.time() - self.start_time if self.start_time else None

        summary = {
            "experiment_name": self.experiment_name,
            "total_epochs": self.current_epoch + 1,
            "training_time": total_time,
            "best_metrics": self.best_values,
            "early_stopped": self.stopped_epoch > 0,
            "stopped_epoch": self.stopped_epoch if self.stopped_epoch > 0 else None,
        }

        if total_time:
            summary["avg_epoch_time"] = total_time / (self.current_epoch + 1)

        return summary

    def __str__(self) -> str:
        """String representation of tracker."""
        summary = self.summary()
        lines = [f"TrainingTracker: {summary['experiment_name']}"]
        lines.append(f"  Epochs: {summary['total_epochs']}")

        if summary.get("training_time"):
            lines.append(f"  Training time: {summary['training_time']:.2f}s")

        if summary["best_metrics"]:
            lines.append("  Best metrics:")
            for metric, value in summary["best_metrics"].items():
                epoch = self.best_epochs.get(metric, 0)
                lines.append(f"    {metric}: {value:.4f} (epoch {epoch + 1})")

        if summary["early_stopped"]:
            lines.append(f"  Early stopped at epoch {summary['stopped_epoch']}")

        return "\n".join(lines)
