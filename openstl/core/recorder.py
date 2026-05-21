import numpy as np
import torch


class Recorder:
    def __init__(self, verbose=False, delta=0, early_stop_time=10, monitor_name="metric"):
        self.verbose = verbose
        self.best_score = None
        self.best_value = np.Inf
        self.delta = delta
        self.decrease_time = 0
        self.early_stop_time = early_stop_time
        self.monitor_name = monitor_name

    def __call__(self, monitor_value, model, path):
        score = -monitor_value
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(monitor_value, model, path)
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(monitor_value, model, path)
            self.decrease_time = 0
        else:
            self.decrease_time += 1
        return self.decrease_time >= self.early_stop_time

    def save_checkpoint(self, monitor_value, model, path):
        if self.verbose:
            print(
                f"Validation {self.monitor_name} decreased "
                f"({self.best_value:.6f} --> {monitor_value:.6f}).  Saving model ..."
            )
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), path+'/'+'checkpoint.pth')
        self.best_value = monitor_value
