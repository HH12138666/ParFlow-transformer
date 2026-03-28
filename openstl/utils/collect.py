import numpy as np
import torch

from .progressbar import ProgressBar


def nondist_forward_collect(func, data_loader, length, to_numpy=False):
    """Forward and collect network outputs in single-process mode."""
    results = []
    prog_bar = ProgressBar(len(data_loader))
    for data in data_loader:
        with torch.no_grad():
            result = func(*data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        if to_numpy:
            results_all[k] = np.concatenate(
                [batch[k].cpu().numpy() for batch in results], axis=0
            )
        else:
            results_all[k] = torch.cat([batch[k] for batch in results], dim=0)
        assert results_all[k].shape[0] == length
    return results_all
