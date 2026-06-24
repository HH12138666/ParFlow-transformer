"""Evaluation and spatial-window reconstruction for model methods."""

import numpy as np
import torch

from openstl.core.metrics import metric
from openstl.utils import ProgressBar


class EvaluationRunner:
    def __init__(self, method):
        self.method = method

    def run(self, data_loader):
        dataset = data_loader.dataset
        if getattr(dataset, "use_space", False):
            return self._run_spatial(data_loader, dataset)
        return self._run_array(data_loader, dataset)

    def _run_array(self, data_loader, dataset):
        collected = self._collect_arrays(data_loader)
        if collected["preds"].size == 0:
            return self._metric_results({}, collected["loss"], "")
        progress = ProgressBar(
            len(self.method.metric_list) + int(dataset.mean is not None)
        )
        eval_res, eval_log = metric(
            collected["preds"],
            collected["trues"],
            dataset.mean,
            dataset.std,
            metrics=self.method.metric_list,
            spatial_norm=self.method.spatial_norm,
            progress_hook=progress.update,
        )
        progress.file.write("\n")
        return self._metric_results(eval_res, collected["loss"], eval_log)

    def _collect_arrays(self, data_loader):
        results = []
        progress = ProgressBar(len(data_loader))
        for batch_x, batch_y in data_loader:
            pred_y, batch_y = self._predict_batch(batch_x, batch_y)
            results.append(
                {
                    "preds": pred_y.cpu().numpy(),
                    "trues": batch_y.cpu().numpy(),
                    "loss": self._loss_array(pred_y, batch_y),
                }
            )
            progress.update()
            self._empty_cuda_cache()
        progress.file.write("\n")
        if not results:
            empty = np.empty((0,), dtype=np.float32)
            return {"preds": empty, "trues": empty, "loss": empty}
        return {
            key: np.concatenate([item[key] for item in results], axis=0)
            for key in results[0]
        }

    def _run_spatial(self, data_loader, dataset):
        sample_count = len(dataset.sample_indices)
        state = self._new_spatial_state(dataset)
        losses = []
        offset = 0
        progress = ProgressBar(len(data_loader))
        for batch_x, batch_y in data_loader:
            pred_y, batch_y = self._predict_batch(batch_x, batch_y)
            losses.append(self._loss_array(pred_y, batch_y))
            offset = self._consume_batch(dataset, state, pred_y, batch_y, offset)
            progress.update()
        progress.file.write("\n")
        if offset != sample_count:
            raise ValueError(f"Consumed {offset} spatial samples, expected {sample_count}")
        self._finalize_slot(state)
        eval_res = self._stream_metrics(state["sums"])
        return self._metric_results(
            eval_res,
            np.concatenate(losses) if losses else np.empty((0,), dtype=np.float32),
            self._format_log(eval_res),
        )

    def _predict_batch(self, batch_x, batch_y):
        with torch.inference_mode():
            batch_x = batch_x.to(self.method.device)
            batch_y = batch_y.to(self.method.device)
            pred_y = self.method._predict(batch_x, batch_y)
        self.method._check_eval_channels(pred_y, batch_y)
        return pred_y, batch_y

    def _loss_array(self, pred_y, batch_y):
        loss = self.method.criterion(pred_y, batch_y)
        return loss.detach().cpu().numpy().reshape(1)

    def _new_spatial_state(self, dataset):
        return {
            "active_slot": None,
            "finalized_slots": set(),
            "pred_buffer": None,
            "true_buffer": None,
            "counts": None,
            "dataset": dataset,
            "sums": {"abs": 0.0, "sq": 0.0, "denom": 0.0},
        }

    def _consume_batch(self, dataset, state, pred_y, true_y, offset):
        pred_np = pred_y.cpu().numpy()
        true_np = true_y.cpu().numpy()
        end = offset + pred_np.shape[0]
        if end > len(dataset.sample_indices):
            raise ValueError("Evaluation batch exceeds spatial merge plan")
        for index in range(pred_np.shape[0]):
            plan_index = offset + index
            self._consume_patch(
                state,
                int(dataset.merge_slots[plan_index]),
                int(dataset.merge_tops[plan_index]),
                int(dataset.merge_lefts[plan_index]),
                pred_np[index],
                true_np[index],
            )
        return end

    def _consume_patch(self, state, slot, top, left, pred_patch, true_patch):
        if state["active_slot"] is None:
            self._start_slot(state, slot, pred_patch)
        elif slot != state["active_slot"]:
            self._finalize_slot(state)
            self._start_slot(state, slot, pred_patch)
        dataset = state["dataset"]
        bottom = top + dataset.space_h
        right = left + dataset.space_w
        state["pred_buffer"][:, :, top:bottom, left:right] += pred_patch
        state["true_buffer"][:, :, top:bottom, left:right] += true_patch
        state["counts"][top:bottom, left:right] += 1

    def _start_slot(self, state, slot, pred_patch):
        if slot in state["finalized_slots"]:
            raise ValueError(f"Spatial slot {slot} is not contiguous")
        dataset = state["dataset"]
        shape = (pred_patch.shape[0], pred_patch.shape[1], dataset.H, dataset.W)
        state["active_slot"] = slot
        state["pred_buffer"] = np.zeros(shape, dtype=pred_patch.dtype)
        state["true_buffer"] = np.zeros(shape, dtype=pred_patch.dtype)
        state["counts"] = np.zeros((dataset.H, dataset.W), dtype=np.int32)

    def _finalize_slot(self, state):
        if state["active_slot"] is None:
            return
        counts = state["counts"]
        if np.any(counts <= 0):
            raise ValueError(f"Spatial slot {state['active_slot']} has uncovered pixels")
        divisor = counts[None, None]
        prediction = state["pred_buffer"] / divisor
        truth = state["true_buffer"] / divisor
        self._update_sums(state["sums"], prediction, truth, state["dataset"])
        state["finalized_slots"].add(state["active_slot"])
        state["active_slot"] = None
        state["pred_buffer"] = None
        state["true_buffer"] = None
        state["counts"] = None

    def _update_sums(self, sums, prediction, truth, dataset):
        difference = prediction.astype(np.float64) - truth.astype(np.float64)
        std = self._output_std(dataset, difference.shape[1])
        if std is not None:
            difference *= std
        sums["abs"] += np.abs(difference).sum(dtype=np.float64)
        sums["sq"] += np.square(difference).sum(dtype=np.float64)
        sums["denom"] += (
            difference.size if self.method.spatial_norm else difference.shape[0]
        )

    @staticmethod
    def _output_std(dataset, channels):
        if dataset.std is None:
            return None
        std = np.asarray(dataset.std, dtype=np.float64)
        if channels > std.shape[0]:
            raise ValueError(f"Output C={channels} exceeds stats C={std.shape[0]}")
        return std[:channels].reshape(1, channels, 1, 1)

    def _stream_metrics(self, sums):
        if sums["denom"] <= 0:
            raise ValueError("Cannot compute metrics with zero denominator")
        invalid = set(self.method.metric_list) - {"mae", "mse", "rmse"}
        if invalid:
            raise ValueError(f"Unsupported metrics: {invalid}")
        results = {}
        if "mae" in self.method.metric_list:
            results["mae"] = sums["abs"] / sums["denom"]
        if "mse" in self.method.metric_list:
            results["mse"] = sums["sq"] / sums["denom"]
        if "rmse" in self.method.metric_list:
            results["rmse"] = np.sqrt(sums["sq"] / sums["denom"])
        return results

    def _metric_results(self, eval_res, losses, eval_log):
        results = {
            "metric_dict": eval_res,
            "eval_log": eval_log,
            "loss": losses,
        }
        for key in self.method.metric_list:
            if key in eval_res:
                results[key] = np.asarray(eval_res[key]).reshape(1)
        results["metrics"] = np.asarray(
            [eval_res[key] for key in self.method.metric_list if key in eval_res],
            dtype=np.float32,
        )
        return results

    @staticmethod
    def _format_log(eval_res):
        return ", ".join(f"{key}:{value}" for key, value in eval_res.items())

    def _empty_cuda_cache(self):
        if self.method.args.empty_cache and self.method.device.type == "cuda":
            torch.cuda.empty_cache()
