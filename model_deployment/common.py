import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from openstl.datasets.parflow.paths import extract_hour_id as extract_id, prepare_press_evap_files, resolve_parflow_roots
from openstl.datasets.parflow.readers import read_combined_frame, read_static_stack
from openstl.models import PredFormer_Model




@dataclass(frozen=True)
class InferenceConfig:
    run_dir: str = ""
    checkpoint_file: str = "latest.pth"
    explicit_checkpoint_path: str = ""
    data_root_override: str = ""
    output_dir: str = "inference_data/press"
    run_param: str = "test1_2019_press_evap_static_train1_1.4_moderate_heavy_20_21_post_p8_latest"
    start_hour: int = 20190000
    end_hour: int = 20198759
    use_rollout: bool = True
    rollout_hours: int = 720
    use_amp: bool = True
    amp_dtype: str = "fp16"
    patch_batch_size: int = 28
    preload_rollout_aux: bool = True
    empty_cache_each_block: bool = False
    eps: float = 1e-8


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_checkpoint(run_dir, checkpoint_file, explicit_path=""):
    if explicit_path:
        path = Path(explicit_path)
    else:
        run_path = Path(run_dir)
        file_path = Path(checkpoint_file)
        if file_path.is_absolute():
            path = file_path
        elif (run_path / file_path).exists():
            path = run_path / file_path
        else:
            path = run_path / "checkpoints" / file_path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.resolve()


def load_model_params(checkpoint_path):
    ckpt_path = Path(checkpoint_path).resolve()
    candidates = [ckpt_path.parent.parent / "model_param.json", ckpt_path.parent / "model_param.json"]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as file_obj:
                params = json.load(file_obj)
            if not isinstance(params, dict):
                raise ValueError(f"model_param.json must be a dict, got {type(params)}")
            return params, path
    raise FileNotFoundError(f"model_param.json not found for checkpoint: {checkpoint_path}")


def require_param(params, key):
    if key not in params:
        raise KeyError(f"model_param.json missing required key: {key}")
    return params[key]


def require_model_key(model_config, key):
    if key not in model_config:
        raise KeyError(f"model_config missing required key: {key}")
    return model_config[key]


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def build_aligned_var_evap_items(var_root, evap_root=None):
    press_files, evap_files = prepare_press_evap_files(var_root, evap_root)
    if evap_files is None:
        evap_files = [None] * len(press_files)
    return [(extract_id(path), path, evap_files[idx]) for idx, path in enumerate(press_files)]

def find_index_by_hour(items, hour):
    for idx, (found_hour, _) in enumerate(items):
        if found_hour == hour:
            return idx
    raise ValueError(f"Hour {hour} not found in input files")


def strip_module_prefix(state_dict):
    if not state_dict or not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}

@dataclass(frozen=True)
class RunSpec:
    checkpoint_path: Path
    model_param_path: Path
    model_config: dict
    var_name: str
    use_evap: bool
    use_static: bool
    static_data: object
    stats_path: str
    data_root: str


@dataclass(frozen=True)
class Sources:
    files: list
    evap_files: list
    items: list
    hours: list
    rel_paths: list
    static_arr: np.ndarray | None
    start_idx: int
    end_idx: int
    c_in: int
    height: int
    width: int


@dataclass(frozen=True)
class NormStats:
    mean_t: torch.Tensor
    std_eps_t: torch.Tensor
    mean_y: np.ndarray
    std_y_eps: np.ndarray


@dataclass
class TimingStats:
    prepare_time: float = 0.0
    read_time: float = 0.0
    forward_time: float = 0.0
    write_time: float = 0.0
    total_time: float = 0.0
    output_count: int = 0


def sync_cuda():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def load_run_spec(config):
    checkpoint_path = resolve_checkpoint(config.run_dir, config.checkpoint_file, config.explicit_checkpoint_path)
    params, model_param_path = load_model_params(checkpoint_path)
    model_config = require_param(params, "model_config")
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError("model_param.json key 'model_config' must be a non-empty dict.")
    data_root = config.data_root_override.strip() or require_param(params, "data_root")
    return RunSpec(
        checkpoint_path=checkpoint_path,
        model_param_path=model_param_path,
        model_config=dict(model_config),
        var_name=require_param(params, "var_name"),
        use_evap=to_bool(require_param(params, "use_evap")),
        use_static=to_bool(require_param(params, "use_static_input")),
        static_data=params.get("static_data", None),
        stats_path=require_param(params, "stats_path"),
        data_root=data_root,
    )


def prepare_sources(config, spec):
    var_root, evap_root, static_root = resolve_parflow_roots(
        spec.data_root, use_static=spec.use_static, var_name=spec.var_name, use_evap=spec.use_evap
    )
    aligned = build_aligned_var_evap_items(var_root, evap_root if spec.use_evap else None)
    files = [var_path for _, var_path, _ in aligned]
    evap_files = [evap_path for _, _, evap_path in aligned]
    items = [(hour_id, var_path) for hour_id, var_path, _ in aligned]
    hours = [hour_id for hour_id, _, _ in aligned]
    rel_paths = [Path(var_path).relative_to(var_root) for var_path in files]
    start_idx = find_index_by_hour(items, int(config.start_hour))
    end_idx = find_index_by_hour(items, int(config.end_hour))
    if end_idx < start_idx:
        raise ValueError(f"end_hour={config.end_hour} is before start_hour={config.start_hour}")
    check_contiguous_hours(hours, start_idx, end_idx)
    static_arr = read_static_stack(static_root, static_data=spec.static_data) if static_root else None
    sample = read_combined_frame(files[start_idx], evap_files[start_idx], static_arr)
    return Sources(files, evap_files, items, hours, rel_paths, static_arr, start_idx, end_idx, *sample.shape)


def check_contiguous_hours(hours, start_idx, end_idx):
    for idx in range(start_idx, end_idx):
        if hours[idx + 1] != hours[idx] + 1:
            raise ValueError(f"Missing hours between {hours[idx]} and {hours[idx + 1]}")


def validate_config(spec, sources):
    cfg = spec.model_config
    expected_in = int(require_model_key(cfg, "input_channels"))
    expected_h = int(require_model_key(cfg, "height"))
    expected_w = int(require_model_key(cfg, "width"))
    if expected_in != sources.c_in:
        raise ValueError(f"Input C mismatch: model={expected_in}, data={sources.c_in}")
    if expected_h != sources.height or expected_w != sources.width:
        raise ValueError(f"Spatial size mismatch: model=({expected_h},{expected_w}), data=({sources.height},{sources.width})")


def load_stats(spec, sources, config):
    if not spec.stats_path or not Path(spec.stats_path).exists():
        raise FileNotFoundError(f"Stats file not found: {spec.stats_path}")
    stats = np.load(spec.stats_path)
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
    if mean.shape[0] != sources.c_in or std.shape[0] != sources.c_in:
        raise ValueError(f"Stats C mismatch: stats={mean.shape[0]}, data={sources.c_in}")
    out_channels = int(require_model_key(spec.model_config, "out_channels"))
    return NormStats(
        mean_t=torch.from_numpy(mean).view(1, sources.c_in, 1, 1).float().to(DEVICE),
        std_eps_t=torch.from_numpy(std + config.eps).view(1, sources.c_in, 1, 1).float().to(DEVICE),
        mean_y=mean[:out_channels].reshape(1, -1, 1, 1),
        std_y_eps=(std[:out_channels] + config.eps).reshape(1, -1, 1, 1),
    )


def load_model(spec):
    model = PredFormer_Model(spec.model_config).to(DEVICE)
    ckpt = torch.load(spec.checkpoint_path, map_location=DEVICE)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    incompatible = model.load_state_dict(strip_module_prefix(state), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    model.eval()
    return model


def autocast_ctx(config):
    if DEVICE != "cuda" or not config.use_amp:
        return nullcontext()
    dtype = torch.float16 if config.amp_dtype.lower() == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


