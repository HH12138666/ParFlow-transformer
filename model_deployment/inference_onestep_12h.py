"""Run PredFormer one-step 12h pressure inference.

This script is intentionally parallel to model_deployment.inference, but fixes the
prediction horizon to one 12h block by default. It writes predicted pressure PFBs;
WTD conversion should be done by the existing downstream WTD scripts.
"""

import argparse
import time

from .common import (
    InferenceConfig,
    TimingStats,
    load_model,
    load_run_spec,
    load_stats,
    prepare_sources,
    validate_config,
)
from .engine import print_timing_summary, run_inference


DEFAULT_ONE_STEP_HOURS = 12


def parse_cli():
    parser = argparse.ArgumentParser(description="Run one-step 12h ParFlow PredFormer pressure inference")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-file", default="latest.pth")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-dir", default="inference_data/press_onestep_12h")
    parser.add_argument("--run-name", default="predformer_onestep_12h")
    parser.add_argument("--start-hour", type=int, required=True)
    parser.add_argument("--end-hour", type=int, required=True)
    parser.add_argument("--patch-batch-size", type=int, default=28)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def config_from_cli(args):
    return InferenceConfig(
        run_dir=args.run_dir,
        checkpoint_file=args.checkpoint_file,
        explicit_checkpoint_path=args.checkpoint_path,
        data_root_override=args.data_root,
        output_dir=args.output_dir,
        run_param=args.run_name,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        use_rollout=True,
        rollout_hours=DEFAULT_ONE_STEP_HOURS,
        use_amp=args.use_amp,
        patch_batch_size=args.patch_batch_size,
    )


def prepare_inference(config):
    timing = TimingStats()
    prepare_start = time.perf_counter()
    spec = load_run_spec(config)
    print(f"[checkpoint] {spec.checkpoint_path}")
    print(f"[model_param] {spec.model_param_path}")
    print(f"[data_root] {spec.data_root}")
    sources = prepare_sources(config, spec)
    validate_config(spec, sources)
    stats = load_stats(spec, sources, config)
    model = load_model(spec)
    timing.prepare_time = time.perf_counter() - prepare_start
    return spec, sources, stats, model, timing


def main():
    config = config_from_cli(parse_cli())
    total_start = time.perf_counter()
    spec, sources, stats, model, timing = prepare_inference(config)
    print(f"[mode] one-step pressure inference, horizon={DEFAULT_ONE_STEP_HOURS}h")
    run_inference(config, spec, sources, stats, model, timing)
    timing.total_time = time.perf_counter() - total_start
    print_timing_summary(timing)
    print(f"One-step 12h pressure inference done. Elapsed time: {timing.total_time:.2f}s")


if __name__ == "__main__":
    main()
