"""Immutable configuration for the ParFlow data pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParFlowDataConfig:
    data_root: str
    batch_size: int
    val_batch_size: int
    num_workers: int
    pre_seq_length: int
    aft_seq_length: int
    out_channels: int
    space_h: int | None
    space_w: int | None
    space_stride_h: int | None
    space_stride_w: int | None
    split_mode: str
    train_years: list[int] | None
    holdout_years: list[int] | None
    val_ratio_in_holdout: float
    var_name: str
    stats_path: str
    use_augment: bool
    drop_last: bool
    distributed: bool
    use_evap: bool
    use_static_input: bool
    use_val: bool
    use_extra_data: bool
    extra_manifest_path: str | None
    extra_data_root: str | None
    use_time_grouped_sampler: bool

    @classmethod
    def from_mapping(cls, values):
        model = values["model_config"]
        return cls(
            data_root=values["data_root"],
            batch_size=int(values["batch_size"]),
            val_batch_size=int(values["val_batch_size"]),
            num_workers=int(values["num_workers"]),
            pre_seq_length=int(values["pre_seq_length"]),
            aft_seq_length=int(values["aft_seq_length"]),
            out_channels=int(model["out_channels"]),
            space_h=values.get("space_h"),
            space_w=values.get("space_w"),
            space_stride_h=values.get("space_stride_h"),
            space_stride_w=values.get("space_stride_w"),
            split_mode=values.get("split_mode", "ratio"),
            train_years=values.get("train_years"),
            holdout_years=values.get("holdout_years"),
            val_ratio_in_holdout=float(values.get("val_ratio_in_holdout", 0.5)),
            var_name=values.get("var_name", "press"),
            stats_path=values["stats_path"],
            use_augment=bool(values.get("use_augment", False)),
            drop_last=bool(values.get("drop_last", False)),
            distributed=bool(values.get("distributed", False)),
            use_evap=bool(values.get("use_evap", True)),
            use_static_input=bool(values.get("use_static_input", True)),
            use_val=bool(values.get("use_val", False)),
            use_extra_data=bool(values.get("use_extra_data", False)),
            extra_manifest_path=values.get("extra_manifest_path"),
            extra_data_root=values.get("extra_data_root"),
            use_time_grouped_sampler=bool(values.get("use_time_grouped_sampler", True)),
        )
