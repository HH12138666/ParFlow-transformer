"""Dataset dispatch for the supported ParFlow experiment."""

from .parflow.config import ParFlowDataConfig
from .parflow.loader import load_data as load_parflow_data


def load_data(dataname, **kwargs):
    if dataname != "parflow":
        raise ValueError(f"Dataname {dataname} is unsupported")
    return load_parflow_data(ParFlowDataConfig.from_mapping(kwargs))
