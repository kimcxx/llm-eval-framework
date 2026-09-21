"""评测数据集：定义、加载与校验。"""
from src.datasets.loader import list_datasets, load_dataset, load_datasets
from src.datasets.schema import (
    DIMENSION_LABELS,
    DIMENSIONS,
    UNTAGGED_DIMENSION,
    UNTAGGED_DIMENSION_LABEL,
    DatasetError,
    EvalCase,
)

__all__ = [
    "DIMENSION_LABELS",
    "DIMENSIONS",
    "UNTAGGED_DIMENSION",
    "UNTAGGED_DIMENSION_LABEL",
    "DatasetError",
    "EvalCase",
    "list_datasets",
    "load_dataset",
    "load_datasets",
]
