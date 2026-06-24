"""Dataset registry for VA pipeline."""

from __future__ import annotations

from typing import Type

from datasets.base import VADatasetSource
from datasets.deam import DEAMDataset
from datasets.memo2496 import Memo2496Dataset
from datasets.merp import MERPDataset

_REGISTRY: dict[str, Type[VADatasetSource]] = {
    "deam": DEAMDataset,
    "memo2496": Memo2496Dataset,
    "merp": MERPDataset,
}


def get_dataset(name: str, storage_dir: str | None = None) -> VADatasetSource:
    """Instantiate a dataset adapter by name."""
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Choose from: {list(_REGISTRY)}")
    return _REGISTRY[key](storage_dir=storage_dir)


def list_datasets() -> list[str]:
    return list(_REGISTRY.keys())
