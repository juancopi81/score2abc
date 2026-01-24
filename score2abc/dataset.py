from __future__ import annotations

from pathlib import Path
from typing import List

from score2abc.manifest import load_metadata_csv
from score2abc.schemas import WorkItem


def load_dataset_metadata(metadata_csv: Path, input_dir: Path) -> List[WorkItem]:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return load_metadata_csv(metadata_csv, input_dir)
