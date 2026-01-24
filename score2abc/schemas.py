from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class WorkMetadata(BaseModel):
    title: str
    composer: str
    rhythm: str = Field(..., description="Rhythm/genre (e.g., pasillo, bambuco)")
    time_signature: Optional[str] = None
    key_hint: Optional[str] = None
    tempo_hint: Optional[str] = None

    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
    }


class WorkItem(BaseModel):
    slug: str
    pdf_path: Path
    metadata: WorkMetadata

    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
    }
