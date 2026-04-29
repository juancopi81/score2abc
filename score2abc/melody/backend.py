from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from score2abc.musicxml import parse_musicxml_events
from score2abc.schemas import WorkItem

INTERMEDIATE_MUSICXML_FILENAME = "musicxml.xml"
DEFAULT_MUSICXML_SOURCE_DIR = Path("dataset/musicxml")
_FIXTURE_EXTENSIONS: tuple[str, ...] = (".musicxml", ".xml")


class MusicXMLBackendError(RuntimeError):
    """Raised when a MusicXML backend has a source but cannot produce a usable file."""


@dataclass(frozen=True)
class MusicXMLProduceResult:
    """Returned by a backend after successfully writing the intermediate MusicXML."""

    output_path: Path
    source_path: Path


class MusicXMLBackend(Protocol):
    """Protocol implemented by every MusicXML production backend.

    Returning ``None`` means "no source available" and is treated as a skipped
    stage. Failure to copy or validate a source MusicXML must raise
    ``MusicXMLBackendError`` so the pipeline can fail the work item rather than
    silently fall back to stub notes.
    """

    @property
    def name(self) -> str: ...

    def produce_musicxml(
        self,
        *,
        item: WorkItem,
        work_dir: Path,
    ) -> MusicXMLProduceResult | None: ...


class FixtureMusicXMLBackend:
    """Copy a pre-existing MusicXML fixture into ``work_dir/intermediate/``.

    Looks for ``<source_dir>/<slug>.musicxml`` (and ``.xml`` as a fallback),
    copies it to ``intermediate/musicxml.xml``, and validates by parsing.
    Used until a real OMR engine produces MusicXML automatically.
    """

    name = "fixture"

    def __init__(self, source_dir: Path) -> None:
        self._source_dir = source_dir

    def produce_musicxml(
        self,
        *,
        item: WorkItem,
        work_dir: Path,
    ) -> MusicXMLProduceResult | None:
        source = self._find_source(item.slug)
        if source is None:
            return None

        output_path = work_dir / "intermediate" / INTERMEDIATE_MUSICXML_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copyfile(source, output_path)
        except OSError as exc:
            output_path.unlink(missing_ok=True)
            raise MusicXMLBackendError(
                f"Failed to copy MusicXML source {source} -> {output_path}: {exc}"
            ) from exc

        try:
            parse_musicxml_events(output_path)
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            raise MusicXMLBackendError(
                f"MusicXML source failed validation ({source}): {exc}"
            ) from exc

        return MusicXMLProduceResult(output_path=output_path, source_path=source)

    def _find_source(self, slug: str) -> Path | None:
        for extension in _FIXTURE_EXTENSIONS:
            candidate = self._source_dir / f"{slug}{extension}"
            if candidate.exists():
                return candidate
        return None


def build_musicxml_backend(*, source_dir: Path) -> MusicXMLBackend:
    """Pick a MusicXMLBackend. Currently only the fixture backend is wired."""
    return FixtureMusicXMLBackend(source_dir=source_dir)
