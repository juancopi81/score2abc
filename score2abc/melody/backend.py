from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from score2abc.musicxml import parse_musicxml_events
from score2abc.schemas import WorkItem

INTERMEDIATE_MUSICXML_FILENAME = "musicxml.xml"
DEFAULT_MUSICXML_SOURCE_DIR = Path("dataset/musicxml")
_FIXTURE_EXTENSIONS: tuple[str, ...] = (".musicxml", ".xml")
_HOMR_OUTPUT_EXTENSIONS: tuple[str, ...] = (".musicxml", ".xml")
DEFAULT_HOMR_COMMAND = "homr"
DEFAULT_HOMR_TIMEOUT_SECONDS = 900


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


class HomrMusicXMLBackend:
    """Run the external ``homr`` CLI and normalize its MusicXML output path.

    homr is intentionally treated as an optional system command, not a Python
    dependency of score2abc. This keeps default and CI runs hermetic while still
    letting local OMR experiments feed the existing MusicXML pipeline.
    """

    name = "homr"

    def __init__(
        self,
        *,
        command: str = DEFAULT_HOMR_COMMAND,
        timeout_seconds: int = DEFAULT_HOMR_TIMEOUT_SECONDS,
    ) -> None:
        self._command = command
        self._timeout_seconds = timeout_seconds

    def produce_musicxml(
        self,
        *,
        item: WorkItem,
        work_dir: Path,
    ) -> MusicXMLProduceResult | None:
        page_path = self._select_page_image(work_dir)
        homr_dir = work_dir / "intermediate" / "homr"
        homr_dir.mkdir(parents=True, exist_ok=True)

        input_path = homr_dir / page_path.name
        shutil.copyfile(page_path, input_path)

        output_path = work_dir / "intermediate" / INTERMEDIATE_MUSICXML_FILENAME
        output_path.unlink(missing_ok=True)
        for candidate in self._stem_output_candidates(input_path):
            candidate.unlink(missing_ok=True)

        before = set(homr_dir.glob("*"))
        command = [*shlex.split(self._command), str(input_path.resolve())]

        try:
            completed = subprocess.run(
                command,
                cwd=homr_dir,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise MusicXMLBackendError(
                "homr command not found. Install homr separately or rerun with "
                "--musicxml-backend fixture."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MusicXMLBackendError(
                f"homr timed out after {self._timeout_seconds} seconds for {item.slug}"
            ) from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or "no output"
            raise MusicXMLBackendError(
                f"homr failed for {item.slug} with exit code {completed.returncode}: {details}"
            )

        produced = self._find_homr_output(
            homr_dir=homr_dir,
            input_path=input_path,
            before=before,
        )
        if produced is None:
            raise MusicXMLBackendError(f"homr did not produce a MusicXML file for {item.slug}")

        try:
            shutil.copyfile(produced, output_path)
            parse_musicxml_events(output_path)
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            raise MusicXMLBackendError(
                f"homr MusicXML failed validation ({produced}): {exc}"
            ) from exc

        return MusicXMLProduceResult(output_path=output_path, source_path=input_path)

    def _select_page_image(self, work_dir: Path) -> Path:
        page_paths = sorted((work_dir / "pages").glob("page_*.png"))
        if not page_paths:
            raise MusicXMLBackendError(f"No rendered page images found under {work_dir / 'pages'}")
        if len(page_paths) > 1:
            raise MusicXMLBackendError(
                "homr backend currently supports one rendered page per work; "
                f"found {len(page_paths)} pages under {work_dir / 'pages'}"
            )
        return page_paths[0]

    def _find_homr_output(
        self,
        *,
        homr_dir: Path,
        input_path: Path,
        before: set[Path],
    ) -> Path | None:
        for candidate in self._stem_output_candidates(input_path):
            if candidate.exists():
                return candidate

        produced = [
            path
            for path in sorted(homr_dir.glob("*"))
            if path not in before and path.suffix.lower() in _HOMR_OUTPUT_EXTENSIONS
        ]
        return produced[0] if produced else None

    def _stem_output_candidates(self, input_path: Path) -> list[Path]:
        return [input_path.with_suffix(extension) for extension in _HOMR_OUTPUT_EXTENSIONS]


def build_musicxml_backend(
    *,
    source_dir: Path,
    backend: str = "fixture",
    homr_command: str = DEFAULT_HOMR_COMMAND,
) -> MusicXMLBackend:
    """Pick a MusicXMLBackend."""
    if backend == "fixture":
        return FixtureMusicXMLBackend(source_dir=source_dir)
    if backend == "homr":
        return HomrMusicXMLBackend(command=homr_command)
    raise ValueError(f"Unsupported MusicXML backend: {backend}")
