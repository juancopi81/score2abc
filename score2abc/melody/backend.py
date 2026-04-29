from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from score2abc.musicxml import parse_musicxml_events
from score2abc.schemas import WorkItem

INTERMEDIATE_MUSICXML_FILENAME = "musicxml.xml"
DEFAULT_MUSICXML_SOURCE_DIR = Path("dataset/musicxml")
_FIXTURE_EXTENSIONS: tuple[str, ...] = (".musicxml", ".xml")
_HOMR_OUTPUT_EXTENSIONS: tuple[str, ...] = (".musicxml", ".xml")
HOMR_INPUT_MODES: tuple[str, ...] = ("page", "deskewed-page", "systems")
DEFAULT_HOMR_COMMAND = "homr"
DEFAULT_HOMR_TIMEOUT_SECONDS = 900
_SYSTEM_COLLAGE_FILENAME = "systems_collage.png"
_SYSTEM_COLLAGE_SEPARATOR_PIXELS = 24


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
        input_mode: str = "page",
        timeout_seconds: int = DEFAULT_HOMR_TIMEOUT_SECONDS,
    ) -> None:
        if input_mode not in HOMR_INPUT_MODES:
            raise ValueError(f"Unsupported homr input mode: {input_mode}")
        self._command = command
        self._input_mode = input_mode
        self._timeout_seconds = timeout_seconds

    def produce_musicxml(
        self,
        *,
        item: WorkItem,
        work_dir: Path,
    ) -> MusicXMLProduceResult | None:
        homr_dir = work_dir / "intermediate" / "homr"
        homr_dir.mkdir(parents=True, exist_ok=True)

        input_path = self._prepare_input_image(work_dir=work_dir, homr_dir=homr_dir)

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

    def _prepare_input_image(self, *, work_dir: Path, homr_dir: Path) -> Path:
        if self._input_mode == "page":
            source_path = self._select_single_image(
                sorted((work_dir / "pages").glob("page_*.png")),
                missing_message=f"No rendered page images found under {work_dir / 'pages'}",
                multiple_message="homr page input currently supports one rendered page per work",
            )
            input_path = homr_dir / source_path.name
            shutil.copyfile(source_path, input_path)
            return input_path

        if self._input_mode == "deskewed-page":
            source_path = self._select_single_image(
                sorted((work_dir / "systems").glob("page_*_deskewed.png")),
                missing_message=f"No deskewed page images found under {work_dir / 'systems'}",
                multiple_message="homr deskewed-page input currently supports one page per work",
            )
            input_path = homr_dir / source_path.name
            shutil.copyfile(source_path, input_path)
            return input_path

        if self._input_mode == "systems":
            return self._build_systems_collage(work_dir=work_dir, homr_dir=homr_dir)

        raise ValueError(f"Unsupported homr input mode: {self._input_mode}")

    def _select_single_image(
        self,
        paths: list[Path],
        *,
        missing_message: str,
        multiple_message: str,
    ) -> Path:
        if not paths:
            raise MusicXMLBackendError(missing_message)
        if len(paths) > 1:
            raise MusicXMLBackendError(f"{multiple_message}; found {len(paths)} images")
        return paths[0]

    def _build_systems_collage(self, *, work_dir: Path, homr_dir: Path) -> Path:
        system_paths = sorted((work_dir / "systems").glob("system_*.png"))
        if not system_paths:
            raise MusicXMLBackendError(f"No system crops found under {work_dir / 'systems'}")

        images: list[Image.Image] = []
        try:
            for path in system_paths:
                images.append(Image.open(path).convert("RGB"))

            max_width = max(image.width for image in images)
            total_height = sum(image.height for image in images)
            total_height += _SYSTEM_COLLAGE_SEPARATOR_PIXELS * (len(images) - 1)
            collage = Image.new("RGB", (max_width, total_height), color="white")

            y = 0
            for image in images:
                collage.paste(image, (0, y))
                y += image.height + _SYSTEM_COLLAGE_SEPARATOR_PIXELS

            output_path = homr_dir / _SYSTEM_COLLAGE_FILENAME
            output_path.unlink(missing_ok=True)
            collage.save(output_path)
            return output_path
        finally:
            for image in images:
                image.close()

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
    homr_input: str = "page",
) -> MusicXMLBackend:
    """Pick a MusicXMLBackend."""
    if backend == "fixture":
        return FixtureMusicXMLBackend(source_dir=source_dir)
    if backend == "homr":
        return HomrMusicXMLBackend(command=homr_command, input_mode=homr_input)
    raise ValueError(f"Unsupported MusicXML backend: {backend}")
