from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from score2abc.chord_ocr import ChordDetection, ChordExtractionRequest
from score2abc.schemas import WorkItem, WorkMetadata

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "record_vlm_fixtures.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("record_vlm_fixtures", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
record_vlm_fixtures = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(record_vlm_fixtures)


class _FakeOCR:
    model_id = "fake-model"
    prompt_version = "fake-prompt"

    def __init__(self) -> None:
        self.requests: list[ChordExtractionRequest] = []

    def extract(self, request: ChordExtractionRequest) -> list[ChordDetection]:
        self.requests.append(request)
        return [
            ChordDetection(
                symbol_raw="C",
                symbol="C",
                x_fraction=0.5,
                confidence=0.9,
                band=request.band,
            )
        ]


def test_record_for_work_can_limit_to_below_band(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    systems_dir = out_dir / "test-work" / "systems"
    systems_dir.mkdir(parents=True)
    (systems_dir / "chord_region_above_001.png").write_bytes(b"above")
    (systems_dir / "chord_region_below_001.png").write_bytes(b"below")

    item = WorkItem(
        slug="test-work",
        pdf_path=tmp_path / "test.pdf",
        metadata=WorkMetadata(title="Test", composer="Composer", rhythm="pasillo"),
    )
    ocr = _FakeOCR()

    written, skipped = record_vlm_fixtures._record_for_work(
        item,
        out_dir=out_dir,
        fixtures_dir=tmp_path / "fixtures",
        ocr=ocr,
        selected_bands=("below",),
        force=False,
        logger=logging.getLogger("test"),
    )

    assert written == 1
    assert skipped == 0
    assert [request.band for request in ocr.requests] == ["below"]
