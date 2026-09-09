"""Evaluate the sealed Gato'e Fique fourth-score gate exactly once.

This wrapper uses the shared sealed heldout evaluator, which verifies all
prepared and frozen hashes before opening the user-supplied MusicXML.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import evaluate_frozen_third_score_heldout as heldout  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return heldout.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
