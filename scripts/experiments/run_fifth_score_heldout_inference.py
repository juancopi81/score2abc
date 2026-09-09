"""Run and freeze the prepared Coqueteos fifth-score melody inference."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_third_score_heldout_inference as heldout  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return heldout.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
