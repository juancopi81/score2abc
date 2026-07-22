"""Run and freeze the prepared fourth-score truth-blind melody inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_third_score_heldout_inference as heldout  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_manifest", type=Path)
    parser.add_argument("--model-dir", type=Path, default=heldout.DEFAULT_MODEL_DIR)
    parser.add_argument("--inference-dirname", default=heldout.DEFAULT_INFERENCE_DIRNAME)
    parser.add_argument("--no-freeze", action="store_true")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        result = heldout.materialize_third_score_inference(
            args.prepared_manifest,
            model_dir=args.model_dir,
            inference_dirname=args.inference_dirname,
        )
        if not args.no_freeze:
            result["freeze"] = heldout.freeze_inference(
                args.prepared_manifest,
                inference_dir=Path(result["inference_dir"]),
                model_dir=args.model_dir,
            )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result["runtime_seconds"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
