"""Freeze or verify six consumed reviewed score references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from score2abc.reference_freeze import freeze_references, verify_freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("freeze", help="create one immutable snapshot")
    create.add_argument("out_dir", type=Path)
    create.add_argument("destination", type=Path)
    verify = sub.add_parser("verify", help="verify a snapshot without changing it")
    verify.add_argument("destination", type=Path)
    verify.add_argument(
        "--out-dir", type=Path, help="also compare pinned PDFs to live source paths"
    )
    args = parser.parse_args()
    manifest = (
        freeze_references(args.out_dir, args.destination)
        if args.command == "freeze"
        else verify_freeze(args.destination, args.out_dir)
    )
    print(
        json.dumps(
            {
                "status": "created" if args.command == "freeze" else "verified",
                "destination": str(args.destination),
                "works": [work["slug"] for work in manifest["works"]],
                "file_count": len(manifest["files"]),
                "omitted_context_references": {
                    work["slug"]: work["omitted_context_references"]
                    for work in manifest["works"]
                    if work["omitted_context_references"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
