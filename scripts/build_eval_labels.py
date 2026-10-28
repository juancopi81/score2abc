"""Create or verify the reviewed-six evidence label sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path

from score2abc.eval_labels import INVENTORY_PATH, create_labels, verify_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("frozen_root", type=Path)
    parser.add_argument("sidecar_root", type=Path)
    parser.add_argument(
        "--inventory", type=Path, help="versioned inventory; defaults to v3 on create"
    )
    args = parser.parse_args()
    if args.action == "create":
        result = create_labels(
            args.sidecar_root, args.frozen_root, args.inventory or INVENTORY_PATH
        )
    else:
        result = verify_labels(args.sidecar_root, args.frozen_root, args.inventory)
    print(f"{args.action}: {len(result['works'])} works; {args.sidecar_root}")


if __name__ == "__main__":
    main()
