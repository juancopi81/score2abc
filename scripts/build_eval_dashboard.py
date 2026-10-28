"""Build an offline readiness dashboard from a verified reference freeze."""

from __future__ import annotations

import argparse
from pathlib import Path

from score2abc.eval_dashboard import render_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Verified reference freeze manifest.json")
    parser.add_argument("output", type=Path, help="Destination HTML file")
    args = parser.parse_args()
    html = render_dashboard(args.manifest, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
