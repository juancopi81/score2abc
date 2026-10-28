"""Build an offline configuration comparison dashboard from a versioned report."""

from __future__ import annotations

import argparse
from pathlib import Path

from score2abc.comparison_dashboard import render_comparison_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Versioned comparison report JSON")
    parser.add_argument("output", type=Path, help="Destination HTML file")
    args = parser.parse_args()
    html = render_comparison_dashboard(args.report, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
