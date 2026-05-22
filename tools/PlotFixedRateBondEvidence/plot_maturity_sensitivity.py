#!/usr/bin/env python3
"""Generate the fixed-rate bond maturity-sensitivity case-study plot."""

from __future__ import annotations

import sys
from pathlib import Path

from plot_phase6_maturity import main


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    if "--svg-out" not in sys.argv:
        sys.argv.extend([
            "--svg-out",
            str(repo_root / "docs/images/fixed-rate-bond-maturity-sensitivity.svg"),
        ])
    main()
