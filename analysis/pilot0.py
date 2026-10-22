"""Regenerate the Pilot 0 report for a saved run.

Usage:  uv run python analysis/pilot0.py results/runs/<run_id>

Thin wrapper around sysone.analysis.pilot0. For ad-hoc analysis, the run's JSONL
files load directly with pandas: pd.read_json(path, lines=True).
"""

import sys

from sysone.analysis.pilot0 import analyze_run

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    analyze_run(sys.argv[1])
    print(open(f"{sys.argv[1]}/report.md").read())
