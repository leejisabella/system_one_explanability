# Research analyses

Results write-ups, one folder per experiment. The plans live in [research_docs/](../research_docs/README.md); this folder records what actually happened when they ran.

| Experiment | Folder | Status |
|---|---|---|
| E0: Jev decision characterization (Pilot 0) | [E0_jev_characterization/](E0_jev_characterization/README.md) | Done 2026-10-28: go for Experiment 1 |
| E1: Faithfulness vs. contestability (Pilot 1 first) | — | Harness built; waits on AWS |
| E2–E6, stretch S1–S5 | — | Not started |

## What goes in each folder

```
E<n>_<short_name>/
  README.md        the write-up: question, setup, results, what it means, caveats, how to reproduce
  make_figures.py  regenerates figures/ and data/ from the saved runs (no API calls)
  figures/         PNGs referenced from README.md
  data/            small derived tables and each run's summary.json and report.md
```

- **Every number traces to a run.** Name the `run_id`, the benchmark and its hash, and the model versions. Runs live under `results/runs/<run_id>/`, which is gitignored and kept locally only, so `data/` keeps the derived tables that the write-up and figures depend on.
- **Figures are regenerated, never edited by hand.** Run `uv run --group analysis python research_analyses/<folder>/make_figures.py`. Matplotlib lives in the optional `analysis` dependency group, so the core harness doesn't need it.
- **Each figure has its numbers in a table next to it** in the write-up, so nothing is read from a chart alone.
- **Colors** come from the reference palette (blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a` on `#fcfcfb`), checked for colorblind separation; aqua is below 3:1 contrast, so values are always labelled directly.
- **Say what was decided** because of the results (a go / no-go, a benchmark change), and link the decision where it is recorded.
