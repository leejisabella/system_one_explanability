# Research analyses

Results write-ups, one folder per experiment. The plans live in [research_docs/](../research_docs/README.md); this folder records what actually happened when they ran.

| Experiment | Folder | Status |
|---|---|---|
| E0: Jev decision characterization (Pilot 0) | [E0_jev_characterization/](E0_jev_characterization/README.md) | Done 2026-10-28: go for Experiment 1 |
| E1 Pilot 1: Jev vs. explainer (faithfulness) | [E1_pilot1/](E1_pilot1/README.md) | Done 2026-10-28: harness works; ceiling effect, so the benchmark needs policy-departure variants before the full E1 |
| E1 full run: Jev vs. explainer (faithfulness and contestability) | — | Needs the window-sweep variants and the LLM judge |
| E2: Distinguishing similar targets | — | Not started; Pilot 1 shows explanations tell the policy's story (see E1_pilot1) |
| E3: Information ablation | — | Not started; Pilot 1 is its first rung (input + choice) |
| Later experiments and stretch S1–S5 | — | Not started |
| **Refined direction (E.1xx): cascade deference** | | [research_docs/13](../research_docs/13_cascade_deference.md) |
| E.101: Jev → reviewer go/no-go pilot (Part 1) | [E101_pilot/](E101_pilot/README.md) | Done 2026-11-04: GO on Civil Comments, NO-GO on WildGuard; Jev's and the reviewer's errors overlap far beyond chance, residual deference not yet significant |
| E.102: deference beyond shared difficulty (Part 1, powered) | [E102/](E102/README.md) | Done 2026-11-04: the reviewer accepts far more of Jev's mistakes than random ones, mostly because they share blind spots; at equal difficulty about +7 points of real deference remain; two thirds of the harmful content Jev misses also gets past the reviewer |
| E.103: cheap alternatives to post-training (the Part 2 baseline) | [E103/](E103/README.md) | Done 2026-11-04: no prompt change (blind-first, two warnings, consider-the-opposite) catches more of Jev's mistakes or makes the cascade more accurate; blind-first shrinks measured deference only by accepting random mistakes more; shared blind spots untouched (91–100% accepted), so Part 2 needs training |
| E.104–E.105: Part 2's training data (Jev screen, traces, datasets) | [E104_E105/](E104_E105/README.md) | Done 2026-11-04: datasets built and checked; half of Jev's mistakes dropped out because no model reached the gold label, and those look like label disagreements, not blind spots; train as-is. Training waits for a GPU |

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
