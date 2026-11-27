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
| **All E.1xx results in one place (E.101–E.107)** | [E1xx_summary/](E1xx_summary/README.md) | Summary of Parts 1 and 2: shared blind spots, not deference, are the main problem; neither prompts nor self-training reach them |
| E.101: Jev → reviewer go/no-go pilot (Part 1) | [E101_pilot/](E101_pilot/README.md) | Done 2026-11-04: GO on Civil Comments, NO-GO on WildGuard; Jev's and the reviewer's errors overlap far beyond chance, residual deference not yet significant |
| E.102: deference beyond shared difficulty (Part 1, powered) | [E102/](E102/README.md) | Done 2026-11-04: the reviewer accepts far more of Jev's mistakes than random ones, mostly because they share blind spots; at equal difficulty about +7 points of real deference remain; two thirds of the harmful content Jev misses also gets past the reviewer |
| E.103: cheap alternatives to post-training (the Part 2 baseline) | [E103/](E103/README.md) | Done 2026-11-04: no prompt change (blind-first, two warnings, consider-the-opposite) catches more of Jev's mistakes or makes the cascade more accurate; blind-first shrinks measured deference only by accepting random mistakes more; shared blind spots untouched (91–100% accepted), so Part 2 needs training |
| E.104–E.105: Part 2's training data (Jev screen, traces, datasets) | [E104_E105/](E104_E105/README.md) | Done 2026-11-04: datasets built and checked; half of Jev's mistakes dropped out because no model reached the gold label, and those look like label disagreements, not blind spots; train as-is. Training waits for a GPU |
| E.106: SFT of the reviewer, with and without Jev's proposal (Part 2) | [E106/](E106/README.md) | Done 2026-11-04: **didn't work**. Both arms fail all five pre-registered criteria. They accept Jev's labels more, right or wrong; on hard items they give up their own right answer to Jev's wrong label more often (25% → 32%); overrides of Jev's mistakes don't rise. Treatment = control, so showing the proposal in training didn't matter |
| E.107: DPO on the reviewer's own right vs. wrong reasoning (Part 2) | [E107/](E107/README.md) | Done 2026-11-05: **didn't work either**, failing all five criteria. It landed back near the base (switching to Jev's wrong label 32% → 27%, base 25%; deference at equal difficulty still +7.8), overriding no more of Jev's mistakes than the base; the shared blind spots don't move. DPO's own loss stayed at chance |
| **Part 2, round 2 in one place (E.108–E.113)** | [Round2_summary/](Round2_summary/README.md) | Closed 2026-11-21: **a negative result**. Round 2's training (E.108–E.110) is built and tested offline but never ran; none of the four cheap checks (step 0, E.111, E.112, E.113) found a cheap route to the mistakes the reviewer can't fix alone before any GPU spend, and the user stopped distillation. $0.58 of Bedrock |
| E.108: consistency training and balanced-proposal SFT | [E108/](E108/README.md) | Done 2026-11-23: **neither arm works** (R1–R3 fail); group A deference unchanged; about $57 GPU, including about $32 idle |
| E.110: activation steering on the base reviewer | [E110/](E110/README.md) | Done 2026-11-23: a give-in direction is **found but not causal**: AUC 0.998 just before the answer, chance before the reasoning; steering it out lowers accepting Jev's wrong and right labels alike (−4.7 / −4.2 points); about $35 GPU |
| E.111: how predictable are Jev's mistakes? (free check) | [E111_jev_error_predictability/](E111_jev_error_predictability/README.md) | Done 2026-11-16: Jev's confidence ranks its own mistakes well (AUC 0.72 / 0.85 / 0.88), but flipping its least confident labels gains at most about 1 point; a router, not a fixer |
| E.114: route only Jev's low-confidence labels to the reviewer (free) | [E114_confidence_routing/](E114_confidence_routing/README.md) | Done 2026-11-21: no routing policy beats Jev alone. Reviewing the lowest-confidence 10% stops the cascade's 3.5-point loss (−0.5 [−1.6, +0.6]) at a tenth of the reviewer calls, but even on those items the reviewer breaks more right labels than it fixes. A cost result, not an accuracy result |
| E.115: WildGuard policy-text audit (prompt-engineering track, free) | [E115/](E115/README.md) | Done 2026-11-23, reviewed: **no real mismatch**. Our policy covers the annotators' rules; the "critiqued stereotype" trigger is in neither text; annotator agreement is moderate (κ 0.55 / 0.50). Clarify paragraphs written for E.116 |
| E.116: prompt-engineering development screen (WildGuard, Bedrock) | [E116/](E116/README.md) | Done 2026-11-23: **no arm passes**, so E.117 doesn't run. Clarify and grounded catch fewer of Jev's mistakes (−4.0, −5.7); worked examples trade over-flagging for letting harm through; cascade accuracy unchanged. $0.58 |
| E.119: does a stronger reviewer fix it? (gpt-oss-120b on E.102's plan) | [E119/](E119/README.md) | Done 2026-11-24: **no**. The 120b cascade is no better than Jev alone (−1.3 [−3.1, +0.3]) or the 20b cascade (+0.2); same blind spots (6% right on the 20b's never-right items), same overlap with Jev (phi 0.57), overrides fewer of Jev's mistakes (29% vs 37%). $1.70 |
| E.20: a correlation ceiling for the cascade (free re-analysis) | [E20/](E20/README.md) | Done 2026-11-24: the reviewer barely defers, it re-decides: every upstream's cascade with the 20b lands at 82.6–83.1% whatever the upstream's accuracy. Ceiling from the reviewer's own answers: +1.6 (20b) to +2.8 (120b) over Jev, in-sample; every cascade is 3–5 points below it. Overlap explains the ceiling only weakly (Spearman −0.27). $0 |
| E.21: route, don't fix; does the reasoning reveal deference? | [E21/](E21/README.md) | Done 2026-11-24: as triage, the reviewer adds nothing clear over Jev's confidence (+4.4 [−1.7, +10.0] at 10%); 55% of Jev's mistakes are silent (92% of never-right ones). A lexical monitor on the reasoning spots giving in (AUC 0.92; 0.85 vs genuine agreement); a 120b judge can't (0.41). Giving in is rare (78 reviews), so it doesn't help routing. $0.21 |
| E.20a: selective override, out of sample (free) | [E20a/](E20a/README.md) | Done 2026-11-24: **no**. Overriding Jev only when the reviewer's alone answers agree, with the rule chosen on training pools, gains −0.3 [−1.8, +1.2] (5 answers: +0.4). It stops the cascade's loss but doesn't beat Jev; E.20's ceiling is in-sample optimism (test-chosen rule +1.2, in other directions; WildGuard's GPT-4 training labels point the wrong way). $0 |
| E.22: a reviewer from another family (Mistral Large 3, WildGuard) | [E22/](E22/README.md) | Done 2026-11-24: **no**. Mistral cascade −3.4 [−5.5, −1.1] vs Jev, harmful let through 3.8% → 9.5%. It reaches Jev's shared blind spots (overrides 37% of the 20b's never-right items vs 4%; phi with Jev 0.44 vs 0.64) but is lenient: misses 30% of harmful items. $4.19 (35% over projection) |
| E.24: minimal-pair probes of the shared blind spot | [E24/](E24/README.md) | Done 2026-11-24: triggers cause over-flagging (H1 pass: Jev +16, 20b +32, 120b +19, Qwen +20, Mistral +1 points) and it's shared (H2 pass: phi Jev–20b 0.41, Jev–Mistral −0.05). Dark fiction is the strongest trigger (Jev 45%, 20b 55%). The 20b cascade adds over-flagging. Euphemism hides harm from no one (H4 fail); H3 untestable (n = 14). $1.22 |
| E.112: teacher screen for distillation | [E112/](E112/README.md) | Done 2026-11-21: **no teacher passes** the pre-registered rule (Mistral Large 3, Qwen3-235B, gpt-oss-120b high); exploratory, Mistral fixes most of Jev's WildGuard over-flagging but not the harmful items Jev misses. $0.58 with the interrupted v1 |

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
