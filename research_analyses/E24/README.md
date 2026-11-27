# E.24: minimal-pair probes of the shared blind spot

**Status:** done 2026-11-24.
- **Run:** `e24-v1-2026-11-24`, $1.22 (Bedrock $1.18, TypeSafe $0.04), against a projected $1.92. That was 9,720 calls (1,620 Jev and 8,100 Bedrock), with 2 replies unparsed (both gpt-oss-120b).
- **Pilot:** `e24-pilot-2026-11-24`, $0.04.

**Plan:** [research_docs/17](../../research_docs/17_second_batch.md#e24-minimal-pair-probes-of-the-shared-blind-spot-about-2). **Hypotheses, fixed before the run:** [e24.yaml](../../configs/experiments/e24.yaml). **Probe set:** [benchmarks/probes/minimal_pairs_v1](../../benchmarks/probes/minimal_pairs_v1/manifest.json), 180 triplets written and checked by subagents before the run.

**Result:**
- **H1 passes: specific triggers cause over-flagging.** Adding the trigger to a benign prompt raises the flag rate:
  - Jev, by 16 points [8, 25];
  - gpt-oss-20b, by 32 [21, 44];
  - gpt-oss-120b, by 19;
  - Qwen3-32B, by 20.
- **H2 passes: it is shared.** On the trigger items, Jev's flags overlap with the 20b's (phi 0.41) and the 120b's (0.48).
- **Mistral Large 3 is almost immune:** its trigger effect is +1 [0, +4], and its flags don't overlap with Jev's (phi −0.05). It still flags 100% of the harmful versions.
- **The strongest trigger is a dark fiction frame** with no actionable request: Jev flags 45% of those benign prompts and the 20b 55%.
- **H3 and H4 fail:**
  - Jev flagged too few trigger items (14) to show that the cascade keeps them more than usual.
  - Euphemism doesn't make any model miss a harmful request.

## Question

Step 0 traced the shared over-flagging on WildGuard to a few kinds of trigger. Do those triggers *cause* the errors, in Jev and gpt-oss alike, and does a model from another family resist them? Counting mistakes on natural items can't separate a trigger from everything else in the text. Controlled pairs can.

## Setup

- **Probe set:** 9 families of 20 triplets. Each triplet is:
  - (a) a neutral benign item;
  - (b) the same item with one trigger added, still benign;
  - (c) a minimal edit of (b) that is truly harmful.
- **P5 is the let-through test:** (b) is a plainly harmful request and (c) the same request in euphemism.
- **How the items were made:**
  - written by subagents from the policies only, with no test item read;
  - checked for overlap with every pool, test and training;
  - checked twice for minimality;
  - labelled by two blind labellers, who agreed with the intended label on all 540 items.
- **Models,** each deciding alone:
  - Jev, pinned to jev-1.14.0 as in every cascade run, ×3;
  - gpt-oss-20b ×5;
  - gpt-oss-120b ×5;
  - Mistral Large 3 ×3, writing its reasoning first as in E.22;
  - Qwen3-32B ×1.

  Plus one 20b cascade review of Jev's label per item.
- **Flag:** a model flags an item if most of its samples give the positive label (harmful or toxic). Intervals are paired bootstraps over triplets.

## Results

**Hypotheses:**

| | Prediction | Result |
|---|---|---|
| H1 | the trigger raises the flag rate (P1–P4, b − a), for Jev and the 20b | **Pass.** Jev +16 [+8, +25], 20b +32 [+21, +44]. Also 120b +19 [+10, +29], Qwen +20 [+11, +29], Mistral +1 [0, +4]. |
| H2 | Jev's and the 20b's trigger flags overlap (phi > 0.3), more than Jev's and Mistral's | **Pass.** Jev–20b 0.41, Jev–120b 0.48, Jev–Qwen 0.26, Jev–Mistral −0.05. |
| H3 | the cascade keeps Jev's trigger flags at least as often as it accepts Jev's natural mistakes | **Fail.** It keeps 64% (9 of 14) against 70% on E.102's WildGuard prompts. n = 14 is too small to say much. |
| H4 | euphemism lowers the flag rate on harmful requests (P5, c − b) | **Fail.** All models flag 95–100% of the euphemistic versions. |

H1 holds without the 5 near-fail triplets and the labeller-unsure items: Jev +14 [+6, +23], 20b +30 [+18, +42].

**Flag rate on the benign trigger items (b), by family.** In every family, every model flags 95–100% of the harmful versions (c), and 0–10% of the neutral versions (a).

| Family (trigger) | Jev | gpt-oss-20b | gpt-oss-120b | Qwen3-32B | Mistral | 20b cascade |
|---|---|---|---|---|---|---|
| P1 harm word, harmless setting | 5% | 45% | 25% | 35% | 5% | 35% |
| P2 stereotype quoted or critiqued | 5% | 25% | 5% | 20% | 0% | 25% |
| P3 education or safety question | 15% | 20% | 15% | 0% | 0% | 20% |
| P4 dark fiction, no actionable ask | **45%** | **55%** | 40% | 25% | 0% | 35% |
| R1 refusal of a harmful request | 0% | 0% | 0% | 0% | 0% | 0% |
| R2 response discusses harm critically | 5% | 20% | 5% | 5% | 0% | 25% |
| C1 undirected mild profanity | 0% | 0% | 5% | 0% | 0% | 10% |
| C2 neutral identity mention | 0% | 0% | 0% | 0% | 0% | 0% |

Full table, with the trigger effect and sensitivity per family and model: [data/report.md](data/report.md).

## What it means

1. **The shared blind spot has a cause we can name.**
   - Benign prompts with dark fiction, harm vocabulary in a harmless setting, or a safety question get flagged as harmful.
   - Jev, both gpt-oss models and Qwen all do this, and they flag the *same* items (phi 0.41–0.48 with Jev).
   - This is the mechanism behind step 0's reading of WildGuard's never-right items, now shown by controlled edits rather than by reading.
2. **The cascade makes this worse, not better.**
   - On the benign trigger prompts (P1–P4), Jev flags 18% on average. The 20b alone flags 36%, and the 20b cascade 29%.
   - Reviewing Jev with a reviewer that has the same blind spot, more strongly, adds over-flagging. On R2 (responses that discuss harm critically), the cascade flags 25% against Jev's 5%.
3. **Mistral Large 3 doesn't have this blind spot.**
   - It flags 0–5% of the benign trigger prompts and 100% of the harmful ones.
   - Its flags don't overlap with Jev's.
   - This fits E.112's lead, where Mistral fixed 85 of 112 of Jev's over-flagged WildGuard items, and it is what E.22 tests on real items.
4. **Not every candidate trigger works.**
   - Refusals of harmful requests (R1) fool nobody.
   - Mild profanity (C1) and identity mentions (C2) don't trigger any model.
   - Euphemism doesn't hide harm from any model (H4).

   The blind spot is specific: dark framing and harm vocabulary in benign prompts. It is not a general sensitivity to surface words.
5. **For explainability:** a cheap counterfactual probe set exposes *which* features drive a black box's mistakes, and *which other model* shares them. That is the kind of evidence the project's main question asks for.

## Caveats

- **The probes are synthetic and short.** They show that the triggers can cause errors, not how often each trigger occurs in WildGuard's real mistakes.
- **20 triplets per family** give wide per-family intervals. The hypotheses pool P1–P4 (80 triplets).
- **The labels are by construction,** confirmed by two blind subagent labellers, not by people. C1's triggers sit on the policy's obscenity line (all 20 flagged as uncertain by a labeller), but no model flagged them anyway.
- **Mistral used E.22's protocol, writing its reasoning first.** That differs from the others' native reasoning or none.
- **H3 rests on 14 items:** Jev rarely flags the trigger items.

## Reproduce

```
uv run python -m sysone.experiments.e24 --report-only --run-id e24-v1-2026-11-24
```

- **Code:** `sysone.experiments.e24`, `sysone.analysis.e24`.
- **Probe build:** `scripts/build_probes_e24.py`.
- **Copied from the run:** `data/report.md`, `data/summary.json`, `data/manifest.json`.
