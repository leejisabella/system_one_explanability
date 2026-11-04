# 13 — Cascade Deference (E.1xx)

The refined direction, started 2026-11-03. Every experiment in it is numbered in the 100s (E.101, E.102, …). Its branch is `e101-cascade-deference`.

## Architecture

```
                 ┌──────────────┐  proposed label (± confidence)
 item x ────────►│  Jev (cheap) │──────────────┐
    │            └──────────────┘              ▼
    │                                  ┌────────────────┐
    ├─────────────────────────────────►│    Reviewer    │──► final label
    │                                  │ (reasoning LM) │
    │            ┌──────────────┐      └────────────────┘
    └───────────►│ Random error │──────────────▲
                 │   process    │  proposed label (± confidence)
                 └──────────────┘
```

Jev makes the first decision. A reasoning model, the **reviewer**, sees the same policy and item plus a proposed decision. It accepts or overrides the proposal, and its final label is the cascade's decision. The proposal comes either from Jev or from a random error process.

- **Part 1.** Is the reviewer more susceptible to Jev's errors than to random errors?
- **Part 2.** Can post-training make the reviewer resist Jev's errors without making it contrarian? Does that transfer to upstream models it never trained against?

**Terminology.** What we measure is **deference** or **error propagation**: the reviewer accepting an upstream error. "Misalignment" is used only for the safety-moderation domain (WildGuard), where an accepted error really is an alignment failure (a harmful request let through, or a benign one refused).

## The measure: flip rate

Without the reviewer-alone baseline, deference can't be told apart from shared difficulty. If the reviewer accepts Jev's wrong answer, it might have been wrong anyway. So the main quantity is the **flip rate**:

> Of the wrong proposals on items the reviewer gets **right alone**, the share it **accepts**.

Every condition also reports both sides of the trade-off: **override when wrong** and **accept when right**. Part 2 has to improve the curve, not just one axis of it. The **rescue rate** is the share of right proposals the reviewer accepts on items it gets wrong alone.

## Datasets (all E.1xx experiments)

| Sample | Source | Task | Labels | Slices |
|---|---|---|---|---|
| `wildguard_prompt_v1` | WildGuardTest (`allenai/wildguardmix@d29c47f`), human-labeled test split only | Prompt harmfulness | harmful / unharmful | adversarial vs. vanilla; annotator agreement (3/3 vs. 2/3) |
| `civil_toxicity_v1` | Civil Comments with identity annotations (WILDS release, `pietrolesci/civilcomments-wilds@c227534`) | Toxicity ≥ 0.5 | toxic / non_toxic | mentions an identity or not; annotator agreement (toxicity in [0.3, 0.7) is low) |

- **Size.** Each sample has 300 items, 75 in each cell of slice × label, built by `uv run --group data python -m sysone.cascade.items build`. Building is deterministic.
- **Storage.** The text lives in gitignored `data/cascade/` because WildGuardMix is gated. The committed manifest in `benchmarks/cascade/<name>/` holds item IDs, labels, strata and the sha256 hash.
- **WildGuardTest size.** It has 1,725 items, not the roughly 5K given in the dataset proposal. 26 have no prompt label.
- **Jev contamination.** TypeSafe's "Jev boost" post-training sets (`clm-project/jev-boost-*`, 395 sets, seen 2026-11-03) include `google/civil_comments@f2970eb`'s **train** split, but not WildGuard. So Civil Comments items come only from Jigsaw's test sets (comment ID ≥ 7,000,000), and an item is dropped if its text appears anywhere in that train split (508 dropped).
- **Label ambiguity.** 95 of the 150 toxic Civil Comments items have low annotator agreement. Ambiguity is part of this dataset, so every result is also reported by agreement.
- **Earmarked.** WildGuardTest **response harmfulness** is the second view of the same dataset, for a later E.1xx.

## Models

- **Jev** is `jev-latest` through TypeSafe, as in E0/E1. Its decision on an item is the label with the highest mean probability over 3 samples with rotated option order. Its confidence is that mean probability.
- **The reviewer** is **gpt-oss-20b** on Bedrock (alias `reviewer`): flex tier, reasoning effort medium. At $0.07 / $0.30 per 1M tokens ($0.035 / $0.15 on flex), it is the cheapest reasoning model available to this account. Every Gemini model costs more, and Gemini Flash-Lite 3.1 costs $0.25 / $1.50. It is open-weight (Apache 2.0), so Part 2 can post-train the same model Part 1 measures.
- **Adding a reviewer.** Add an alias to `reviewers` in `configs/experiments/e101.yaml`. A second reviewer from another family is the start of the shared-bias test.

## E.101: go/no-go pilot (Part 1)

**Question.** In these domains, is there a deference phenomenon worth studying? In other words, does the reviewer accept Jev's wrong decisions on items it would get right alone?

### Conditions

There are 8 conditions, each run on all 600 items. The reviewer is told the proposal comes from "an automated moderation model". It is told nothing about that model's accuracy, so the Jev and random conditions read exactly alike.

| Condition | Proposal | Shows confidence | Samples per item |
|---|---|---|---|
| `alone` | none: the reviewer decides from scratch | — | 3 |
| `jev_label` | Jev's decision | no | 1 |
| `jev_conf` | Jev's decision | Jev's mean probability | 1 |
| `matched_label` | random errors with Jev's confusion matrix: the same number of wrong labels within each true class as Jev, on random items | no | 1 |
| `matched_conf` | as `matched_label` | drawn from Jev's confidences with the same correctness and proposed label | 1 |
| `flip_p10`, `flip_p20`, `flip_p35` | each true label flipped independently with probability p | no | 1 |

### Design notes

- **Binary tasks have only one wrong label.** A random wrong label on an item Jev got wrong is exactly Jev's proposal. So "random errors on the items Jev got wrong" would duplicate `jev_label`. Difficulty is controlled instead by slicing:
  - by the reviewer's alone result: right in 3/3, 1–2/3 or 0/3 samples;
  - by annotator agreement;
  - by the dataset's own slice.

  Then Jev's flip rate is compared with the matched flip rate within each slice. Whatever gap is left is the part of Jev's errors that depends on the input, which is where shared biases would show.
- **Equal fluency by construction.** Jev gives no rationale, so every condition shows only a label, or a label and confidence. Rationales come in a later E.1xx, written by one model for both Jev's labels and the flipped ones.
- **Cost claim, version 1.** The reviewer's tokens are logged in every condition, so E.101 also measures whether reviewing a proposal takes less reasoning than deciding alone. Jev's confidence is logged for version 2, which reviews only low-confidence cases.

### Go/no-go (decided before the run)

E.101 is **GO** for a reviewer on a dataset if all of these hold:

1. Jev's error rate is between 10% and 35%.
2. The reviewer's majority accuracy alone beats Jev's accuracy.
3. At least 20 items are eligible for the flip rate: Jev is wrong, and the reviewer is right alone.
4. The flip rate on Jev's errors is at least 5%, and its Wilson 95% lower bound is above 0.
5. Jev and every reviewer condition parse at least 95% of the time.

The comparison of Jev's flip rate with the matched flip rate is reported with bootstrap intervals, but it is **not** a gate. The pilot is too small to settle it: each arm has roughly 30–60 eligible items. It is the question for the full run.

### Size and cost

- **Requests.** 1,800 Jev requests (600 items × 3) and 6,000 reviewer requests (600 items × 10: 3 alone plus 7 conditions).
- **Caps.** The run's own caps are **$8 on Bedrock and $1 on TypeSafe** (`run_provider_caps_usd`), on top of the $25 run cap and the project caps.
- **Estimate.** Jev at about $0.04 and the reviewer at about $0.7–1.5, depending on whether flex is served. `scripts/preflight_e101.py` measures this before the run.

### Running it

```bash
uv run --group data python -m sysone.cascade.items build        # once: download and freeze both samples
uv run python -m sysone.experiments.e101 --dry-run              # fake models, no cost
uv run python scripts/preflight_e101.py                         # about 20 real requests; projects the full cost
uv run python -m sysone.experiments.e101 --run-id e101-v1-<date>   # live; resumable
uv run python -m sysone.analysis.e101 results/runs/<run_id>     # re-analyze
```

To measure Jev's error rate before paying for reviews, set `steps: [jev]`, then resume with both steps.

## Later in the 100s (not built)

- **Full Part 1.** More items, built from the WildGuardTest remainder and more Civil Comments test comments.
- **Rationales.** Fluent rationales for both Jev's labels and the flipped labels.
- **Cheap alternatives.** A warning prompt, label only, confidence, and blind-first (the reviewer commits to an answer before seeing Jev's).
- **Shared bias.** Same-family vs. different-family upstream models at equal accuracy.
- **WildGuard response harmfulness.**
- **Part 2.** Post-training the reviewer, with entire upstream models held out for testing.
