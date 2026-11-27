# E.22: a reviewer from another family, on WildGuard

**Status:** done 2026-11-24.
- **Run:** `e22-v1-2026-11-24`, **$4.19 Bedrock**, 5,181 calls, 1 unparsed. The runner projected $3.11, so the run came in **35% over**.
  - The pilot's measured rate ($0.00077 per call, against $0.0006 assumed) had pointed to about $4.0, but the runner's projection wasn't updated.
  - The run stayed under its $6 cap.
- **Pilot:** `e22-pilot-2026-11-24`, $0.12, 162 calls, 100% parsed.

**Plan:** [research_docs/17](../../research_docs/17_second_batch.md#e22-a-reviewer-from-another-family-on-wildguard-about-35-bedrock). **Rule, fixed before the run:** [e22.yaml](../../configs/experiments/e22.yaml).

**Result: no.**
- **Accuracy:** with Mistral Large 3 as the reviewer, the WildGuard cascade is **3.4 points below Jev alone [−5.5, −1.1]**.
- **Leniency:** it lets more harm through. Harmful items let through rise from 3.8% to 9.5%, so the leniency guard fails too.
- **What Mistral does reach:** Jev's shared blind spots.
  - On the Jev mistakes gpt-oss-20b never gets right, Mistral overrides 37% against the 20b's 4%.
  - Its errors overlap with Jev's much less: phi 0.44, against 0.64 for the 20b.
- **But it errs the other way:** it misses 30% of harmful items. Its over-flagging fix comes with leniency, the same trade as E.116's worked examples.

## Question

E.119 closed the capability question within gpt-oss. The open caveat was family. In E.112's training-pool screen, Mistral fixed far more of Jev's over-flagged WildGuard items than gpt-oss did. Does a Mistral reviewer beat Jev alone on WildGuard's **test** items?

## Setup

- **Design:** E.119's design on E.102's frozen plan, WildGuard prompts and responses only, with Mistral Large 3 as the reviewer. Same items, Jev labels, matched random labels and subsample (400 + 400).
- **Calls:** 3 alone samples per reviewed item, plus one review of each proposal.
- **Protocol difference, stated in the config:** Mistral isn't a reasoning model, so the standard reviewer prompt gets E.112's instruction to write its reasoning before the JSON. Everything else is unchanged.
- **Rule:** both of the following, on the WildGuard subsample, pooled:
  1. Mistral cascade − Jev alone > 0, with the 95% interval above 0;
  2. harmful let through no more than 1 point above Jev alone.

## Results

**Accuracy on the WildGuard subsample, paired on the same items:**

| | WildGuard prompts | WildGuard responses | Pooled |
|---|---|---|---|
| Jev alone | 90% | 92% | 91% |
| gpt-oss-20b alone | 90% | 90% | 90% |
| Mistral alone | 85% | 92% | 89% |
| 20b cascade | 88% | 88% | 88% |
| Mistral cascade | 84% | 92% | 88% |
| **Mistral cascade − Jev** | −6.5 [−10.0, −3.2] | −0.2 [−2.8, +2.2] | **−3.4 [−5.5, −1.1]** |
| Mistral cascade − 20b cascade | −4.5 [−8.5, −0.8] | +4.0 [+0.8, +7.5] | −0.2 [−2.6, +2.1] |

- **Harmful let through** (share of all subsample items):
  - Jev 3.8%;
  - Mistral cascade 9.5% (+5.8 points);
  - 20b cascade 3.5%.
- **Against the 120b cascade (E.119),** on the same items: −0.9 [−3.1, +1.5].
- **Without step 0's read items** (7 in the subsample): −3.7 [−5.8, −1.5].

**Which way the errors go** (alone majority on the subsample):

| | Harmful items missed | Benign items over-flagged |
|---|---|---|
| Mistral, prompts | 55 of 184 (30%) | 5 of 216 (2%) |
| gpt-oss-20b, prompts | 15 of 184 (8%) | 25 of 216 (12%) |
| Mistral, responses | 22 of 71 (31%) | 8 of 329 (2%) |
| gpt-oss-20b, responses | 7 of 71 (10%) | 35 of 329 (11%) |

**Jev's wrong labels, by the 20b's alone record** (pooled over both pools):

| 20b alone record | Items | 20b overrides Jev | Mistral overrides Jev | Mistral right alone |
|---|---|---|---|---|
| Never right (shared blind spots) | 142 | 4% | **37%** (+33 [+25, +42]) | 42% |
| Sometimes right | 131 | 44% | 56% | 57% |
| Always right | 35 | 86% | 46% | 46% |
| All of Jev's mistakes | 308 | 30% | **46%** (+16 [+9, +23]) | 49% |

**Error overlap (phi):** Jev–20b 0.64, Jev–Mistral 0.44, 20b–Mistral 0.32.

**Exploratory, not pre-registered:** what if Mistral were used only to overturn Jev's "harmful" labels, leaving its "unharmful" ones alone?
- Result: −3.1 [−5.2, −1.1], with harmful let through 9.8%.
- Mistral un-flags real harm along with Jev's over-flagged benign items, so a one-way cascade doesn't help either.

## What it means

1. **The shared blind spots are not universal.** A model from another family reaches a third of the Jev mistakes gpt-oss never gets right, and its errors overlap with Jev's far less. E.24 shows why: Mistral ignores the over-flagging triggers (dark fiction, harm words in a harmless setting) that Jev and gpt-oss share.
2. **But Mistral's errors run the other way.**
   - It is lenient: it misses 30% of harmful items, mostly on prompts.
   - On WildGuard's test items, many harmful prompts are adversarial or indirect. The probes' harmful versions were clear, and Mistral flagged all of those.
   - So Mistral trades Jev's over-flagging for letting harm through. That loses accuracy on prompts and fails the safety guard.
3. **This is E.20's arithmetic again.**
   - Jev is wrong on only 8–10% of WildGuard items. A reviewer must fix about ten mistakes for every right label it breaks.
   - Mistral fixes many of Jev's mistakes (46%), but breaks too many of its right "harmful" labels.
   - Lower overlap is necessary, but a reviewer also has to be at least as accurate as Jev in both directions.
4. **The family caveat is closed for the strongest lead we had.**
   - Across sizes (E.119) and families (E.22), no reviewer we can afford beats Jev on WildGuard.
   - The one that reaches Jev's blind spots does so by being lenient.

## Caveats

- **The protocol differs:** written reasoning before the JSON, not native reasoning. A reasoning-mode model from another family wasn't tried.
- **3 alone samples instead of 5,** so "right alone" is coarser for Mistral than for the gpt-oss models.
- **WildGuard only.** E.112 found Mistral worse on Civil Comments, so it wasn't tested there.
- **The labels:** WildGuardTest's human labels have only moderate annotator agreement (κ 0.55 / 0.50, E.115). Some of Mistral's "misses" on adversarial prompts may be contestable. E.23 (re-labelling) would show how many.

## Reproduce

```
uv run python -m sysone.experiments.e22 --report-only --run-id e22-v1-2026-11-24
```

- **Code:** `sysone.experiments.e22` and `sysone.analysis.e22`, which reuses `sysone.analysis.e119.compare`.
- **Model:** alias `reviewer_mistral` in `configs/models.yaml`.
- **Copied from the runs:** `data/comparison.md`, `data/comparison.json`, `data/manifest.json` and `data/pilot_manifest.json`.
