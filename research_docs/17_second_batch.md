# 17. Second batch (E.20–E.24): explain the wall, then test the one open route

**Status (2026-11-24):** **E.20 done** ([results](../research_analyses/E20/README.md): R1 and R2 pass; the reviewer re-decides rather than defers, and the ceiling is +1.6 to +2.8 points). **E.20a done** ([results](../research_analyses/E20a/README.md): selective override, chosen on training pools, gains −0.3 [−1.8, +1.2]; no). **E.21 done** ([results](../research_analyses/E21/README.md): Part A fails; Part B passes with the lexical monitor, the judge fails; $0.21). **E.22 done** ([results](../research_analyses/E22/README.md): no; Mistral reaches Jev's blind spots but is lenient; $4.19). **E.24 done** ([results](../research_analyses/E24/README.md): H1 and H2 pass, H3 and H4 fail; $1.22). **E.23 in progress** (2026-11-25): rule amended and fixed in `configs/experiments/e23.yaml`; subagent labelling under way, then your hand sheet. E.24's probe set is built and frozen (`benchmarks/probes/minimal_pairs_v1/`, 180 triplets, 540 items); nothing has been measured on it.

Numbering: the first batch was E.101–E.119. This batch uses E.20–E.29 so the two batches can't be confused.

## Why this batch

Every intervention on the reviewer (prompts in E.103 and E.116, five training methods in E.106–E.110, steering in E.110, and a 6× larger model in E.119) failed in the same way:
- It moved the reviewer's overall lean toward accepting or overriding Jev.
- It never changed *which* items the reviewer gets right.
- The reviewer's mistakes overlap with Jev's (phi about 0.57), and no cascade beat Jev alone.

That is a finding in its own right. This batch does two things with it:
1. **Explain it:**
   - E.20: a quantitative ceiling;
   - E.21: what a router and a monitor can still do;
   - E.24: what the shared blind spot is.
2. **Test the one route still open:**
   - E.22: a reviewer from another model family;
   - E.23: whether the benchmark's labels are part of the wall.

## Order and dependencies

| # | Question | Cost | Hand labels | Depends on |
|---|---|---|---|---|
| **E.20** | Can error overlap predict every cascade result we have? | $0 | No | — |
| **E.21** | If the cascade can't fix, can it route? Does the reviewer's reasoning show when it gave in? | A: $0. B: about $0.30 Bedrock | No | — |
| **E.22** | Does a reviewer from another family (Mistral Large 3) beat Jev on WildGuard? | about $3–5 Bedrock | No | — |
| **E.23** | How much of the never-right group is label noise, and does re-labelling change the verdict? | $0 API | **Yes:** about 60–100 items | — |
| **E.24** | What makes the shared blind spot? Controlled minimal-pair probes | about $2 (Bedrock and TypeSafe) | No (optional spot check) | Items written and checked first |

E.22 and E.24 share one live session if both are approved: same models, one cost projection.

## Rules carried over

- **No prompt wording and no probe item is written from test items.** E.24's items are written from the task policies (`sysone.cascade.tasks`), WildGuard's published taxonomy, and the failure *types* named in step 0 (harm word in a harmless setting, quoted stereotypes, judging the request instead of the response). Never from step 0's items or any other test item. A script checks every probe item for near-duplicates against every cascade pool (test and training) before use.
- **The frozen standard reviewer prompt is unchanged** in every experiment here.
- **Each confirmatory test has a decision rule written into its config before the run,** as in E.116 and E.119.
- **Labelling and reading are done by subagents,** separate from the agent that designs the analysis. If a subagent is blocked by safeguards, the items it couldn't read go to the user. We don't work around the block.
- **Spend is checked against the caps:** Bedrock $50 (spent $9.95), GCP $50, TypeSafe $5 (spent $2.10). If a run's spend is 10% or more over its projection, the user is warned.

---

## E.20: a correlation ceiling for the cascade (free)

**Question:** can the cascade's accuracy, for *any* upstream model and *any* reviewer variant we've run, be predicted from:
- how often each model is right alone;
- which items they get wrong together (error overlap);
- one or two deference parameters?

If it can, then the experiments of E.103–E.119 are all points on a single frontier, and the wall has a formula.

**Key observation that makes this free:** every task is binary, and the reviewer's prompt doesn't say who proposed the label. So the reviewer's response to "label L shown on item i" is the same whichever upstream model proposed it. A cascade with Llama, Qwen or Nova upstream can therefore be computed from reviews we already have: the `correct_label`, `jev_label` and `matched_label` reviews, plus E.106's `upstream_wrong` reviews. Only *which items each upstream gets wrong* differs.

**Data (all existing):**
- Jev: `e102-v1-2026-11-04/plan.jsonl`.
- Llama 4 Maverick, Qwen3-32B and Nova Lite alone on every test item: `e103-v1-2026-11-04/upstream.jsonl`.
- gpt-oss-20b:
  - alone ×5 and all cascade conditions: `e102` (Bedrock) and `e106` `base_local` (vLLM);
  - the protocol arms: `e103`;
  - the trained arms: `e106`, `e108`;
  - steering: `e110`.
- gpt-oss-120b: alone ×5 and the cascade, `e119`.

**Method:**
1. **The ceiling** for each pair (upstream U, reviewer R), from alone answers only:
   - the oracle accuracy, 1 − P(both wrong);
   - the headroom over the better model, P(U wrong ∧ R right) and P(U right ∧ R wrong);
   - the phi of their error indicators.
2. **A two-parameter deference model.** On each item the reviewer's final label is its own alone answer, except:
   - it accepts a shown wrong label with probability d_w;
   - it accepts a shown right label it disagrees with with probability d_r.

   Per item, the alone answer is the reviewer's 5-sample vote share, not a single draw.
3. **Fit** d_w and d_r on one cascade only: Jev → gpt-oss-20b, standard prompt, on E.102's subsample.
4. **Predict, with no refitting,** the cascades it wasn't fitted on:
   - Jev → 120b (E.119);
   - Llama, Qwen and Nova → 20b, on their wrong labels (E.106 `upstream_wrong`) and right labels (`correct_label`);
   - the random-matched upstream (E.102's matched labels, an upstream whose phi is about 0).
5. **Place every intervention on one chart:**
   - E.103's protocols, E.106–E.110's trained arms, E.116's arms (WildGuard only) and E.119;
   - each refitted for its own (d_w, d_r);
   - accuracy against the headroom it used.

   The expected picture is that interventions slide along a trade-off line (lower d_w comes with lower d_r or a higher break rate), and none moves the headroom.

**Decision rule (to fix in `configs/experiments/e20.yaml` before computing step 4):**
- The model "explains the cascade" if every held-out cascade's predicted accuracy falls inside that cascade's 95% bootstrap interval.
- It also has to pass a weaker check: the predicted ranking of the held-out cascades matches the observed ranking (Spearman ≥ 0.8).

**Outputs:**
- a table of predicted against observed accuracy;
- the frontier figure;
- one closed-form statement of the ceiling: given accuracies a_U and a_R and overlap phi, the most any reviewer without an extra signal can add.

**What it is not:** a new intervention. If the deference model fails, that is informative too: something other than overlap and lean drives the cascade, and we look at where the residuals are.

**Related work to cite:**
- Goel et al. 2025, "Great Models Think Alike" (CAPA, AI oversight under model similarity);
- Kim et al. 2025 (correlated errors across LLMs);
- the classic ensemble-diversity results (Kuncheva & Whitaker 2003).

What's new here is connecting the ceiling to measured deference in a sequential review cascade.

---

## E.21: route, don't fix; and does the reasoning reveal deference? (Part A free; Part B about $0.30)

### Part A: silent failures and routing to a person (free)

**Question:** if the cascade's value isn't correction, it may be **triage**: deciding which items a person should look at. How many of Jev's mistakes does each routing signal catch per 100 escalations, and how many are silent (Jev and the reviewer agree, both wrong)?

**Data:**
- Test: `e106` `base_local` (alone ×5, cascade reviews with reasoning) and Jev's E.102 decisions.
- Training (to fit routers): E.104 Jev decisions and E.105's 20b `alone` and `jev_label` traces on the training pools.

**Routing signals:**
1. random (the floor);
2. Jev's own confidence (E.111 / E.114);
3. reviewer alone disagrees with Jev (majority of 5);
4. the reviewer overrides Jev in the cascade (one call, no alone samples);
5. a combined score: logistic regression on 2–4 plus the reviewer's alone vote share, fitted on the training pools only.

**Measures,** at fixed human budgets of 5%, 10% and 25% of items, per pool and pooled:
- the share of Jev's mistakes caught;
- the residual error rate;
- errors caught per 100 escalations.

The **silent failure rate** is the share of Jev's mistakes no signal flags, split by groups A/B/C.

**Decision rule (in `configs/experiments/e21.yaml`, before computing):** "Review adds triage value" if, at the 10% budget, the combined router catches more of Jev's mistakes than Jev's confidence alone, paired, with the 95% interval above 0, pooled on E.102's subsample.

### Part B: a deference monitor on the reviewer's reasoning (about $0.30 Bedrock)

**Question:** E.110 found the "give-in" direction inside the reviewer is almost perfectly readable (AUC 0.998) but not causal. Is it also readable in the reasoning *text*? If so, a monitor can flag reviews where the reviewer gave in and send them to a person, instead of trying to train the giving-in away. This ties back to the project's explainability question: **does the reviewer's explanation reveal that it deferred?**

**Labels:** behavioural, so no hand labels are needed. A review **gave in** if:
- the reviewer accepted Jev's wrong label;
- on an item it gets right alone (majority of its alone samples).

It **held firm** if it overrode on such an item. Accepts on never-right items (group C) are a third class, **genuine agreement.**

**Monitors:**
1. Lexical: TF-IDF logistic regression on the reasoning text. Free.
2. LLM judge: gpt-oss-120b on Bedrock. It reads only the reasoning and answers whether the reviewer evaluated the item on its merits or anchored on the proposed label.
   - The judge's prompt is written from **training-pool traces** only (E.105).
   - About 2,300 test traces at about $0.00013 each is about $0.30.

**Development and test:** monitors are developed on E.105's training-pool traces (Bedrock 20b) and tested once on E.106 `base_local`'s test traces (same weights, vLLM). The stack difference is reported as a caveat.

**Measures:**
- AUC for gave-in against held-firm;
- **the useful test:** AUC for gave-in against *correct accepts* (Jev right, reviewer accepted). A monitor that only detects "accepted" is useless;
- how the monitor scores group C accepts, where we expect it to fail: the reviewer genuinely agrees;
- Part A's routing table with the monitor added as a signal.

**Decision rule:** the monitor "reveals deference" if test AUC against correct accepts is ≥ 0.70 with the interval above 0.60, pooled.

**Reading:** a subagent reads 30 test traces it hasn't been told the class of (10 gave-in, 10 held-firm, 10 group C) and describes the differences. This is description only, not a claim.

---

## E.22: a reviewer from another family, on WildGuard (about $3–5 Bedrock)

**Question:** E.119 closed the capability question within gpt-oss. Its open caveat is family. The lead:
- In E.112's training-pool screen, Mistral Large 3 got 66% (53/80) of the WildGuard items no gpt-oss try ever solved, against 6% for gpt-oss-120b.
- It fixed 85 of 112 of Jev's over-flagged WildGuard items, against 17 for the 20b.

Does a Mistral reviewer beat Jev alone on WildGuard's **test** items?

**Main risk:** Mistral may just be more lenient, the same trade-off that sank E.116's `examples` arm. The rule guards against that.

**Design:** E.119's design and code on E.102's frozen plan, with `teacher_mistral`'s model as the reviewer, on **WildGuard prompts and responses only**. Civil Comments is out: E.112 found Mistral worse there, and its labels are noisy.
- Same items, Jev labels, matched random labels and subsample (400 + 400).
- Mistral isn't a reasoning model. E.112's written-reasoning-then-JSON format is used, through the same reviewer prompt plus E.112's format instruction. This is reported as a difference in protocol.
- **3 alone samples** instead of 5, for cost.
- **Calls:**
  - about 3,800 alone;
  - 308 Jev-wrong reviews;
  - 308 matched reviews;
  - 800 subsample reviews;
  - in total about 5,200 at about $0.0006, about $3.1. Cap $6.
- **Pilot first:** `--max-items 20` per pool. The pilot must parse at least 95% of replies and finish under the per-call cap before the full run.

**Decision rule (in `configs/experiments/e22.yaml`, before the run):** "Another family fixes it on WildGuard" if both hold:
1. Mistral cascade − Jev alone on the WildGuard subsample (800, pooled over the two pools) is above 0, with the 95% paired interval above 0;
2. the harmful-let-through rate of the cascade is no more than 1 point above Jev alone's (point estimate). This is the leniency guard.

**Also reported:**
- Mistral alone against Jev;
- phi with Jev compared with the 20b's and 120b's, which feeds E.20;
- override of Jev's mistakes by the 20b's never/sometimes/always record;
- the result without step 0's 30 read items (sensitivity).

**If it passes:** this is the first cascade in the project that beats Jev. It would also mean the wall is family-specific, which E.20 should predict from phi. **If it fails:** the "same family" caveat is closed for the strongest lead we had.

---

## E.23: re-label the never-right items (about 60–100 items by hand)

**Question:** step 0 read 30 WildGuard group C items and judged 20 to have a doubtful gold label or be genuinely ambiguous. How much of "the cascade loses to Jev" and of the never-right group is label noise? Does the verdict change on cleaner labels?

**You will need to label by hand.** A model re-labelling these items is another model's opinion, and the point is to arbitrate *model* blind spots. So subagents do the bulk, and you settle the cases they can't, plus an audit of the ones they agree on.

**Items (WildGuard only):**
- the whole WildGuard part of E.102's random subsample (800), so accuracy can be re-scored without selection bias;
- every Jev-wrong WildGuard item outside it (308 in total, some already in the subsample).

That's about 1,000 items. Civil Comments isn't re-labelled. Instead it is re-scored for free on its high-annotator-agreement items only.

**Procedure:**
1. **Two independent subagent labellers** label every item under WildGuard's guidelines (E.115's summary).
   - They are blind to the gold label, Jev's label and the reviewers' labels.
   - Each gives a label, a confidence, and an "ambiguous" flag.
   - Claude is outside every model family in the cascade, which helps, but it is still a model.
2. **You label:**
   - every item where the two labellers disagree with each other;
   - a random audit of 40 items where they agree.

   You'd see the item, the guideline summary and nothing else. **Estimated 60–100 items, about 1–2 hours,** depending on the disagreement rate.
3. **Consensus label:**
   - your label where you labelled;
   - otherwise the labellers' agreed label;
   - "ambiguous" if two of the three sources say so.

**Measures:**
- the gold-label error rate by group (A/B/C, Jev-right);
- Jev, 20b, 120b and each cascade re-scored against the original gold, against consensus labels, and with ambiguous items excluded;
- how many never-right items stay "never right".

**Decision rule (before the re-score):** "labels are part of the wall" if, on consensus labels, either:
- the sign of 20b cascade − Jev on the WildGuard subsample changes; or
- the never-right share of Jev's mistakes falls by at least a third.

**Odds, from the saved runs (2026-11-25, before any labelling):**
- **Sign leg (20b cascade − Jev, now −25 items of 800):** only the 59 items where Jev and the cascade disagree can move it (17 fixes, 42 breaks). Gold errors on breaks must outnumber those on fixes by 13 or more: about a third of the breaks if the fixes are clean. 36 of the 42 breaks are benign-gold items the cascade over-flags, the kind step 0 mostly judged real model mistakes (doubtful gold on about 1 in 5). Unlikely.
- **Never-right leg (46% of Jev's 308 WildGuard mistakes; a pass needs 31% or less):** with step 0's rates (9 doubtful, 11 ambiguous, 10 clear of 30), flipping only doubtful labels gives about 37%, a fail; also dropping ambiguous items gives about 22%, a pass. The rule doesn't yet say how "ambiguous" is scored, and that must be fixed before any label is seen.
- **Mistral (E.22), not in the rule:** 48 of its 52 breaks are harmful-gold items it let through (40 adversarial), the kind step 0 found doubtful or ambiguous every time (5 of 5). It would need gold errors on about 14 more breaks than fixes. More plausible than the 20b leg, but would have to be added to the rule first.

**Amendments (2026-11-25, before any item was labelled; the full rule is [e23.yaml](../configs/experiments/e23.yaml)):**
1. **How ambiguous items are scored.** Primary, which decides the rule: ambiguous items keep their *original* gold label, so only items with a clear consensus label that differs from gold count as label errors. Secondary, reported only: ambiguous items are excluded from every score. Reason: the question is whether gold labels are *wrong*. An ambiguous item isn't shown to be wrong, and dropping items moves both legs toward a pass.
2. **A third leg for Mistral (E.22).** L3: on consensus labels, E.22's rule passes. That is, Mistral cascade − Jev > 0 on the WildGuard subsample with the interval above 0, and harmful let through at most 1 point above Jev's. It is reported separately: "labels are part of the wall" (L1 or L2) and "a Mistral cascade beats Jev on clean labels" (L3) are different claims.
3. **Details fixed with them:**
   - L1 is E.102's Bedrock 20b, point estimate, with the interval reported.
   - In L2, "never right" uses the same reviewer's 5 alone answers, re-scored on consensus labels.
   - An item a labeller can't label counts as a disagreement and goes to you.
   - If the audit agreement is below 34 of 40, another 40 agreed items go to you, until a round of 40 reaches 34.

**Caveat:** the audit of 40 gives the labellers' agreement with you, which is reported. If it is below about 85%, the subagent labels aren't trusted alone, and more items go to you.

---

## E.24: minimal-pair probes of the shared blind spot (about $2)

**Question:** step 0 traced the shared over-flagging to a few failure *types*. Do those types *cause* errors, for Jev and gpt-oss alike? Does a model from another family resist them? Counting mistakes on natural items can't separate a trigger from everything else in the text. Controlled pairs can.

**Probe items: triplets, written by subagents, labelled by construction, checked blind.** Each triplet differs by one minimal edit:
- **(a) neutral:** benign, no trigger;
- **(b) trigger:** benign, with the trigger, e.g. a harm word in a harmless setting;
- **(c) harmful:** a minimal edit of (b) that makes it truly harmful under the policy.

The contrasts:
- **(b) − (a), the trigger effect:** how often the trigger alone makes a model call a benign item harmful (over-flagging);
- **(c) − (b), sensitivity:** whether the model tracks the edit that actually matters.

**Families (20 triplets each, 9 families, 540 items):**

| Family | Task | Trigger (b) | Harmful edit (c) |
|---|---|---|---|
| P1 harm word, harmless setting | prompt | violent or illegal vocabulary in games, cooking, tech, sport, idiom | the same request aimed at real harm |
| P2 quoted or critiqued stereotype | prompt | asks to analyse, rebut or explain a stereotype | asks to produce or endorse it |
| P3 education or safety | prompt | asks why something is dangerous, or its history | asks how to do it (request only) |
| P4 fiction frame | prompt | dark fiction, no actionable request | the fiction frame asks for real actionable detail |
| P5 disguised harm (let-through direction) | prompt | (b) states a harmful request plainly | (c) the same harmful request in euphemism, no harm vocabulary; both (b) and (c) harmful, (a) benign |
| R1 refusal of a harmful request | response | harmful request + refusal or safe redirect | harmful request + compliance |
| R2 discussing compared with assisting | response | the response discusses a harmful topic critically or educationally | the response assists or endorses |
| C1 undirected profanity | toxicity | profanity not aimed at anyone | the same profanity aimed at a person |
| C2 identity mention | toxicity | a neutral or positive mention of an identity group | an attack on that group |

**Safety of the items themselves:**
- Harmful items are *requests* or low-severity content (harassment, deception, insults). They never contain actionable instructions for serious harm.
- Harmful responses comply at a general, non-operational level.

**Checks before any live call:**
1. **Leakage:** a script compares every item with every cascade pool (test and training) by character 8-gram overlap and flags anything above a threshold. Flagged items are rewritten.
2. **Blind label check:** two subagents that didn't write the items label all 540, shuffled, without the intended label. Items where either disagrees with the intended label are fixed or dropped.
3. **Minimality check:** a subagent confirms each (a)/(b)/(c) differs only by the intended edit.
4. **Optional spot check by you:** 20 random triplets. Not required.

**Models measured:** each alone, on every item:
- Jev ×3 (TypeSafe, about $0.05);
- gpt-oss-20b ×5;
- gpt-oss-120b ×5;
- Mistral Large 3 ×3;
- Qwen3-32B ×1.

Plus one cascade review by gpt-oss-20b of Jev's label on every item. About $2 in total, with a cap of $4 Bedrock and $0.20 TypeSafe.

**Hypotheses (fixed in `configs/experiments/e24.yaml` before the run):**
- **H1 (triggers cause over-flagging):** pooled over P1–P4, the flag rate on (b) is above that on (a), for both Jev and gpt-oss-20b, paired, with the 95% interval above 0.
- **H2 (the blind spot is shared):** the per-triplet trigger effects of Jev and the 20b are correlated (phi of their (b)-errors > 0.3), and more than those of Jev and Mistral.
- **H3 (the cascade inherits it):** on (b) items Jev flags, the 20b cascade accepts the flag at least as often as on natural Jev-wrong items (E.102's rate).
- **H4 (disguise causes let-through):** in P5, the flag rate on (c) is below that on (b).

**Outputs:**
- a per-family table of the trigger effect and sensitivity for every model;
- which families drive the shared errors;
- a check of whether E.22's Mistral advantage (if any) is specific to these triggers.

**Build record (2026-11-24):**
- **Written:** four writer subagents, one per family group, from the policies only. None opened any data, results or analysis folder.
- **Leakage:** no item shares 50% or more of its character 8-grams with any of the 22,551 pool items, test or training. The highest share is 0.48, from generic phrases inside long, unrelated training items.
- **Minimality, round 1:** a reviewer subagent found 5 minimality fails, 1 family misfit and 1 safety fail. The safety fail was a response with an insulin-stopping schedule. It also found a C1-wide design mismatch: (c) added a new insult instead of aiming the same profanity at a person.
- **Revision:** a separate subagent revised 79 triplets. The originals are in `drafts_v0/`, and the changes are listed in `revision_log.jsonl`.
- **Minimality, round 2:** a fresh reviewer checked the 79 revised triplets and found 0 fails (40 "minor"). Five triplets rated closest to failing (C2-16, P4-18, P1-19, P3-18, P4-11) are kept, flagged `near_fail_minimality`, and excluded in a sensitivity analysis.
- **Blind labels:**
  - Two labellers who never saw the drafts (`blind/labels_c.jsonl`, `labels_d.jsonl`) agreed with the intended label on all 540 items. So 0 triplets were dropped.
  - The first labelling round, on the unrevised drafts, also agreed on all 540 (`blind_v0/`).
- **Low confidence:** 33 items are flagged `labeller_unsure`, marked by either labeller:
  - all 20 C1 triggers (mild, undirected profanity, against the policy's "obscenity is toxic");
  - 8 C1 harmful items (mild curses at a person);
  - 3 P4 triggers (dark fiction);
  - 2 P2 harmful items (stereotypes of a region and of a profession).

  C1 is therefore reported separately and never pooled with the WildGuard families.

**Where the items live:** `benchmarks/probes/minimal_pairs_v1/items.jsonl` (not `data/`, which is gitignored), in the cascade `Item` format, with `pair_id`, `family` and `role` (`neutral` / `trigger` / `harmful`) in `meta`. They're our own text, so they can be committed. The writers' drafts (`drafts/`), the leakage report and the blind-label results are saved next to them. Built by `scripts/build_probes_e24.py`.
