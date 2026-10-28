# Pilot 0 report pilot0-v1-2026-10-28

Rows: 564 · cases: 74 · variants: 188

**P0.1 parse success:** 100% (564/564)

**P0.2 agreement with policy on the presented record (per sample)**

| Case type | Agreement |
|---|---|
| ambiguous | 91% (82/90) |
| correct | 90% (54/60) |
| hard_negative | 90% (81/90) |
| ignored_evidence | 83% (75/90) |
| incorrect_input | 100% (144/144) |
| irrelevant_cue | 89% (80/90) |

By the rule that decides the presented record. Date-window rules need date arithmetic, which TypeSafe lists as unreliable for Jev.

| Rule | Agreement |
|---|---|
| R1_manager_preapproval | 100% (30/30) |
| R2_final_sale | 100% (24/24) |
| R3_no_receipt | 100% (72/72) |
| R3_receipt_unknown | 58% (7/12) |
| R4_condition_unknown | 100% (9/9) |
| R4_opened | 100% (84/84) |
| R5_outside_window | 51% (44/87) |
| R5_purchase_date_unknown | 100% (9/9) |
| R5_within_window | 100% (237/237) |
| **date window** | 87% (281/324) |
| **other rules** | 98% (235/240) |

**P0.3 agreement with ground truth (majority decision on the original case)**

| Case type | Agreement |
|---|---|
| ambiguous | 90% (9/10) |
| correct | 100% (10/10) |
| hard_negative | 80% (8/10) |
| ignored_evidence | 100% (10/10) |
| incorrect_input | 0% (0/24) |
| irrelevant_cue | 90% (9/10) |

| Rule deciding the true record | Agreement |
|---|---|
| R1_manager_preapproval | 100% (5/5) |
| R2_final_sale | 100% (4/4) |
| R3_no_receipt | 100% (4/4) |
| R3_receipt_unknown | 75% (3/4) |
| R4_condition_unknown | 100% (3/3) |
| R4_opened | 100% (5/5) |
| R5_outside_window | 57% (4/7) |
| R5_purchase_date_unknown | 100% (3/3) |
| R5_within_window | 38% (15/39) |

Contestable cases (Jev wrong under ground truth): 27 of 74; tied: 0

| Why Jev is wrong | Cases |
|---|---|
| follows presented record | 24 |
| misapplies policy: date window | 3 |

| Who the wrong decision harms | Cases |
|---|---|
| customer: wrongly denied | 24 |
| store: wrongly approved | 3 |

"Follows presented record" means Jev applied the policy correctly to a corrupted or incomplete record, a defect planted by the benchmark. The other rows are Jev's own errors.

**P0.4 stability:** mean majority share 0.9911, unanimous 0.9734 over 188 variants

**P0.4 probability noise:** range of P(approve) across repeats of the same input

| Measure | Value |
|---|---|
| Variants with 2+ parsed repeats | 188 |
| Share with identical P(approve) on every repeat | 0.383 |
| Mean range | 0.0173 |
| 95th-percentile range (candidate noise floor) | 0.07 |
| Largest range | 0.13 |

Repeats listed the options in different orders, so these ranges include any order effect and are an upper bound on pure repeat noise. For a pure noise floor, run with `rotate_option_order: false`.

**P0.4 option order:** the first-listed option was chosen in 0.3351 of 564 answers (0.3333 expected with no order effect, when every variant sees each order equally often). P(approve) was -0.0091 higher on average when approve was listed first (188 variants).

**P0.5 / P0.6 decision flips vs. the original**

| Variant kind | Policy expectation | Flip rate | Mean shift in P(approve) |
|---|---|---|---|
| correction | policy flips | 100% (24/24) | 0.9867 |
| cue_edit | policy unchanged | 0% (0/20) | 0.0063 |
| decisive_edit | policy flips | 80% (16/20) | 0.8 |
| discrepancy_fixed | policy unchanged | 10% (1/10) | 0.0087 |
| exception_removed | policy flips | 50% (5/10) | 0.472 |
| exception_salient | policy unchanged | 0% (0/10) | 0.0597 |
| resolution | original undetermined | 95% (19/20) | 0.5383 |

**P0.7 model versions:** {'jev-1.14.0': 564}

**P0.8 cost:** $0.0160 total; 564 billed decisions; per decision 2.8e-05

**Go / no-go**

- Parse rate ≥ 98%: yes
- At least 20 cases where Jev wrongly denies the customer: yes (24)
- At least 90% of variants unanimous across repeats: yes (0.9734)
- One model version throughout: yes
