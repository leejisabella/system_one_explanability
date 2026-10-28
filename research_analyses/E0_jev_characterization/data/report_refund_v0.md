# Pilot 0 report pilot0-v0-2026-10-28

Rows: 480 · cases: 60 · variants: 160

**P0.1 parse success:** 100% (480/480)

**P0.2 agreement with policy on the presented record (per sample)**

| Case type | Agreement |
|---|---|
| ambiguous | 92% (83/90) |
| correct | 95% (57/60) |
| hard_negative | 92% (83/90) |
| ignored_evidence | 84% (76/90) |
| incorrect_input | 80% (48/60) |
| irrelevant_cue | 100% (90/90) |

By the rule that decides the presented record. Date-window rules need date arithmetic, which TypeSafe lists as unreliable for Jev.

| Rule | Agreement |
|---|---|
| R1_manager_preapproval | 100% (30/30) |
| R2_final_sale | 100% (24/24) |
| R3_no_receipt | 100% (45/45) |
| R3_receipt_unknown | 92% (11/12) |
| R4_condition_unknown | 100% (9/9) |
| R4_opened | 100% (57/57) |
| R5_outside_window | 58% (57/99) |
| R5_purchase_date_unknown | 100% (9/9) |
| R5_within_window | 100% (195/195) |
| **date window** | 86% (252/294) |
| **other rules** | 99% (185/186) |

**P0.3 agreement with ground truth (majority decision on the original case)**

| Case type | Agreement |
|---|---|
| ambiguous | 100% (10/10) |
| correct | 100% (10/10) |
| hard_negative | 90% (9/10) |
| ignored_evidence | 100% (10/10) |
| incorrect_input | 40% (4/10) |
| irrelevant_cue | 100% (10/10) |

| Rule deciding the true record | Agreement |
|---|---|
| R1_manager_preapproval | 100% (5/5) |
| R2_final_sale | 100% (4/4) |
| R3_no_receipt | 100% (4/4) |
| R3_receipt_unknown | 100% (4/4) |
| R4_condition_unknown | 100% (3/3) |
| R4_opened | 100% (5/5) |
| R5_outside_window | 86% (6/7) |
| R5_purchase_date_unknown | 100% (3/3) |
| R5_within_window | 76% (19/25) |

Contestable cases (Jev wrong under ground truth): 7 of 60; tied: 0

| Why Jev is wrong | Cases |
|---|---|
| follows presented record | 6 |
| misapplies policy: date window | 1 |

| Who the wrong decision harms | Cases |
|---|---|
| customer: wrongly denied | 6 |
| store: wrongly approved | 1 |

"Follows presented record" means Jev applied the policy correctly to a corrupted or incomplete record, a defect planted by the benchmark. The other rows are Jev's own errors.

**P0.4 stability:** mean majority share 0.9896, unanimous 0.9688 over 160 variants

**P0.4 probability noise:** range of P(approve) across repeats of the same input

| Measure | Value |
|---|---|
| Variants with 2+ parsed repeats | 160 |
| Share with identical P(approve) on every repeat | 0.3312 |
| Mean range | 0.0239 |
| 95th-percentile range (candidate noise floor) | 0.09 |
| Largest range | 0.17 |

Repeats listed the options in different orders, so these ranges include any order effect and are an upper bound on pure repeat noise. For a pure noise floor, run with `rotate_option_order: false`.

**P0.4 option order:** the first-listed option was chosen in 0.3271 of 480 answers (0.3333 expected with no order effect, when every variant sees each order equally often). P(approve) was -0.0104 higher on average when approve was listed first (160 variants).

**P0.5 / P0.6 decision flips vs. the original**

| Variant kind | Policy expectation | Flip rate | Mean shift in P(approve) |
|---|---|---|---|
| correction | policy flips | 60% (6/10) | 0.7483 |
| cue_edit | policy unchanged | 0% (0/20) | 0.0063 |
| decisive_edit | policy flips | 90% (18/20) | 0.8115 |
| discrepancy_fixed | policy unchanged | 0% (0/10) | 0.0143 |
| exception_removed | policy flips | 50% (5/10) | 0.4653 |
| exception_salient | policy unchanged | 0% (0/10) | 0.0703 |
| resolution | original undetermined | 100% (20/20) | 0.5665 |

**P0.7 model versions:** {'jev-1.14.0': 480}

**P0.8 cost:** $0.0137 total; 480 billed decisions; per decision 2.8e-05

**Go / no-go**

- Parse rate ≥ 98%: yes
- At least 20 cases where Jev wrongly denies the customer: no (6)
- At least 90% of variants unanimous across repeats: yes (0.9688)
- One model version throughout: yes
