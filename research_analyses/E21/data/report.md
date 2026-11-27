# E.21 results

## Decisions (rules fixed in configs/experiments/e21.yaml before computing)

- **Part A, review adds triage value: FAIL.** At a 10% budget the combined router catches +4.4 points [-1.7, +10.0] more of Jev's mistakes than Jev's confidence alone (pooled).
- **Part B, the reasoning reveals deference: PASS.** Monitor chosen on training: lexical (training AUC {'lexical': 0.856, 'judge': 0.4615}); test AUC gave-in vs correct accepts 0.9171 [0.8922, 0.9395].

## Part A: share of Jev's mistakes caught by sending k% of items to a person (E.102 subsample, pooled)

| Signal | 5% | 10% | 25% |
|---|---|---|---|
| random | 4.0% | 11.2% | 26.0% |
| jev_confidence | 18.5% | 30.8% | 62.1% |
| alone_disagree | 17.5% | 29.5% | 61.3% |
| cascade_override | 18.8% | 28.8% | 63.0% |
| combined | 18.7% | 35.2% | 63.9% |

Per pool at 10%: jev_confidence: Civil Comments 16%, WildGuard prompts 36%, WildGuard responses 40%; combined: Civil Comments 24%, WildGuard prompts 38%, WildGuard responses 43%

Combined model coefficients (training pools): Civil Comments {'jev_risk': 3.07, 'jev_split': -0.176, 'alone_disagree': 1.175, 'cascade_override': 0.349}; WildGuard prompts {'jev_risk': 7.08, 'jev_split': -0.194, 'alone_disagree': 1.074, 'cascade_override': 0.108}; WildGuard responses {'jev_risk': 6.687, 'jev_split': 0.195, 'alone_disagree': 2.065, 'cascade_override': 0.284}

### Silent failures (every Jev mistake; reviewer's alone majority agrees with Jev and the cascade accepts)

| Pool | Group | Jev mistakes | Silent |
|---|---|---|---|
| Civil Comments | A | 281 | 9% |
| Civil Comments | B | 154 | 55% |
| Civil Comments | C | 315 | 91% |
| Civil Comments | all | 750 | 53% |
| all | A | 364 | 9% |
| all | B | 220 | 51% |
| all | C | 474 | 92% |
| all | all | 1058 | 55% |
| WildGuard prompts | A | 41 | 10% |
| WildGuard prompts | B | 42 | 50% |
| WildGuard prompts | C | 81 | 91% |
| WildGuard prompts | all | 164 | 60% |
| WildGuard responses | A | 42 | 10% |
| WildGuard responses | B | 24 | 25% |
| WildGuard responses | C | 78 | 96% |
| WildGuard responses | all | 144 | 59% |

### With the lexical monitor as a routing signal (reported only, not part of rule A)

| Signal | 5% | 10% | 25% |
|---|---|---|---|
| jev_confidence | 18.5% | 30.8% | 62.1% |
| combined | 18.7% | 35.2% | 63.9% |
| monitor | 12.8% | 22.2% | 48.8% |
| combined+monitor | 21.1% | 34.1% | 64.4% |

## Part B: monitors on the reviewer's reasoning

Test classes: {'gave_in': 78, 'held_firm': 286, 'genuine': 437, 'correct_accept': 930}. Training classes: {'gave_in': 93, 'held_firm': 388, 'genuine': 902, 'correct_accept': 3650}.

Lexical monitor: training CV AUC 0.856 (n = 3743). Terms pointing to gave-in: real, but, but does, considered, mild, might, does, might be, no direct, private. Terms pointing to correct accept: so verdict, accept, is, final_label toxic, is insulting, toxic, verdict, the, toxic so, verdict accept.

Judge run: {'run_id': 'e21-judge-v1-2026-11-24', 'spent_usd': 0.2134, 'dev_auc_gave_in_vs_correct_accept': 0.4615, 'parsed_test': 1729}.

| Monitor | Gave in vs correct accept | Gave in vs genuine agreement | Gave in vs held firm | Mean score by class |
|---|---|---|---|---|
| lexical | 0.9171 [0.8922, 0.9395] (78/930) | 0.8477 [0.8044, 0.8882] (78/437) | 0.7335 [0.6783, 0.7839] (78/286) | {'gave_in': 0.394, 'held_firm': 0.262, 'genuine': 0.197, 'correct_accept': 0.143} |
| judge | 0.4099 [0.3562, 0.4605] (78/930) | 0.4686 [0.4113, 0.5242] (78/435) | 0.8097 [0.7642, 0.8569] (78/286) | {'gave_in': 1.936, 'held_firm': 0.923, 'genuine': 2.136, 'correct_accept': 2.731} |

Within one proposed label (robustness, reported only; at least 10 per class):

- lexical, gave_in_vs_correct_accept|harmful: AUC 0.903 [0.8325, 0.958] (13/218)
- lexical, gave_in_vs_correct_accept|non_toxic: AUC 0.9079 [0.857, 0.9516] (39/123)
- lexical, gave_in_vs_correct_accept|toxic: AUC 0.9357 [0.8701, 0.9778] (11/133)
- lexical, gave_in_vs_correct_accept|unharmful: AUC 0.8895 [0.8245, 0.9417] (15/456)
- lexical, gave_in_vs_genuine|harmful: AUC 0.7538 [0.6307, 0.865] (13/110)
- lexical, gave_in_vs_genuine|non_toxic: AUC 0.8504 [0.7815, 0.9079] (39/97)
- lexical, gave_in_vs_genuine|toxic: AUC 0.9086 [0.8179, 0.9695] (11/191)
- lexical, gave_in_vs_genuine|unharmful: AUC 0.759 [0.6229, 0.889] (15/39)
- lexical, gave_in_vs_held_firm|harmful: AUC 0.5688 [0.38, 0.7518] (13/33)
- lexical, gave_in_vs_held_firm|non_toxic: AUC 0.8262 [0.7665, 0.8867] (39/191)
- lexical, gave_in_vs_held_firm|toxic: AUC 0.607 [0.4383, 0.7566] (11/31)
- lexical, gave_in_vs_held_firm|unharmful: AUC 0.529 [0.3559, 0.6882] (15/31)
- judge, gave_in_vs_correct_accept|harmful: AUC 0.3813 [0.2622, 0.5243] (13/218)
- judge, gave_in_vs_correct_accept|non_toxic: AUC 0.4583 [0.3535, 0.5576] (39/123)
- judge, gave_in_vs_correct_accept|toxic: AUC 0.5933 [0.4346, 0.7323] (11/133)
- judge, gave_in_vs_correct_accept|unharmful: AUC 0.3561 [0.2613, 0.4631] (15/456)
- judge, gave_in_vs_genuine|harmful: AUC 0.3696 [0.2391, 0.5186] (13/110)
- judge, gave_in_vs_genuine|non_toxic: AUC 0.4959 [0.4147, 0.5936] (39/97)
- judge, gave_in_vs_genuine|toxic: AUC 0.5688 [0.443, 0.7024] (11/189)
- judge, gave_in_vs_genuine|unharmful: AUC 0.4222 [0.29, 0.5663] (15/39)
- judge, gave_in_vs_held_firm|harmful: AUC 0.6841 [0.5087, 0.8276] (13/33)
- judge, gave_in_vs_held_firm|non_toxic: AUC 0.8376 [0.7683, 0.8988] (39/191)
- judge, gave_in_vs_held_firm|toxic: AUC 0.827 [0.6957, 0.9413] (11/31)
- judge, gave_in_vs_held_firm|unharmful: AUC 0.7462 [0.5817, 0.8775] (15/31)
