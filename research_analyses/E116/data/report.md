# E.116 report: e116-v1-2026-11-23

Status: completed. Spent $0.5765 ({'bedrock': 0.576469}). Items: {'wildguard_prompt_train_v1/let_through': 73, 'wildguard_prompt_train_v1/over_flagged': 300, 'wildguard_prompt_train_v1/right': 300, 'wildguard_response_train_v1/let_through': 105, 'wildguard_response_train_v1/over_flagged': 190, 'wildguard_response_train_v1/right': 300}.

Development items only (WildGuard training pools, GPT-4 labels). Paired against `standard` on items both arms parsed. Cascade accuracy and harmful let through are weighted back to the pools ({'wildguard_prompt_train_v1': {'let_through': 73, 'over_flagged': 300, 'right_harmful': 2427, 'right_unharmful': 2200}, 'wildguard_response_train_v1': {'let_through': 105, 'over_flagged': 190, 'right_harmful': 2395, 'right_unharmful': 2310}}).

## WildGuard pooled

| Arm | Override Jev's wrong | Over-flagged | Let through | Accept Jev's right | Cascade accuracy | Harmful let through | Parsed | Out tokens |
|---|---|---|---|---|---|---|---|---|
| standard | 34% (229/668; 31%–38%) | 27% (134/490; 24%–31%) | 53% (95/178; 46%–61%) | 95% (570/600; 93%–96%) | 91.0% | 3.8% | 100% (1268/1268; 100%–100%) | 339.4 |
| clarify | 30% (202/668; 27%–34%) | 20% (99/490; 17%–24%) | 58% (103/178; 51%–65%) | 94% (567/600; 92%–96%) | 90.3% | 3.4% | 100% (1268/1268; 100%–100%) | 367.6 |
| clarify+grounded | 27% (181/668; 24%–31%) | 21% (101/490; 17%–24%) | 45% (80/178; 38%–52%) | 94% (562/600; 91%–95%) | 89.3% | 4.5% | 100% (1268/1268; 100%–100%) | 402.7 |
| examples | 35% (232/668; 31%–38%) | 31% (153/490; 27%–35%) | 44% (79/178; 37%–52%) | 95% (569/600; 93%–96%) | 90.9% | 5.1% | 100% (1268/1268; 100%–100%) | 440.3 |
| grounded | 29% (191/668; 25%–32%) | 22% (110/490; 19%–26%) | 46% (81/178; 38%–53%) | 95% (568/599; 93%–96%) | 90.4% | 4.1% | 100% (1267/1268; 100%–100%) | 384.3 |

| Arm vs. standard (paired) | Override Jev's wrong | Over-flagged | Let through | Accept Jev's right | Cascade accuracy | Harmful let through |
|---|---|---|---|---|---|---|
| clarify | -4.0% [-7.3%, -0.8%] | -7.1% [-10.8%, -3.7%] | +4.5% [-2.2%, +11.2%] | -0.5% [-2.3%, +1.3%] | -0.7% [-2.3%, +1.0%] | -0.5% [-1.8%, +0.9%] |
| clarify+grounded | -7.2% [-10.6%, -3.7%] | -6.7% [-10.6%, -2.9%] | -8.4% [-15.7%, -1.1%] | -1.3% [-3.0%, +0.5%] | -1.7% [-3.4%, +0.1%] | +0.6% [-0.7%, +2.1%] |
| examples | +0.4% [-3.3%, +4.0%] | +3.9% [+0.2%, +7.8%] | -9.0% [-17.4%, -0.6%] | -0.2% [-2.0%, +1.7%] | -0.1% [-1.8%, +1.5%] | +1.2% [-0.3%, +2.9%] |
| grounded | -5.7% [-9.0%, -2.5%] | -4.9% [-8.8%, -1.0%] | -7.9% [-15.2%, -0.6%] | -0.2% [-1.8%, +1.3%] | -0.5% [-2.1%, +0.9%] | +0.3% [-1.2%, +1.8%] |

## wildguard_prompt_train_v1

| Arm | Override Jev's wrong | Over-flagged | Let through | Accept Jev's right | Cascade accuracy | Harmful let through | Parsed | Out tokens |
|---|---|---|---|---|---|---|---|---|
| standard | 33% (122/373; 28%–38%) | 31% (94/300; 26%–37%) | 38% (28/73; 28%–50%) | 93% (278/300; 89%–95%) | 88.3% | 4.9% | 100% (673/673; 99%–100%) | 348.1 |
| clarify | 30% (113/373; 26%–35%) | 26% (78/300; 21%–31%) | 48% (35/73; 37%–59%) | 92% (277/300; 89%–95%) | 87.8% | 3.4% | 100% (673/673; 99%–100%) | 379.5 |
| clarify+grounded | 27% (101/373; 23%–32%) | 25% (75/300; 20%–30%) | 36% (26/73; 26%–47%) | 90% (271/300; 86%–93%) | 85.7% | 5.6% | 100% (673/673; 99%–100%) | 432.9 |
| examples | 38% (140/373; 33%–43%) | 36% (109/300; 31%–42%) | 42% (31/73; 32%–54%) | 93% (280/300; 90%–96%) | 89.2% | 4.2% | 100% (673/673; 99%–100%) | 458.2 |
| grounded | 28% (106/373; 24%–33%) | 26% (78/300; 21%–31%) | 38% (28/73; 28%–50%) | 93% (279/300; 90%–95%) | 88.2% | 4.3% | 100% (673/673; 99%–100%) | 406.1 |

| Arm vs. standard (paired) | Override Jev's wrong | Over-flagged | Let through | Accept Jev's right | Cascade accuracy | Harmful let through |
|---|---|---|---|---|---|---|
| clarify | -2.4% [-7.2%, +2.1%] | -5.3% [-10.3%, +0.0%] | +9.6% [-1.4%, +19.2%] | -0.3% [-3.3%, +3.0%] | -0.5% [-3.2%, +2.1%] | -1.5% [-3.5%, -0.1%] |
| clarify+grounded | -5.6% [-10.5%, -1.1%] | -6.3% [-11.3%, -1.0%] | -2.7% [-13.7%, +9.6%] | -2.3% [-5.3%, +0.7%] | -2.5% [-5.3%, +0.2%] | +0.7% [-1.3%, +2.9%] |
| examples | +4.8% [+0.0%, +9.7%] | +5.0% [-0.3%, +10.3%] | +4.1% [-6.9%, +15.1%] | +0.7% [-2.3%, +3.3%] | +1.0% [-1.6%, +3.7%] | -0.8% [-3.1%, +1.3%] |
| grounded | -4.3% [-9.1%, +0.5%] | -5.3% [-11.0%, +0.0%] | +0.0% [-11.0%, +11.0%] | +0.3% [-2.3%, +2.7%] | -0.0% [-2.4%, +2.3%] | -0.6% [-2.9%, +1.3%] |

## wildguard_response_train_v1

| Arm | Override Jev's wrong | Over-flagged | Let through | Accept Jev's right | Cascade accuracy | Harmful let through | Parsed | Out tokens |
|---|---|---|---|---|---|---|---|---|
| standard | 36% (107/295; 31%–42%) | 21% (40/190; 16%–27%) | 64% (67/105; 54%–72%) | 97% (292/300; 95%–99%) | 93.7% | 2.8% | 100% (595/595; 99%–100%) | 329.6 |
| clarify | 30% (89/295; 25%–36%) | 11% (21/190; 7%–16%) | 65% (68/105; 55%–73%) | 97% (290/300; 94%–98%) | 92.7% | 3.3% | 100% (595/595; 99%–100%) | 354.2 |
| clarify+grounded | 27% (80/295; 22%–32%) | 14% (26/190; 10%–19%) | 51% (54/105; 42%–61%) | 97% (291/300; 94%–98%) | 92.8% | 3.3% | 100% (595/595; 99%–100%) | 368.5 |
| examples | 31% (92/295; 26%–37%) | 23% (44/190; 18%–30%) | 46% (48/105; 37%–55%) | 96% (289/300; 94%–98%) | 92.5% | 6.0% | 100% (595/595; 99%–100%) | 420.1 |
| grounded | 29% (85/295; 24%–34%) | 17% (32/190; 12%–23%) | 50% (53/105; 41%–60%) | 97% (289/299; 94%–98%) | 92.6% | 4.0% | 100% (594/595; 99%–100%) | 359.8 |

| Arm vs. standard (paired) | Override Jev's wrong | Over-flagged | Let through | Accept Jev's right | Cascade accuracy | Harmful let through |
|---|---|---|---|---|---|---|
| clarify | -6.1% [-10.8%, -1.4%] | -10.0% [-15.3%, -5.3%] | +0.9% [-8.6%, +10.5%] | -0.7% [-2.7%, +1.3%] | -1.0% [-2.9%, +1.0%] | +0.6% [-1.5%, +2.8%] |
| clarify+grounded | -9.2% [-14.2%, -4.4%] | -7.4% [-12.6%, -1.6%] | -12.4% [-22.9%, -1.9%] | -0.3% [-2.3%, +1.7%] | -0.9% [-2.9%, +1.1%] | +0.5% [-1.4%, +2.3%] |
| examples | -5.1% [-10.5%, +0.0%] | +2.1% [-3.2%, +7.9%] | -18.1% [-28.6%, -6.7%] | -1.0% [-3.3%, +1.3%] | -1.2% [-3.5%, +1.0%] | +3.2% [+1.1%, +5.9%] |
| grounded | -7.5% [-12.9%, -3.0%] | -4.2% [-9.5%, +1.1%] | -13.3% [-22.9%, -3.8%] | -0.7% [-2.7%, +1.0%] | -1.1% [-3.0%, +0.6%] | +1.2% [-0.9%, +3.5%] |

## Selection (rule fixed before the run)

Rule: {'max_accept_right_drop': 0.02, 'max_let_through_override_drop': 0.03, 'min_parse_rate': 0.95, 'min_gain': 0.05}

| Arm | Gain in override [95%] | Accept-right ok | Let-through ok | Parse ok | Admissible |
|---|---|---|---|---|---|
| clarify | -4.0% [-7.3%, -0.8%] | True | True | True | True |
| clarify+grounded | -7.2% [-10.6%, -3.7%] | True | False | True | False |
| examples | +0.4% [-3.3%, +4.0%] | True | False | True | False |
| grounded | -5.7% [-9.0%, -2.5%] | True | False | True | False |

**Goes to E.117: nothing** (best admissible: clarify).
