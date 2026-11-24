# E.119 comparison: e119-v1-2026-11-24

Status: completed. Spent $1.6962 ({'bedrock': 1.69618}). Reviewer: gpt-oss-120b vs. E.102's gpt-oss-20b, same items, prompt and settings.

## Decision (rule fixed before the run)

120b cascade minus Jev alone, pooled subsample: **-1.3% [-3.1%, +0.3%]**. A stronger reviewer fixes it: **no**.

## Accuracy on E.102's random subsample (paired)

| Scope | Items | Jev alone | 20b alone | 120b alone | 20b cascade | 120b cascade | 120b cascade − Jev | 120b cascade − 20b cascade | 120b alone − 20b alone |
|---|---|---|---|---|---|---|---|---|---|
| civil_toxicity_pool_v1 | 400 | 76% (303/400; 71%–80%) | 75% (301/400; 71%–79%) | 79% (317/400; 75%–83%) | 77% (309/400; 73%–81%) | 77% (307/400; 72%–81%) | +1.0% [-2.8%, +4.8%] | -0.5% [-4.2%, +3.2%] | +4.0% [+0.5%, +7.8%] |
| wildguard_prompt_full_v1 | 400 | 90% (361/400; 87%–93%) | 90% (360/400; 87%–93%) | 89% (357/400; 86%–92%) | 88% (353/400; 85%–91%) | 88% (352/400; 84%–91%) | -2.2% [-4.5%, +0.0%] | -0.2% [-2.5%, +2.0%] | -0.8% [-3.2%, +1.8%] |
| wildguard_response_full_v1 | 400 | 92% (370/400; 90%–95%) | 90% (358/400; 86%–92%) | 90% (359/400; 86%–92%) | 88% (353/400; 85%–91%) | 90% (359/400; 86%–92%) | -2.8% [-5.2%, -0.5%] | +1.5% [-1.2%, +4.0%] | +0.2% [-2.2%, +2.8%] |
| pooled | 1200 | 86% (1034/1200; 84%–88%) | 85% (1019/1200; 83%–87%) | 86% (1033/1200; 84%–88%) | 85% (1015/1200; 82%–87%) | 85% (1018/1200; 83%–87%) | -1.3% [-3.1%, +0.3%] | +0.2% [-1.4%, +1.9%] | +1.2% [-0.5%, +2.8%] |

Error overlap on the subsample (phi of error indicators, alone majority): civil_toxicity_pool_v1: Jev–20b 0.473, Jev–120b 0.5448, 20b–120b 0.6208; wildguard_prompt_full_v1: Jev–20b 0.7051, Jev–120b 0.6478, 20b–120b 0.6376; wildguard_response_full_v1: Jev–20b 0.5836, Jev–120b 0.4984, 20b–120b 0.6371; pooled: Jev–20b 0.573, Jev–120b 0.5711, 20b–120b 0.6378.

## Jev's wrong labels, by the 20b's alone record (5 samples)

| Scope | 20b record | 20b overrides | 120b overrides | Change (paired) | 120b right alone |
|---|---|---|---|---|---|
| civil_toxicity_pool_v1 | never | 7% (19/292; 4%–10%) | 6% (18/292; 4%–10%) | -0.3% [-4.1%, +3.4%] | 5% (16/292; 3%–9%) |
| civil_toxicity_pool_v1 | sometimes | 48% (153/321; 42%–53%) | 34% (110/321; 29%–40%) | -13.4% [-19.6%, -6.9%] | 38% (122/321; 33%–43%) |
| civil_toxicity_pool_v1 | always | 94% (129/137; 89%–97%) | 74% (101/137; 66%–80%) | -20.4% [-28.5%, -12.4%] | 79% (108/137; 71%–85%) |
| civil_toxicity_pool_v1 | all | 40% (301/750; 37%–44%) | 31% (229/750; 27%–34%) | -9.6% [-13.2%, -5.9%] | 33% (246/750; 30%–36%) |
| wildguard_prompt_full_v1 | never | 4% (3/75; 1%–11%) | 8% (6/75; 4%–16%) | +4.0% [-4.0%, +12.0%] | 7% (5/75; 3%–15%) |
| wildguard_prompt_full_v1 | sometimes | 47% (36/76; 37%–58%) | 30% (23/76; 21%–41%) | -17.1% [-31.6%, -4.0%] | 37% (28/76; 27%–48%) |
| wildguard_prompt_full_v1 | always | 85% (11/13; 58%–96%) | 46% (6/13; 23%–71%) | -38.5% [-69.2%, -15.4%] | 62% (8/13; 36%–82%) |
| wildguard_prompt_full_v1 | all | 30% (50/164; 24%–38%) | 21% (35/164; 16%–28%) | -9.2% [-17.1%, -1.2%] | 25% (41/164; 19%–32%) |
| wildguard_response_full_v1 | never | 3% (2/67; 1%–10%) | 3% (2/67; 1%–10%) | +0.0% [-6.0%, +6.0%] | 6% (4/67; 2%–14%) |
| wildguard_response_full_v1 | sometimes | 38% (21/55; 27%–51%) | 38% (21/55; 27%–51%) | +0.0% [-14.5%, +14.5%] | 31% (17/55; 20%–44%) |
| wildguard_response_full_v1 | always | 86% (19/22; 67%–95%) | 86% (19/22; 67%–95%) | +0.0% [-22.7%, +22.7%] | 86% (19/22; 67%–95%) |
| wildguard_response_full_v1 | all | 29% (42/144; 22%–37%) | 29% (42/144; 22%–37%) | +0.0% [-6.9%, +7.6%] | 28% (40/144; 21%–36%) |
| pooled | never | 6% (24/434; 4%–8%) | 6% (26/434; 4%–9%) | +0.5% [-2.8%, +3.7%] | 6% (25/434; 4%–8%) |
| pooled | sometimes | 46% (210/452; 42%–51%) | 34% (154/452; 30%–39%) | -12.4% [-18.1%, -6.6%] | 37% (167/452; 33%–41%) |
| pooled | always | 92% (159/172; 88%–96%) | 73% (126/172; 66%–79%) | -19.2% [-26.2%, -12.2%] | 78% (135/172; 72%–84%) |
| pooled | all | 37% (393/1058; 34%–40%) | 29% (306/1058; 26%–32%) | -8.2% [-11.2%, -5.1%] | 31% (327/1058; 28%–34%) |

## Deference at equal difficulty (each reviewer by its own alone record; E.102's measure, pooled)

- 20b (E.102): all_wrong_proposals +7.8% [+3.1%, +12.2%] (Jev 63% vs 55%, n 1058); eligible +7.0% [+0.8%, +12.6%] (Jev 21% vs 14%, n 351); strict +5.4% [+1.4%, +9.6%] (Jev 8% vs 2%, n 172)
- 120b: all_wrong_proposals +6.1% [+2.2%, +10.2%] (Jev 71% vs 65%, n 1057); eligible +9.6% [+3.8%, +17.4%] (Jev 25% vs 15%, n 326); strict +8.0% [+3.8%, +13.0%] (Jev 10% vs 2%, n 176)
