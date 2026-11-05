# E.104 report: e104-v1-2026-11-04

Status: completed. Spent $1.4682.

| Training pool | Jev error rate | Test pool | Jev error rate on test |
|---|---|---|---|
| civil_toxicity_train_v1 | 24.5% | civil_toxicity_pool_v1 | 25.0% |
| wildguard_prompt_train_v1 | 7.5% | wildguard_prompt_full_v1 | 9.7% |
| wildguard_response_train_v1 | 5.9% | wildguard_response_full_v1 | 8.4% |

| Training pool | Mix (group/split: rows) |
|---|---|
| civil_toxicity_train_v1 | alone/train 1280, alone/val 77, jev_right/train 2567, jev_right/val 147, jev_wrong/train 1302, jev_wrong/val 55 |
| wildguard_prompt_train_v1 | alone/train 350, alone/val 23, jev_right/train 710, jev_right/val 36, jev_wrong/train 348, jev_wrong/val 25 |
| wildguard_response_train_v1 | alone/train 282, alone/val 13, jev_right/train 567, jev_right/val 23, jev_wrong/train 283, jev_wrong/val 12 |
