# E.105 report: e105-v1-2026-11-04

Status: completed. Spent $2.7600 ({'bedrock': 2.760044}). 38036 traces.

| Pool | Group | Format | Examples | Solved by the reviewer | By the teacher | Unsolved |
|---|---|---|---|---|---|---|
| civil_toxicity_train_v1 | alone | alone | 1357 | 1293 | 34 | 30 |
| civil_toxicity_train_v1 | jev_right | alone | 2714 | 2570 | 71 | 73 |
| civil_toxicity_train_v1 | jev_right | review | 2714 | 2601 | 51 | 62 |
| civil_toxicity_train_v1 | jev_wrong | alone | 1357 | 689 | 79 | 589 |
| civil_toxicity_train_v1 | jev_wrong | review | 1357 | 732 | 65 | 560 |
| wildguard_prompt_train_v1 | alone | alone | 373 | 362 | 6 | 5 |
| wildguard_prompt_train_v1 | jev_right | alone | 746 | 730 | 4 | 12 |
| wildguard_prompt_train_v1 | jev_right | review | 746 | 729 | 3 | 14 |
| wildguard_prompt_train_v1 | jev_wrong | alone | 373 | 208 | 27 | 138 |
| wildguard_prompt_train_v1 | jev_wrong | review | 373 | 209 | 24 | 140 |
| wildguard_response_train_v1 | alone | alone | 295 | 293 | 0 | 2 |
| wildguard_response_train_v1 | jev_right | alone | 590 | 585 | 2 | 3 |
| wildguard_response_train_v1 | jev_right | review | 590 | 585 | 2 | 3 |
| wildguard_response_train_v1 | jev_wrong | alone | 295 | 155 | 13 | 127 |
| wildguard_response_train_v1 | jev_wrong | review | 295 | 154 | 16 | 125 |

Exported `part2_all_v1`: dpo_review.train.jsonl 1019, dpo_review.val.jsonl 64, sft_plain.train.jsonl 6592, sft_plain.val.jsonl 370, sft_review.train.jsonl 6592, sft_review.val.jsonl 370

Exported `part2_no_responses_v1`: dpo_review.train.jsonl 900, dpo_review.val.jsonl 58, sft_plain.train.jsonl 5604, sft_plain.val.jsonl 328, sft_review.train.jsonl 5604, sft_review.val.jsonl 328
