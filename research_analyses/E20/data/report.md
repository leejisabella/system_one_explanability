# E.20 results

## Decision (rules fixed in configs/experiments/e20.yaml before computing)

- **R1, overlap and deference explain the cascade: PASS** (every held-out prediction inside the observed 95% interval: True; item-level model beats the independence model: True).
- **R2, the ceiling holds: PASS.**

## Part 2: the deference model, fitted on Jev → gpt-oss-20b only

Fitted per pool (item-level model): Civil Comments d_w 0.00, d_r 0.00; WildGuard prompts d_w 0.00, d_r 0.00; WildGuard responses d_w 0.00, d_r 0.00.

| Upstream → 20b | Upstream alone | Observed cascade [95%] | Predicted (item-level) | Predicted (independence) | Coverage |
|---|---|---|---|---|---|
| Jev (fitted) | 86.2% | 82.6% [80.5%, 84.6%] | 83.0% ✓ | 82.6% | 400/400, 400/400, 400/400 |
| Llama 4 Maverick | 84.5% | 83.0% [80.9%, 85.0%] | 83.5% ✓ | 81.4% | 376/400, 394/399, 400/400 |
| Qwen3-32B | 85.7% | 83.1% [80.9%, 85.1%] | 83.3% ✓ | 82.2% | 381/400, 398/400, 400/400 |
| Nova Lite | 85.6% | 83.1% [81.1%, 85.2%] | 83.3% ✓ | 81.9% | 381/399, 394/398, 400/400 |
| matched random | 86.2% | 82.9% [80.7%, 84.9%] | 83.0% ✓ | 82.6% | 400/400, 400/400, 400/400 |

Mean absolute error over the held-out cascades: item-level 0.3 points, independence 1.0 points.

Jev → 120b (E.119): observed 84.9%; predicted from the 120b's alone answers with the 20b's deference 85.1%, with its own fitted deference 84.9%.

## Part 1: the ceiling

Best gain over the upstream alone of any policy that uses only the reviewer's own alone answers (thresholds on its vote share, chosen separately per proposed label, in-sample, so optimistic). Pooled, equal pool weight.

| Upstream | Reviewer | Alone answers | phi (errors) | Ceiling [95%] |
|---|---|---|---|---|
| Jev | balanced proposals (E.108) | 1 | 0.5098 | +0.6 [+0.1, +2.4] |
| Jev | gpt-oss-20b (E.106) | 5 | 0.5601 | +1.6 [+0.7, +3.2] |
| Jev | consistency (E.108) | 1 | 0.5195 | +0.4 [+0.0, +2.1] |
| Jev | DPO (E.107) | 1 | 0.5304 | +0.5 [+0.2, +2.5] |
| Jev | gpt-oss-20b (E.102, Bedrock) | 5 | 0.573 | +2.2 [+1.3, +4.2] |
| Jev | gpt-oss-120b (E.119) | 5 | 0.5711 | +2.8 [+1.5, +4.5] |
| Jev | Nova (1 answer) | 1 | 0.56 | +0.1 [+0.0, +1.5] |
| Jev | Llama (1 answer) | 1 | 0.6242 | +0.0 [+0.0, +0.8] |
| Jev | Qwen3-32B (1 answer) | 1 | 0.6308 | +0.4 [+0.1, +1.5] |
| Jev | Qwen3-32B (E.102 reference) | 3 | 0.6593 | +0.5 [+0.1, +1.7] |
| Jev | SFT plain (E.106) | 1 | 0.5174 | +0.3 [+0.0, +1.9] |
| Jev | SFT review (E.106) | 1 | 0.542 | +0.5 [+0.2, +1.9] |
| Nova Lite | balanced proposals (E.108) | 1 | 0.4224 | +2.0 [+1.1, +4.2] |
| Nova Lite | gpt-oss-20b (E.106) | 5 | 0.4514 | +3.2 [+2.0, +5.6] |
| Nova Lite | consistency (E.108) | 1 | 0.438 | +1.6 [+0.7, +3.6] |
| Nova Lite | DPO (E.107) | 1 | 0.4612 | +1.8 [+1.0, +3.9] |
| Nova Lite | gpt-oss-20b (E.102, Bedrock) | 5 | 0.4313 | +3.6 [+2.6, +5.9] |
| Nova Lite | gpt-oss-120b (E.119) | 5 | 0.4519 | +4.3 [+2.8, +6.3] |
| Nova Lite | Llama (1 answer) | 1 | 0.5951 | +0.6 [+0.1, +1.7] |
| Nova Lite | Qwen3-32B (1 answer) | 1 | 0.6293 | +1.3 [+0.7, +2.9] |
| Nova Lite | Qwen3-32B (E.102 reference) | 3 | 0.6371 | +1.5 [+1.0, +3.2] |
| Nova Lite | SFT plain (E.106) | 1 | 0.3917 | +1.8 [+1.0, +3.9] |
| Nova Lite | SFT review (E.106) | 1 | 0.4214 | +1.9 [+1.1, +3.8] |
| Llama 4 Maverick | balanced proposals (E.108) | 1 | 0.4738 | +3.5 [+1.9, +5.4] |
| Llama 4 Maverick | gpt-oss-20b (E.106) | 5 | 0.505 | +4.8 [+3.3, +6.8] |
| Llama 4 Maverick | consistency (E.108) | 1 | 0.4814 | +2.8 [+1.5, +4.8] |
| Llama 4 Maverick | DPO (E.107) | 1 | 0.5225 | +3.4 [+1.8, +5.3] |
| Llama 4 Maverick | gpt-oss-20b (E.102, Bedrock) | 5 | 0.5124 | +5.2 [+3.8, +7.7] |
| Llama 4 Maverick | gpt-oss-120b (E.119) | 5 | 0.4882 | +5.9 [+4.0, +7.9] |
| Llama 4 Maverick | Nova (1 answer) | 1 | 0.5951 | +2.2 [+0.8, +4.1] |
| Llama 4 Maverick | Qwen3-32B (1 answer) | 1 | 0.6423 | +2.3 [+1.1, +4.0] |
| Llama 4 Maverick | Qwen3-32B (E.102 reference) | 3 | 0.6495 | +2.2 [+1.2, +4.2] |
| Llama 4 Maverick | SFT plain (E.106) | 1 | 0.4793 | +3.1 [+1.7, +4.9] |
| Llama 4 Maverick | SFT review (E.106) | 1 | 0.4772 | +3.1 [+1.4, +4.8] |
| matched random | balanced proposals (E.108) | 1 | -0.0042 | +2.1 [+0.8, +4.1] |
| matched random | gpt-oss-20b (E.106) | 5 | 0.0124 | +2.9 [+1.7, +5.7] |
| matched random | consistency (E.108) | 1 | -0.0154 | +1.8 [+0.6, +3.9] |
| matched random | DPO (E.107) | 1 | 0.0133 | +1.2 [+0.4, +3.8] |
| matched random | gpt-oss-20b (E.102, Bedrock) | 5 | 0.0143 | +3.5 [+2.2, +6.1] |
| matched random | gpt-oss-120b (E.119) | 5 | -0.0207 | +3.7 [+2.0, +6.3] |
| matched random | Nova (1 answer) | 1 | 0.0054 | +1.1 [+0.1, +3.2] |
| matched random | Llama (1 answer) | 1 | 0.0474 | +0.4 [+0.0, +2.0] |
| matched random | Qwen3-32B (1 answer) | 1 | 0.0476 | +0.9 [+0.3, +2.8] |
| matched random | Qwen3-32B (E.102 reference) | 3 | 0.0588 | +1.5 [+0.7, +3.6] |
| matched random | SFT plain (E.106) | 1 | 0.0507 | +1.8 [+0.8, +3.8] |
| matched random | SFT review (E.106) | 1 | 0.0296 | +2.2 [+1.0, +4.1] |
| Qwen3-32B | balanced proposals (E.108) | 1 | 0.4631 | +1.9 [+0.9, +4.0] |
| Qwen3-32B | gpt-oss-20b (E.106) | 5 | 0.4976 | +3.8 [+2.3, +5.7] |
| Qwen3-32B | consistency (E.108) | 1 | 0.4592 | +1.8 [+0.9, +3.8] |
| Qwen3-32B | DPO (E.107) | 1 | 0.4761 | +1.8 [+0.8, +4.1] |
| Qwen3-32B | gpt-oss-20b (E.102, Bedrock) | 5 | 0.5038 | +4.5 [+3.0, +6.5] |
| Qwen3-32B | gpt-oss-120b (E.119) | 5 | 0.4758 | +4.4 [+3.0, +6.2] |
| Qwen3-32B | Nova (1 answer) | 1 | 0.6293 | +1.2 [+0.2, +2.7] |
| Qwen3-32B | Llama (1 answer) | 1 | 0.6423 | +0.6 [+0.0, +1.7] |
| Qwen3-32B | Qwen3-32B (E.102 reference) | 3 | 0.887 | +0.5 [+0.2, +1.7] |
| Qwen3-32B | SFT plain (E.106) | 1 | 0.4572 | +1.5 [+0.8, +3.6] |
| Qwen3-32B | SFT review (E.106) | 1 | 0.4679 | +1.8 [+1.0, +3.6] |

Spearman correlation of the ceiling with phi across all pairs: -0.27.

## R2: observed cascades against their ceilings

| Upstream | Reviewer | Observed gain [95%] | Ceiling | Above? | In rule |
|---|---|---|---|---|---|
| Nova Lite | balanced proposals (E.108) | -1.9 [-3.8, +0.2] | +2.0 | no | no (1 alone answer) |
| Nova Lite | gpt-oss-20b (E.106) | -2.5 [-4.7, -0.1] | +3.2 | no | yes |
| Nova Lite | consistency (E.108) | -1.8 [-4.2, +0.4] | +1.6 | no | no (1 alone answer) |
| Nova Lite | DPO (E.107) | +0.4 [-1.6, +2.4] | +1.8 | no | no (1 alone answer) |
| Nova Lite | gpt-oss-20b (E.102, Bedrock) | -4.3 [-6.2, -2.6] | +3.6 | no | yes |
| Nova Lite | blind-first (E.103) | -3.8 [-5.7, -1.7] | +3.6 | no | yes |
| Nova Lite | calibrated warning (E.103) | -4.7 [-6.6, -2.7] | +3.6 | no | yes |
| Nova Lite | consider-the-opposite (E.103) | -4.9 [-7.0, -2.9] | +3.6 | no | yes |
| Nova Lite | warning (E.103) | -4.0 [-6.1, -2.0] | +3.6 | no | yes |
| Nova Lite | gpt-oss-120b (E.119) | -3.7 [-5.5, -2.0] | +4.3 | no | yes |
| Nova Lite | SFT plain (E.106) | -0.9 [-2.9, +1.1] | +1.8 | no | no (1 alone answer) |
| Nova Lite | SFT review (E.106) | -1.2 [-3.5, +1.1] | +1.9 | no | no (1 alone answer) |
| Llama 4 Maverick | balanced proposals (E.108) | -0.5 [-2.6, +1.5] | +3.5 | no | no (1 alone answer) |
| Llama 4 Maverick | gpt-oss-20b (E.106) | -1.4 [-3.6, +0.8] | +4.8 | no | yes |
| Llama 4 Maverick | consistency (E.108) | +0.3 [-1.7, +2.4] | +2.8 | no | no (1 alone answer) |
| Llama 4 Maverick | DPO (E.107) | +1.6 [-0.2, +3.6] | +3.4 | no | no (1 alone answer) |
| Llama 4 Maverick | gpt-oss-20b (E.102, Bedrock) | -2.6 [-4.7, -0.7] | +5.2 | no | yes |
| Llama 4 Maverick | blind-first (E.103) | -2.1 [-3.8, -0.1] | +5.2 | no | yes |
| Llama 4 Maverick | calibrated warning (E.103) | -2.8 [-4.8, -0.8] | +5.2 | no | yes |
| Llama 4 Maverick | consider-the-opposite (E.103) | -3.1 [-5.5, -1.0] | +5.2 | no | yes |
| Llama 4 Maverick | warning (E.103) | -2.6 [-4.8, -0.8] | +5.2 | no | yes |
| Llama 4 Maverick | gpt-oss-120b (E.119) | -2.2 [-4.0, -0.4] | +5.9 | no | yes |
| Llama 4 Maverick | SFT plain (E.106) | +0.8 [-1.2, +3.1] | +3.1 | no | no (1 alone answer) |
| Llama 4 Maverick | SFT review (E.106) | +0.5 [-1.6, +2.8] | +3.1 | no | no (1 alone answer) |
| Jev | balanced proposals (E.108) | -1.9 [-3.9, -0.2] | +0.6 | no | no (1 alone answer) |
| Jev | gpt-oss-20b (E.106) | -3.6 [-5.6, -1.4] | +1.6 | no | yes |
| Jev | consistency (E.108) | -2.6 [-4.6, -0.7] | +0.4 | no | no (1 alone answer) |
| Jev | DPO (E.107) | -0.2 [-2.0, +1.6] | +0.5 | no | no (1 alone answer) |
| Jev | gpt-oss-20b (E.102, Bedrock) | -1.7 [-3.6, +0.2] | +2.2 | no | yes |
| Jev | blind-first (E.103) | -2.1 [-4.3, -0.1] | +2.2 | no | yes |
| Jev | calibrated warning (E.103) | -2.2 [-4.3, -0.3] | +2.2 | no | yes |
| Jev | consider-the-opposite (E.103) | -2.0 [-3.9, +0.3] | +2.2 | no | yes |
| Jev | warning (E.103) | -1.9 [-3.8, +0.2] | +2.2 | no | yes |
| Jev | gpt-oss-120b (E.119) | -1.3 [-3.1, +0.4] | +2.8 | no | yes |
| Jev | SFT plain (E.106) | -0.8 [-2.7, +1.1] | +0.3 | no | no (1 alone answer) |
| Jev | SFT review (E.106) | -1.2 [-3.2, +0.7] | +0.5 | no | no (1 alone answer) |
| matched random | balanced proposals (E.108) | -1.6 [-4.5, +1.5] | +2.1 | no | no (1 alone answer) |
| matched random | gpt-oss-20b (E.106) | -3.3 [-6.5, -0.4] | +2.9 | no | yes |
| matched random | consistency (E.108) | -2.4 [-5.2, +0.4] | +1.8 | no | no (1 alone answer) |
| matched random | DPO (E.107) | -1.0 [-3.8, +1.8] | +1.2 | no | no (1 alone answer) |
| matched random | gpt-oss-20b (E.102, Bedrock) | -1.2 [-4.1, +1.6] | +3.5 | no | yes |
| matched random | blind-first (E.103) | -2.2 [-5.2, +0.7] | +3.5 | no | yes |
| matched random | calibrated warning (E.103) | -2.3 [-5.3, +0.4] | +3.5 | no | yes |
| matched random | consider-the-opposite (E.103) | -3.0 [-5.7, -0.2] | +3.5 | no | yes |
| matched random | warning (E.103) | -2.0 [-4.8, +0.9] | +3.5 | no | yes |
| matched random | gpt-oss-120b (E.119) | -0.7 [-3.5, +2.1] | +3.7 | no | yes |
| matched random | SFT plain (E.106) | -0.4 [-3.2, +2.5] | +1.8 | no | no (1 alone answer) |
| matched random | SFT review (E.106) | -1.3 [-4.2, +1.6] | +2.2 | no | no (1 alone answer) |
| Qwen3-32B | balanced proposals (E.108) | -1.1 [-3.1, +0.9] | +1.9 | no | no (1 alone answer) |
| Qwen3-32B | gpt-oss-20b (E.106) | -2.6 [-4.9, -0.4] | +3.8 | no | yes |
| Qwen3-32B | consistency (E.108) | -1.4 [-3.4, +0.6] | +1.8 | no | no (1 alone answer) |
| Qwen3-32B | DPO (E.107) | +0.4 [-1.5, +2.3] | +1.8 | no | no (1 alone answer) |
| Qwen3-32B | gpt-oss-20b (E.102, Bedrock) | -3.6 [-5.4, -1.6] | +4.5 | no | yes |
| Qwen3-32B | blind-first (E.103) | -3.6 [-5.6, -1.6] | +4.5 | no | yes |
| Qwen3-32B | calibrated warning (E.103) | -3.8 [-5.8, -1.8] | +4.5 | no | yes |
| Qwen3-32B | consider-the-opposite (E.103) | -4.1 [-6.3, -2.1] | +4.5 | no | yes |
| Qwen3-32B | warning (E.103) | -3.1 [-5.1, -1.4] | +4.5 | no | yes |
| Qwen3-32B | gpt-oss-120b (E.119) | -3.1 [-4.8, -1.4] | +4.4 | no | yes |
| Qwen3-32B | SFT plain (E.106) | -0.2 [-2.2, +1.7] | +1.5 | no | no (1 alone answer) |
| Qwen3-32B | SFT review (E.106) | -0.8 [-2.9, +1.3] | +1.8 | no | no (1 alone answer) |

## Frontier (Jev upstream): fix rate on Jev's mistakes vs break rate on its right labels

| Reviewer | Pool | Fix | Break | Gain | d_w | d_r | Alone fix | Alone break |
|---|---|---|---|---|---|---|---|---|
| balanced proposals (E.108) | Civil Comments | 43.0% | 16.2% | -1.8 | 0.11 | 0.00 | 48.4% | 14.2% |
| balanced proposals (E.108) | WildGuard prompts | 22.4% | 3.3% | -0.8 | 0.20 | 0.29 | 28.2% | 4.7% |
| balanced proposals (E.108) | WildGuard responses | 31.1% | 5.9% | -3.2 | 0.00 | 0.00 | 26.7% | 5.1% |
| gpt-oss-20b (E.106) | Civil Comments | 43.8% | 15.5% | -1.1 | 0.00 | 0.00 | 39.8% | 15.4% |
| gpt-oss-20b (E.106) | WildGuard prompts | 23.5% | 7.2% | -4.2 | 0.00 | 0.00 | 22.1% | 6.0% |
| gpt-oss-20b (E.106) | WildGuard responses | 31.5% | 8.4% | -5.4 | 0.00 | 0.00 | 28.0% | 6.7% |
| consistency (E.108) | Civil Comments | 42.1% | 15.2% | -1.3 | 0.11 | 0.00 | 47.4% | 13.9% |
| consistency (E.108) | WildGuard prompts | 22.9% | 5.8% | -3.0 | 0.00 | 0.00 | 20.5% | 5.6% |
| consistency (E.108) | WildGuard responses | 26.8% | 5.9% | -3.5 | 0.11 | 0.00 | 30.0% | 5.1% |
| DPO (E.107) | Civil Comments | 44.9% | 11.2% | +2.4 | 0.00 | 0.15 | 43.3% | 13.2% |
| DPO (E.107) | WildGuard prompts | 26.9% | 4.4% | -1.4 | 0.05 | 0.00 | 28.2% | 4.4% |
| DPO (E.107) | WildGuard responses | 36.9% | 4.9% | -1.7 | 0.00 | 0.18 | 30.0% | 5.9% |
| gpt-oss-20b (E.102, Bedrock) | Civil Comments | 45.4% | 13.2% | +1.0 | 0.00 | 0.15 | 43.1% | 15.5% |
| gpt-oss-20b (E.102, Bedrock) | WildGuard prompts | 23.1% | 4.7% | -2.0 | 0.24 | 0.13 | 30.3% | 5.4% |
| gpt-oss-20b (E.102, Bedrock) | WildGuard responses | 30.0% | 6.8% | -4.0 | 0.12 | 0.00 | 34.0% | 6.8% |
| blind-first (E.103) | Civil Comments | 49.5% | 14.5% | +1.0 | 0.00 | 0.06 | 43.1% | 15.5% |
| blind-first (E.103) | WildGuard prompts | 30.8% | 5.8% | -2.2 | 0.00 | 0.00 | 30.3% | 5.4% |
| blind-first (E.103) | WildGuard responses | 13.3% | 6.5% | -5.0 | 0.61 | 0.04 | 34.0% | 6.8% |
| calibrated warning (E.103) | Civil Comments | 42.8% | 16.2% | -1.9 | 0.01 | 0.00 | 43.1% | 15.5% |
| calibrated warning (E.103) | WildGuard prompts | 29.5% | 4.2% | -0.9 | 0.03 | 0.23 | 30.3% | 5.4% |
| calibrated warning (E.103) | WildGuard responses | 23.3% | 6.2% | -4.0 | 0.31 | 0.08 | 34.0% | 6.8% |
| consider-the-opposite (E.103) | Civil Comments | 49.0% | 14.5% | +0.9 | 0.00 | 0.06 | 43.1% | 15.5% |
| consider-the-opposite (E.103) | WildGuard prompts | 30.8% | 5.5% | -2.0 | 0.00 | 0.00 | 30.3% | 5.4% |
| consider-the-opposite (E.103) | WildGuard responses | 36.7% | 8.1% | -4.8 | 0.00 | 0.00 | 34.0% | 6.8% |
| warning (E.103) | Civil Comments | 46.9% | 14.8% | +0.1 | 0.00 | 0.04 | 43.1% | 15.5% |
| warning (E.103) | WildGuard prompts | 28.2% | 6.1% | -2.8 | 0.07 | 0.00 | 30.3% | 5.4% |
| warning (E.103) | WildGuard responses | 33.3% | 5.9% | -3.0 | 0.02 | 0.12 | 34.0% | 6.8% |
| gpt-oss-120b (E.119) | Civil Comments | 34.5% | 9.6% | +1.1 | 0.05 | 0.01 | 36.3% | 9.7% |
| gpt-oss-120b (E.119) | WildGuard prompts | 15.4% | 4.2% | -2.2 | 0.40 | 0.11 | 25.6% | 4.7% |
| gpt-oss-120b (E.119) | WildGuard responses | 26.7% | 5.1% | -2.8 | 0.23 | 0.14 | 34.7% | 6.0% |
| SFT plain (E.106) | Civil Comments | 48.5% | 12.9% | +2.0 | 0.00 | 0.13 | 42.3% | 14.8% |
| SFT plain (E.106) | WildGuard prompts | 25.0% | 5.8% | -2.8 | 0.25 | 0.00 | 33.3% | 4.4% |
| SFT plain (E.106) | WildGuard responses | 30.9% | 4.3% | -1.7 | 0.00 | 0.24 | 30.0% | 5.7% |
| SFT review (E.106) | Civil Comments | 46.6% | 12.9% | +1.5 | 0.00 | 0.07 | 38.1% | 13.9% |
| SFT review (E.106) | WildGuard prompts | 28.6% | 5.5% | -2.2 | 0.00 | 0.00 | 28.2% | 4.4% |
| SFT review (E.106) | WildGuard responses | 29.8% | 5.7% | -3.0 | 0.19 | 0.00 | 36.7% | 5.7% |
