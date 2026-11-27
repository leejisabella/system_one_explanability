# E.20a results

**Decision (rule fixed before computing): selective override beats Jev: NO.** Gain over Jev alone on E.102's subsample (Bedrock 20b, first 2 alone answers): -0.3 [-1.8, +1.2].

## Policy chosen on the training pools

| Pool | Jev proposed | Chosen rule | Training gain: never / unanimous / majority (points) | Jev wrong / right share of pool |
|---|---|---|---|---|
| Civil Comments | toxic | **unanimous** | +0.0 / +0.1 / -0.3 | 11.1% / 32.5% |
| Civil Comments | non_toxic | **unanimous** | +0.0 / +0.1 / -2.1 | 13.4% / 43.0% |
| WildGuard prompts | harmful | **majority** | +0.0 / +0.5 / +1.3 | 6.0% / 48.5% |
| WildGuard prompts | unharmful | **never** | +0.0 / -3.4 / -8.5 | 1.5% / 44.0% |
| WildGuard responses | harmful | **unanimous** | +0.0 / +0.2 / -0.4 | 3.8% / 47.9% |
| WildGuard responses | unharmful | **never** | +0.0 / -0.3 / -2.9 | 2.1% / 46.2% |

## Test (E.102 subsample, 400 per pool, pooled with equal weight)

| Variant | Gain over Jev [95%] | Policy accuracy | Fixed / broken (CC, WG-p, WG-r) |
|---|---|---|---|
| **Bedrock 20b, 2 answers (the rule)** | -0.3 [-1.8, +1.2] | 85.8% | 32/35, 5/10, 4/0 |
| Bedrock 20b, 5 answers | +0.4 [-0.8, +1.6] | 86.6% | 23/20, 2/3, 3/0 |
| vLLM 20b, 2 answers | -0.2 [-1.6, +1.3] | 86.0% | 30/32, 4/8, 4/0 |
| vLLM 20b, 5 answers | +0.4 [-0.7, +1.5] | 86.6% | 22/16, 2/4, 1/0 |
| In-sample best rule on the test items (optimistic) | +1.2 [+0.3, +2.0] | 87.3% | 5/2, 8/4, 8/1 |

In-sample best policy: {'civil_toxicity_pool_v1': {'toxic': 'unanimous', 'non_toxic': 'never'}, 'wildguard_prompt_full_v1': {'harmful': 'never', 'unharmful': 'unanimous'}, 'wildguard_response_full_v1': {'harmful': 'majority', 'unharmful': 'never'}}.
