# E.110 report: e110-v1-2026-11-22

Activation steering on the reviewer (research_docs/15). Directions are fit on training-pool items only.

Data: 605 items (474 fit, 131 held out), 1242 give-in and 451 held traces. The direction is fit within items, on the 353 fit items with both a give-in and a held trace; gaps are paired on the 98 such held-out items.

## 1. Is it there?

Best layer 12 (held-out AUC 0.9984): **found** (found if the best layer's held-out AUC at pre_answer is at least 0.7). Steering layers [11, 12, 13].

| Layer | AUC pre_answer | AUC reasoning_mean | AUC last_prompt | Probe AUC (pre_answer) | Std. gap (pre_answer) | Norm |
|---|---|---|---|---|---|---|
| 0 | 0.735 | 0.638 | 0.479 | 0.992 | 0.71 | 1.6 |
| 1 | 0.810 | 0.637 | 0.540 | 0.998 | 0.93 | 4.1 |
| 2 | 0.821 | 0.582 | 0.486 | 0.997 | 1.13 | 6.2 |
| 3 | 0.859 | 0.567 | 0.502 | 0.998 | 1.21 | 8.7 |
| 4 | 0.839 | 0.527 | 0.512 | 0.998 | 1.35 | 12.1 |
| 5 | 0.895 | 0.540 | 0.509 | 0.998 | 1.30 | 14.7 |
| 6 | 0.880 | 0.529 | 0.511 | 0.998 | 1.29 | 16.3 |
| 7 | 0.879 | 0.522 | 0.502 | 0.998 | 1.15 | 21.6 |
| 8 | 0.889 | 0.516 | 0.494 | 0.998 | 1.36 | 25.4 |
| 9 | 0.904 | 0.522 | 0.534 | 0.999 | 1.50 | 28.5 |
| 10 | 0.977 | 0.529 | 0.547 | 0.998 | 2.45 | 49.5 |
| 11 | 0.941 | 0.540 | 0.540 | 0.999 | 1.84 | 54.1 |
| 12 | 0.998 | 0.555 | 0.503 | 1.000 | 3.35 | 92.7 |
| 13 | 0.998 | 0.578 | 0.489 | 0.999 | 3.64 | 124.4 |
| 14 | 0.998 | 0.598 | 0.490 | 1.000 | 3.49 | 141.5 |
| 15 | 0.996 | 0.619 | 0.510 | 1.000 | 3.20 | 180.2 |
| 16 | 0.995 | 0.663 | 0.518 | 1.000 | 3.01 | 272.2 |
| 17 | 0.997 | 0.690 | 0.512 | 1.000 | 3.14 | 321.1 |
| 18 | 0.994 | 0.689 | 0.491 | 1.000 | 2.49 | 388.6 |
| 19 | 0.986 | 0.678 | 0.504 | 1.000 | 2.22 | 466.9 |
| 20 | 0.989 | 0.660 | 0.460 | 1.000 | 2.27 | 562.3 |
| 21 | 0.987 | 0.644 | 0.507 | 1.000 | 2.19 | 636.6 |
| 22 | 0.910 | 0.633 | 0.489 | 1.000 | 1.37 | 904.6 |
| 23 | 0.908 | 0.626 | 0.489 | 1.000 | 1.33 | 1011.9 |

## α (held-out training items)

Frozen α = 0.5 at layers [11, 12, 13] (no α kept accept-when-right within the limit: flagged). Rule: the smallest α with the largest drop in giving in on the held-out training items, among those whose accept-when-right falls by at most 2% from α = 0.

| α | Jev's wrong accepted | Jev's right accepted |
|---|---|---|
| 0 | 59% (77/131; 50%–67%) | 95% (142/150; 90%–97%) |
| 0.5 | 51% (67/131; 43%–60%) | 87% (131/150; 81%–92%) |
| 1 | 50% (65/130; 42%–58%) | 91% (137/150; 86%–95%) |
| 2 | 44% (55/124; 36%–53%) | 79% (115/145; 72%–85%) |
| 4 | 38% (21/55; 27%–51%) | 42% (22/53; 29%–55%) |

## 2. Is it causal? (test: group A with Jev's wrong label; the Jev-right subsample)

| Reviewer | Jev's wrong accepted (group A) | Jev's right accepted |
|---|---|---|
| none | 27% (100/364; 23%–32%) | 92% (954/1033; 91%–94%) |
| steer | 23% (83/364; 19%–27%) | 88% (912/1034; 86%–90%) |
| random | 17% (12/70; 10%–28%) | 94% (85/90; 88%–98%) |
| vllm | 25% (91/364; 21%–30%) | 90% (930/1034; 88%–92%) |

| Paired change | Jev's wrong accepted (group A) | Jev's right accepted |
|---|---|---|
| none -> steer | -4.7% [-9.9%, +0.5%] | -4.2% [-5.9%, -2.3%] |
| none -> random | -1.4% [-11.4%, +8.6%] | +0.0% [-4.4%, +4.4%] |
| random -> steer | +1.4% [-10.0%, +12.9%] | -4.4% [-11.1%, +2.2%] |
| vllm -> none | +2.5% [-2.8%, +7.4%] | +2.4% [+0.7%, +4.2%] |
| vllm -> steer | -2.2% [-7.7%, +3.3%] | -1.7% [-3.6%, +0.2%] |
| vllm -> random | -2.9% [-14.3%, +8.6%] | +0.0% [-5.6%, +6.7%] |

**Causal: no** against `none` (steering lowers acceptance of Jev's wrong labels on group A (paired 95% interval below 0) and accept-when-right falls by at most 2%). Beats the controls (paired interval below 0): random no, proposal –.

`vllm` is the base's E.106 review on vLLM, for reference; `none` is the unsteered pass here.

## 3. Did training remove it?

No trained model compared yet.
