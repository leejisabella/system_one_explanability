# E.22 comparison: e22-v1-2026-11-24

Status: completed. Spent $4.1901. Reviews 5181, parsed 100.0%.

## Decision (rule fixed before the run)

- Mistral cascade − Jev alone, WildGuard subsample pooled: **-3.4% [-5.5%, -1.1%]** (beats Jev: False).
- Harmful let through: Jev 3.8%, Mistral cascade 9.5% (increase +5.8 points; guard ok: False). For reference, 20b cascade 3.5%.
- **Another family fixes it on WildGuard: NO.**

## Accuracy on the WildGuard subsample (paired)

| Scope | Items | Jev alone | 20b alone | Mistral alone | 20b cascade | Mistral cascade | Mistral cascade − Jev | Mistral − 20b cascade | Mistral − 20b alone |
|---|---|---|---|---|---|---|---|---|---|
| WildGuard prompts | 400 | 90% (361/400; 87%–93%) | 90% (360/400; 87%–93%) | 85% (340/400; 81%–88%) | 88% (353/400; 85%–91%) | 84% (335/400; 80%–87%) | -6.5% [-10.0%, -3.2%] | -4.5% [-8.5%, -0.8%] | -5.0% [-9.0%, -1.2%] |
| WildGuard responses | 400 | 92% (370/400; 90%–95%) | 90% (358/400; 86%–92%) | 92% (370/400; 90%–95%) | 88% (353/400; 85%–91%) | 92% (369/400; 89%–94%) | -0.2% [-2.8%, +2.2%] | +4.0% [+0.8%, +7.5%] | +3.0% [-0.2%, +6.5%] |
| pooled | 800 | 91% (731/800; 89%–93%) | 90% (718/800; 87%–92%) | 89% (710/800; 86%–91%) | 88% (706/800; 86%–90%) | 88% (704/800; 86%–90%) | -3.4% [-5.5%, -1.1%] | -0.2% [-2.6%, +2.1%] | -1.0% [-3.4%, +1.6%] |

Error overlap (phi, pooled): Jev–20b 0.6449, Jev–Mistral 0.4402, 20b–Mistral 0.3231.

Mistral cascade − 120b cascade (E.119), same items: -0.9% [-3.1%, +1.5%].
Without step 0's read items (7 in the subsample): Mistral cascade − Jev -3.7% [-5.8%, -1.5%].

## Jev's wrong labels, by the 20b's alone record (pooled)

| 20b record | 20b overrides | Mistral overrides | Change (paired) | Mistral right alone |
|---|---|---|---|---|
| never | 4% (5/142; 2%–8%) | 37% (52/142; 29%–45%) | +33.1% [+24.6%, +41.5%] | 42% (60/142; 34%–50%) |
| sometimes | 44% (57/131; 35%–52%) | 56% (73/131; 47%–64%) | +12.2% [+0.8%, +22.9%] | 57% (75/131; 49%–65%) |
| always | 86% (30/35; 71%–94%) | 46% (16/35; 30%–62%) | -40.0% [-60.0%, -22.9%] | 46% (16/35; 30%–62%) |
| all | 30% (92/308; 25%–35%) | 46% (141/308; 40%–51%) | +15.9% [+9.1%, +23.1%] | 49% (151/308; 43%–55%) |
