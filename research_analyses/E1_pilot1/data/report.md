# Pilot 1 report pilot1-v1b-2026-10-28

Explainer decisions: 1128 · explanations: 370 · simulations: 669

**Explanations parsed**

| Explainer / condition | Parsed |
|---|---|
| explainer_gemini/free_form | 100% (74/74) |
| explainer_gemini/structured | 100% (74/74) |
| explainer_open/free_form | 100% (74/74) |
| explainer_open/structured | 89% (66/74) |
| none/decision_probabilities | 100% (74/74) |

**Control (no explanation):** simulator parse 99% (113/114); agreement with Jev 90% (102/113)

**Faithfulness on hidden variants** (A-direct is the main score; the others are checks, not blended)

| Explainer / condition | Scored | A-direct | Control | NSG | Covered | A-direct vs A-executed | A-executed vs Jev |
|---|---|---|---|---|---|---|---|
| explainer_gemini/free_form | 113 | 89% | 90% | -0.0909 | 89% (101/114) | 96% | 89% |
| explainer_gemini/structured | 112 | 84% | 90% | -0.6364 | 94% (106/113) | 96% | 85% |
| explainer_open/free_form | 112 | 90% | 90% | 0.0 | 82% (93/113) | 97% | 90% |
| explainer_open/structured | 98 | 93% | 91% | 0.2222 | 97% (96/99) | 91% | 88% |
| none/decision_probabilities | 113 | 90% | 90% | 0.0 | 56% (64/114) | 92% | 85% |

NSG = (A-direct − control) / (1 − control). A low A-direct vs A-executed agreement means the simulator reasoned on its own instead of from the explanation.

**Where Jev departs from the policy** (hidden variants whose policy outcome differs from Jev's decision; knowing the policy alone predicts these wrong)

| Explainer / condition | A-direct | Control (same variants) |
|---|---|---|
| explainer_gemini/free_form | 27% (3/11) | 27% (3/11) |
| explainer_gemini/structured | 36% (4/11) | 27% (3/11) |
| explainer_open/free_form | 36% (4/11) | 27% (3/11) |
| explainer_open/structured | 64% (7/11) | 27% (3/11) |
| none/decision_probabilities | 45% (5/11) | 27% (3/11) |

Simulator replies repaired before parsing (spelled-out digits such as "0. nine"): 87 of 669.

**Option B: structured explanations run directly against the records**

| Explainer | Variants | Decidable | Agreement with Jev |
|---|---|---|---|
| explainer_gemini | 114 | 96% | 84% |
| explainer_open | 99 | 98% | 88% |

**Jev–explainer disagreement** (independent majority decisions)

| Explainer | Parsed | Originals | Variants | All | Explainer unanimous | Explainer vs policy |
|---|---|---|---|---|---|---|
| explainer_gemini | 100% (564/564) | 7% (5/74) | 10% (11/114) | 9% (16/188) | 95% | 99% |
| explainer_open | 100% (564/564) | 16% (12/74) | 8% (9/114) | 11% (21/188) | 89% | 93% |

**Model versions:** {'us.meta.llama4-maverick-17b-instruct-v1:0': 712, 'gemini-3.1-flash-lite': 712, 'openai.gpt-oss-120b-1:0': 669}

**Cost:** $0.4057 total; by model {'bedrock/openai.gpt-oss-120b-1:0': 0.140155, 'bedrock/us.meta.llama4-maverick-17b-instruct-v1:0': 0.118901, 'gemini/gemini-3.1-flash-lite': 0.146628}; by step {'explainer_decision': 0.137011, 'explanation': 0.128518, 'simulation': 0.140155}

Contestability scoring against the verified defects needs the LLM judge and is not computed yet.
