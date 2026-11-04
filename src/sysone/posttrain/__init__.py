"""Part 2 of the cascade direction: post-training the reviewer to be robust to Jev's errors.

research_docs/14 has the design, the pre-registered success criteria and the runbook. In order:
  E.104 (sysone.experiments.e104)  Jev screens the training pools; the training mix is saved once.
  E.105 (sysone.experiments.e105)  reasoning traces for every example, then the exported datasets.
  E.106 (sysone.posttrain.sft, then sysone.experiments.e106)  SFT on a GPU; the base and trained
                                   reviewers are evaluated on E.102's test plan.
  E.107 (sysone.posttrain.dpo)     DPO on preference pairs, evaluated the same way.
  E.108                            generalization: held-out upstream models and a held-out domain.

This package holds the pieces that don't call models: the mix (mix), the training records
(formats), the training configs (train_config) and the GPU training scripts (sft, dpo), which
import torch only when they run.
"""
