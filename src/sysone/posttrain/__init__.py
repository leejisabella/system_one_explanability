"""Part 2 of the cascade direction: post-training the reviewer to be robust to Jev's errors.

research_docs/14 has the design, the pre-registered success criteria and the runbook. In order:
  E.104 (sysone.experiments.e104)  Jev screens the training pools; the training mix is saved once.
  E.105 (sysone.experiments.e105)  reasoning traces for every example, then the exported datasets.
  E.106 (sysone.posttrain.sft, then sysone.experiments.e106)  SFT on a GPU; the base and trained
                                   reviewers are evaluated on E.102's test plan.
  E.107 (sysone.posttrain.dpo)     DPO on preference pairs, evaluated the same way.
Round 2 (research_docs/15):
  E.108 (sysone.experiments.e108, then sysone.posttrain.sft)  consistency training and
                                   balanced-proposal SFT, two arms on the same items.
  E.109 (sysone.posttrain.grpo)    GRPO in review mode.
  E.110 (sysone.interp.steering)   activation steering, an interpretability study on top.

This package holds the pieces that don't call models: the mix (mix), the training records
(formats), the training configs (train_config) and the GPU training scripts (sft, dpo), which
import torch only when they run.
"""
