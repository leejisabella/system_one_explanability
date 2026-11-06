"""Interpretability studies of the reviewer (research_docs/15).

  E.110 (sysone.interp.steering)  is giving in to Jev a direction in the reviewer's residual
                                  stream; does removing it stop the giving in; did training remove it.

The selection, splits and statistics run offline; activation capture and steered generation run on
the GPU machine and import torch only when they run, like sysone.posttrain.sft.
"""
