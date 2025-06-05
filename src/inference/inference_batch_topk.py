# %%
from ..utils import load_sae
from ..training.sae import BatchTopKSAE, JumpReLUSAE
import torch

"""
To perform inference with batch topk SAEs we need to:
1) Compute the mean threshold for the batch topK sae (this is done one time only)
2) Use the JumpReLU implementation with the new threshold value
"""


def convert_to_jumprelu(sae: BatchTopKSAE, thresholds: torch.tensor) -> JumpReLUSAE:
    sae_state_dict = sae.state_dict()
    sae_state_dict["jumprelu.log_threshold"] = torch.log(thresholds)
    jump_relu = JumpReLUSAE(sae.cfg)
    jump_relu.load_state_dict(sae_state_dict, strict=False)
    return jump_relu



