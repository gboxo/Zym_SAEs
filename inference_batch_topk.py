# %%
from utils import load_sae
from sae import BatchTopKSAE, JumpReLUSAE
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




if __name__ == "__main__":

    sae_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_RAW_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_90000"
    cfg, sae, thresholds = load_sae(sae_path, load_thresholds=True)

    jump_relu = convert_to_jumprelu(sae, thresholds)








