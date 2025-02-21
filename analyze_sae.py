import torch
from sae_lens import SAE, SAEConfig
import os


train_path = "checkpoints/im4ex11m/final_15360000/"
sae_weights = torch.load(train_path+"sae_weights.safetensors", weights_only=False, map_location='cpu')




sparsity_path = train_path+"sparsity.safetensors"
sparsity = torch.load(sparsity_path, weights_only=False, map_location='cpu', )







