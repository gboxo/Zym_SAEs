from inference_batch_topk import convert_to_jumprelu
from utils import load_sae, load_model, get_ht_model
from sae import BatchTopKSAE, JumpReLUSAE
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def get_acts():
    model_path = "AI4PD/ZymCTRL"
    test_set_path = "micro_brenda.txt"
    is_tokenized = False
    tokenizer, model = load_model(model_path)
    model_config = model.config
    model_config.d_mlp = 5120
    model = get_ht_model(model,model_config).to("cuda")
    with open(test_set_path, "r") as f:
        test_set = f.read()
    test_set = test_set.split("\n")
    test_set = [seq.strip("<pad>") for seq in test_set]
    test_set = [elem for seq in test_set for elem in seq.split("<|endoftext|>")]
    test_set_tokenized = [tokenizer.encode(elem, padding=False, truncation=True, return_tensors="pt", max_length=256) for elem in test_set]

    names_filter = lambda x: x in "blocks.26.hook_resid_pre"
    activations = []
    max_len = 0
    with torch.no_grad():
        for i, elem in enumerate(test_set_tokenized[:100]):
            logits, cache = model.run_with_cache(elem.to("cuda"))
            acts = cache["blocks.26.hook_resid_pre"]
            if acts.shape[1] > max_len:
                max_len = acts.shape[1]
            activations.append(acts.cpu())
    return activations, max_len





activations, max_len = get_acts()

sae_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000/"
cfg, sae, thresholds = load_sae(sae_path, load_thresholds=True)
sae.to("cuda")

jump_relu = convert_to_jumprelu(sae, thresholds)

# ================== 
# The loss for the BatchTopK looks good


acts = activations[0].to("cuda")[0]
x, x_mean, x_std = sae.preprocess_input(acts)
sae_out = sae(acts)
feature_acts = sae_out["feature_acts"]
reconstruct_post = sae_out["sae_out"]
reconstruct_pre = (reconstruct_post - x_mean)/(x_std + 1e-6)
l2 = (reconstruct_pre - x).pow(2).mean(dim = -1)
losses = l2.detach().cpu().numpy()
sns.lineplot(x=range(len(losses)), y=losses)
plt.show()


# ================== 


sae_out_jumprelu = jump_relu.forward(acts, use_pre_enc_bias=True)






