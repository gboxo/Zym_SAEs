#from SAELens.sae_lens import HookedSAETransformer, SAE, SAEConfig
from sae_lens import HookedSAETransformer, SAE, SAEConfig
from src.utils import load_model, get_sl_model, load_sae
from functools import partial
import torch
import pandas as pd
import os
import pickle as pkl
from tqdm import tqdm

def ablation(activations, hook, ablation_feature):
    # Check prompt processing
    if activations.shape[1] > 1:
        activations[:,:,ablation_feature] = 0
        
    else:
        activations[:,:,ablation_feature] = 0
        return activations



def generate_with_ablation(model: HookedSAETransformer, sae: SAE, prompt: str, ablation_feature: int, max_new_tokens=256, n_samples=10):
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)

    all_outputs = []

    ablation_hook = partial(
        ablation,
        ablation_feature=ablation_feature,
    )
    




    for i in range(n_samples):

        # standard transformerlens syntax for a hook context for generation
        with model.hooks(fwd_hooks=[('blocks.26.hook_resid_pre.hook_sae_acts_post', ablation_hook)]):
            output = model.generate(
                input_ids, 
                temperature=0.9,
                top_k=9, #tbd
                freq_penalty=1.2,
                max_new_tokens=max_new_tokens,
                eos_token_id=1,
                do_sample=True,
                verbose=False,
                ) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.

        all_outputs.append(model.tokenizer.decode(output[0]))

    return all_outputs




cfg = SAEConfig(
    architecture="jumprelu",
    d_in=1280,
    d_sae=10240,
    activation_fn_str="relu",
    apply_b_dec_to_input=True,
    finetuning_scaling_factor=False,
    context_size=256,
    model_name="ZymCTRL",
    hook_name="blocks.26.hook_resid_pre",
    hook_layer=26,
    hook_head_index=None,
    prepend_bos=False,
    dtype="float32",
    normalize_activations="layer_norm",
    dataset_path=None,
    dataset_trust_remote_code=False,
    device="cuda",
    sae_lens_training_version=None,
)

if __name__ == "__main__":

    for model_iteration, data_iteration in [(0,1),(0,2), (1,1),(2,2)]:
        print(f"Model iteration: {model_iteration}, Data iteration: {data_iteration}")
        cs = torch.load("Data/Diffing_Analysis_Data/all_cs.pt")
        cs = cs[f"M{model_iteration}_D{data_iteration}_vs_M0_D0"].cpu().numpy()
        # Load the dataframe
        if True:
            if model_iteration == 0:
                model_path = "AI4PD/ZymCTRL"
            else:
                model_path = f"/users/nferruz/gboxo/Alpha Amylase/output_iteration{model_iteration}" 
            sae_path = f"/users/nferruz/gboxo/Diffing Alpha Amylase/M{model_iteration}_D{data_iteration}/diffing/"
            cfg_sae, sae = load_sae(sae_path)
            thresholds = torch.load(sae_path+"/percentiles/feature_percentile_99.pt")
            thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
            state_dict = sae.state_dict()
            state_dict["threshold"] = thresholds
            del sae


            sae = SAE(cfg)
            sae.load_state_dict(state_dict)
            sae.use_error_term = True

            tokenizer, model = load_model(model_path)
            model = get_sl_model(model, model.config, tokenizer).to("cuda")
            model.add_sae(sae)
        
        path = f"Data/Diffing_Analysis_Data/correlations/top_correlations_M{model_iteration}_D{data_iteration}.pkl"
        with open(path, "rb") as f:
            top_correlations = pkl.load(f)
        feature_indices = top_correlations["feature_indices"]
        

        prompt = "3.2.1.1<sep><start>"


        os.makedirs(f"Data/Diffing_Analysis_Data/ablation/M{model_iteration}_D{data_iteration}", exist_ok=True)


        for ablation_feature in tqdm(feature_indices):
            out = generate_with_ablation(model, sae, prompt, ablation_feature, max_new_tokens=256, n_samples=20)
            with open(f"Data/Diffing_Analysis_Data/ablation/M{model_iteration}_D{data_iteration}/ablation_feature_{ablation_feature}.txt", "w") as f:
                for i, o in enumerate(out):
                    f.write(f">3.2.1.1_{i}\n")
                    f.write(o+"\n")
    del model, sae 
    torch.cuda.empty_cache()