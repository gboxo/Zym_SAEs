from argparse import ArgumentParser
from src.tools.generate.generate_utils import load_config
from sae_lens import HookedSAETransformer, SAE, SAEConfig
from src.utils import load_model, get_sl_model, load_sae
from functools import partial
import torch
import os
import pickle as pkl
from tqdm import tqdm

def ablation(activations:torch.Tensor, hook, ablation_feature:list[int]):
    # Check prompt processing
    if activations.shape[1] > 1:
        pass
        
    else:
        activations[:,:,ablation_feature] = 0
    
    return activations




def generate_with_ablation(model: HookedSAETransformer, sae: SAE, prompt: str, ablation_feature: int, max_new_tokens=256, n_samples=10):
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)
    input_ids_batch = input_ids.repeat(n_samples, 1)
    

    all_outputs = []

    ablation_hook = partial(
        ablation,
        ablation_feature=ablation_feature,
    )
    
    #for i in range(n_samples):

    # standard transformerlens syntax for a hook context for generation
    with model.hooks(fwd_hooks=[('blocks.25.hook_resid_pre.hook_sae_acts_pre', ablation_hook)]):
        output = model.generate(
            input_ids_batch, 
            top_k=9, #tbd
            max_new_tokens=max_new_tokens,
            eos_token_id=1,
            do_sample=True,
            verbose=False,
            ) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.
    
    all_outputs = model.tokenizer.batch_decode(output)
    all_outputs = [o.replace("<|endoftext|>", "") for o in all_outputs]


    return all_outputs



cfg = SAEConfig(
    architecture="jumprelu",
    d_in=1280,
    d_sae=12*1280,
    activation_fn_str="relu",
    apply_b_dec_to_input=True,
    finetuning_scaling_factor=False,
    context_size=256,
    model_name="ZymCTRL",
    hook_name="blocks.25.hook_resid_pre",
    hook_layer=25,
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
    parser = ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    args = parser.parse_args()
    cfg_path = args.cfg_path
    config = load_config(cfg_path)

    for model_iteration, data_iteration in zip(range(30), range(30)):
        if model_iteration == 0 and data_iteration == 0:
            model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"
        else:
            model_path = config["paths"]["model_path"].format(model_iteration)

        sae_path = config["paths"]["sae_path"]
        top_features_path = config["paths"]["top_features_path"].format(model_iteration, data_iteration)
        df_path = config["paths"]["df_path"].format(data_iteration)
        FEATURE_SELECTION_METHOD = config["FEATURE_SELECTION_METHOD"]
        out_dir = config["paths"]["out_dir"].format(model_iteration, data_iteration)









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

        with open(top_features_path, "rb") as f:
            important_features = pkl.load(f)
        feature_indices = important_features["unique_coefs"]
            

        prompt = "3.2.1.1<sep><start>"


        os.makedirs(out_dir, exist_ok=True)

        for ablation_feature in tqdm(feature_indices):
            out = generate_with_ablation(model, sae, prompt, ablation_feature, max_new_tokens=1024, n_samples=10)
            with open(f"{out_dir}/ablation_feature_{ablation_feature}.txt", "w") as f:
                for i, o in enumerate(out):
                    f.write(f">3.2.1.1_{i},"+o+"\n")
    del model, sae 
torch.cuda.empty_cache()
