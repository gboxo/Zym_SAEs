from sae_lens import HookedSAETransformer, SAE, SAEConfig
from src.utils import get_paths, load_model, get_sl_model
from functools import partial
import torch


def ablation(activations, hook, ablation_feature):
    # Check prompt processing
    if activations.shape[0] > 1:
        return activations
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
        with model.hooks(fwd_hooks=[(sae.cfg.hook_name, ablation_hook)]):
            output = model.generate(
                input_ids, 
                temperature=0.9,
                top_k=9, #tbd
                #repetition_penalty=1.2,
                max_new_tokens=max_new_tokens,
                eos_token_id=1,
                do_sample=True,
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

    paths = get_paths()
    sae_path = paths.sae_path
    state_dict = torch.load(sae_path+"sae.pt")
    model_path = paths.model_path
    thresholds = torch.load(sae_path+"/percentiles/feature_percentile_99.pt")
    thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
    state_dict["threshold"] = thresholds


    sae = SAE(cfg)
    sae.load_state_dict(state_dict)
    sae.use_error_term = True

    tokenizer, model = load_model(model_path)
    model = get_sl_model(model, model.config, tokenizer).to("cuda")
    print(model)

    prompt = "3.2.1.1<sep><start>"
    ablation_feature = 3340
    out = generate_with_ablation(model, sae, prompt, ablation_feature, max_new_tokens=256, n_samples=10)
    print(out)