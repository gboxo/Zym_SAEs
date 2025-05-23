from torch.nn.functional import log_softmax
from tqdm import tqdm
from sae_lens import SAE, SAEConfig
from src.utils import load_model, get_sl_model, load_sae
from functools import partial
import torch
from experiments_alex.circuit_analysis.tooling import *        
from sae_lens import SAE
from functools import partial
import pandas as pd


cfg = SAEConfig(
    architecture="jumprelu",
    d_in=1280,
    d_sae=12*1280,
    activation_fn_str="relu",
    apply_b_dec_to_input=True,
    finetuning_scaling_factor=False,
    context_size=512,
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

    print("Starting")


    model_path = "/home/woody/b114cb/b114cb23/DPO_amylase_run_SAPI_FT_v2/output_iteration3/"
    sae_path = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/dpo_noelia/M3_D3_rl2/diffing/"
    sequences = "/home/woody/b114cb/b114cb23/boxo/kl_divergence/trans_comb/original_sequences.fasta"

    all_saes_path = {
        10: "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/dpo_noelia/M3_D3_rl2_layer_10/diffing/",
        15: "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/dpo_noelia/M3_D3_rl2_layer_15/diffing/",
        20: "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/dpo_noelia/M3_D3_rl2_layer_20/diffing/",
        25: "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/dpo_noelia/M3_D3_rl2/diffing/",
    }


    saes_dict = {}
    for layer, sae_path in all_saes_path.items():
        hook_name = f"blocks.{layer}.hook_resid_pre"
        cfg_partial = cfg
        cfg_partial.hook_name = hook_name
        cfg_partial.hook_layer = layer


        cfg_sae, sae = load_sae(sae_path)
        thresholds = torch.load(sae_path+"/percentiles/feature_percentile_50.pt")
        thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
        state_dict = sae.state_dict()
        state_dict["threshold"] = thresholds
        del sae

        sae = SAE(cfg)
        sae.load_state_dict(state_dict)
        sae.use_error_term = True
        saes_dict[hook_name] = sae
    






    
    # Load the sequences
    with open(sequences, "r") as f:
        seqs = f.read()
        seqs = seqs.split(">")
        seqs = [seq.split("\n") for seq in seqs if seq != ""]
        seqs = {seq[0]: seq[1] for seq in seqs if seq != ""}
    
    
    pos_seqs = {} 
    for key,val in seqs.items():
        _,pos,_, id = key.split("_")
        if pos in pos_seqs.keys():
            pos_seqs[pos].append(val)
        else:
            pos_seqs[pos] = [val]
    
    




    tokenizer, model = load_model(model_path)
    model = get_sl_model(model, model.config, tokenizer).to("cuda")



        
    prompt = "3.2.1.1<sep><start>"

    positions = [285, 293, 107, 102, 138]

    position_transition_dict = {
        285:("L", "S"),
        293:("L", "I"),
        107:("F", "I"),
        102:("L", "I"),
        138:("H", "S"),
    }

    
    
    # Tokenize the sequences


    """

    Position,Count,Avg KL,Same Token %,Top Transitions
    285,153,5.889991454439225,28.75816993464052,"['449 (S) → 442 (L) (103 times)']
    293,136,2.318561604794334,2.2058823529411766,"['440 (I) → 442 (L) (103 times)' ]
    107,117,2.886726530189188,3.418803418803419,"['440 (I) → 437 (F) (112 times)']
    102,88,0.9261001914062283,23.863636363636363,['440 (I) → 442 (L) (67 times)']
    138,84,1.4521822613619624,9.523809523809524,['449 (S) → 439 (H) (76 times)']

    """    


    position_transitions_tok_id = {
        285:("449", "442"),
        293:("440", "442"),
        107:("440", "437"),
        102:("440", "442"),
        138:("449", "439"),
    }






    def metric_fn(logits: torch.Tensor, position: int) -> torch.Tensor:
        # Get the token IDs for this position from the transition dict
        orig_token_id = int(position_transitions_tok_id[position][0])
        trans_token_id = int(position_transitions_tok_id[position][1])
        
        # Get logprobs
        log_probs = log_softmax(logits, dim=-1)
        
        # Calculate logit difference between original and transition token
        logit_diff = log_probs[..., orig_token_id] - log_probs[..., trans_token_id]
        
        return logit_diff


    number_of_seqs = len(seqs)
    # Create a dictionary to store results for each position
    position_results = {int(pos): {layer: {'attributions': None, 'transition': position_transition_dict[int(pos)], 'token_ids': position_transitions_tok_id[int(pos)]} for layer in saes_dict.keys()} for pos in positions}
    number_of_seqs = {}

    
    
    for pos, seqs in pos_seqs.items():
        pos = int(pos) +8 
    
        for i, seq in tqdm(enumerate(seqs)):
            # Analyze each position
            tokenized_seqs = tokenizer(prompt+seq, return_tensors="pt", padding="max_length", truncation=True, max_length=512)["input_ids"][:,:pos]
            
            # Create the metric function for this specific position
            position_metric_fn = partial(metric_fn, position=pos)
            
            # Run the attribution analysis
            attribution = calculate_feature_attribution(
                model=model,
                input=tokenized_seqs,
                metric_fn=position_metric_fn,
                include_saes=saes_dict,
                include_error_term=True,
                return_logits=True,
            )
            
            # Get the feature attributions for this position
            all_attributions = {}
            for layer, sae in saes_dict.items():
                feature_attributions = attribution.sae_feature_attributions[layer].cpu()
                all_attributions[layer] = feature_attributions
            
            
            # Store the attributions for averaging later
            for layer, sae in saes_dict.items():
                if position_results[pos][layer]['attributions'] is None:
                    position_results[pos][layer]['attributions'] = all_attributions[layer]
                else:
                    position_results[pos][layer]['attributions'] += all_attributions[layer]
    
    

import pickle
with open("/home/woody/b114cb/b114cb23/boxo/dpo_noelia/attributions.pkl", "wb") as f:
    pickle.dump(position_results, f)
    