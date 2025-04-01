#from SAELens.sae_lens import HookedSAETransformer, SAE, SAEConfig
import random
from sae_lens import HookedSAETransformer, SAE, SAEConfig
from src.utils import load_model, get_sl_model, load_sae
from functools import partial
import torch
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pickle as pkl
from peft import LoraConfig, inject_adapter_in_model
import pandas as pd

def get_model_sae(model_iteration, data_iteration):

    if model_iteration == 0:
        model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"
    else:
        model_path = f"/home/woody/b114cb/b114cb23/Filippo/Q4_2024/DPO/DPO_Clean/DPO_clean_alphamylase/output_iteration{model_iteration}/" 
    sae_path = f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/Diffing Alpha Amylase New/M{model_iteration}_D{data_iteration}/diffing/"
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

    prompt = "3.2.1.1<sep><start>"

    return model, tokenizer, sae



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

def load_esm_model(checkpoint, num_labels, half_precision, full=False, deepspeed=True):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels,
        torch_dtype=torch.float16 if half_precision and deepspeed else None
    )
    if full:
        return model, tokenizer

    peft_config = LoraConfig(
        r=4, lora_alpha=1, bias="all", target_modules=["query", "key", "value", "dense"]
    )
    model = inject_adapter_in_model(peft_config, model)
    for param_name, param in model.classifier.named_parameters():
        param.requires_grad = True
    return model, tokenizer




def load_oracle_model(checkpoint, filepath, num_labels=1, mixed=False, full=False, deepspeed=True):
    model, tokenizer = (
        load_esm_model(checkpoint, num_labels, mixed, full, deepspeed)
        if "esm" in checkpoint
        else load_T5_model(checkpoint, num_labels, mixed, full, deepspeed)
    )
    non_frozen_params = torch.load(filepath)
    for param_name, param in model.named_parameters():
        if param_name in non_frozen_params:
            param.data = non_frozen_params[param_name].data
    return tokenizer, model






def ablation(activations, hook, ablation_feature):
    # Check prompt processing
    if activations.shape[1] > 1:
        #activations[:,:,ablation_feature] = 0
        pass
        
    else:
        activations[:,:,ablation_feature] = 0
    
    return activations

def pass_hook(activations, hook):
    return activations


def generate_with_ablation(model: HookedSAETransformer, sae: SAE, prompt: str, ablation_feature: int, max_new_tokens=256, n_samples=10):
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)
    input_ids_batch = input_ids.repeat(n_samples, 1)

    if ablation_feature == -1:
        ablation_hook = pass_hook
    else:
        ablation_hook = partial(
            ablation,
            ablation_feature=ablation_feature,
        )
    
    with model.hooks(fwd_hooks=[('blocks.26.hook_resid_pre.hook_sae_acts_post', ablation_hook)]):
        output = model.generate(
            input_ids_batch, 
            top_k=9, #tbd
            max_new_tokens=max_new_tokens,
            eos_token_id=1,
            do_sample=True,
            verbose=False,
            ) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.
    return output






if __name__ == "__main__":
    if True:
        model_iteration = 2
        data_iteration = 2
        csv_path = f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/ablation/ablation_data_M{model_iteration}_D{data_iteration}.csv"
        df = pd.read_csv(csv_path, header=[0, 1,2], index_col=0)
        df.columns.names = ["type", "feature", "logits"]
        df = df.apply(pd.to_numeric)
        aggregated_df = df.groupby(level=["type", "feature"], axis=1).mean()
        print(aggregated_df)
    else:
        model_iteration = 2
        data_iteration = 2
        ec_label = "3.2.1.1"
        ablation_feature = 714
        prompt = "3.2.1.1<sep><start>"


        path = f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/important_features/important_features_M{model_iteration}_D{data_iteration}.pkl"
        with open(path, "rb") as f:
            important_features = pkl.load(f)
        positive_feature_indices = important_features["unique_coefs"]
        set_of_pos_feats = set(positive_feature_indices.numpy().tolist())
        set_of_neg_feats = set(torch.arange(10240).numpy().tolist()) - set_of_pos_feats
        list_of_neg_feats = list(set_of_neg_feats)
        sample_list_of_neg_feats = random.sample(list_of_neg_feats, len(set_of_pos_feats))


        checkpoint = "/home/woody/b114cb/b114cb23/models/esm2_t33_650M_UR50D"
        oracle_tokenizer, oracle_model = load_oracle_model(
            checkpoint,
            "/home/woody/b114cb/b114cb23/Filippo/alpha_amylase_activity_predictor/LoRa_esm2_3B/esm_GB1_finetuned.pth",
            num_labels=1
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        oracle_model.to(device)
        oracle_model.eval()


        model, tokenizer, sae = get_model_sae(model_iteration, data_iteration)


        experiment_data = {}

        for type,ablation_features in [("pos",list(set_of_pos_feats)[:20]), ("neg",sample_list_of_neg_feats[:20]), ("neutral",[-1])]:

            all_logits = {}
            
            for ablation_feature in tqdm(ablation_features):
                print(f"Ablating feature {ablation_feature}")
                out = generate_with_ablation(model, sae, prompt, ablation_feature, max_new_tokens=1024, n_samples=10)
                seqs_out = tokenizer.batch_decode(out, skip_special_tokens=True)
                seqs_out = [s.replace("<|endoftext|>", "") for s in seqs_out]
                tokenized_out = oracle_tokenizer(seqs_out, return_tensors="pt", padding=True)
                tokenized_out = tokenized_out.to(device)

                with torch.no_grad():
                    logits = oracle_model(tokenized_out["input_ids"], attention_mask=tokenized_out["attention_mask"]).logits
                all_logits[str(ablation_feature)] = logits.detach().cpu().numpy().reshape(-1)

                torch.cuda.empty_cache()
            
            experiment_data[type] = all_logits
        

        # Create lists of values and types for each logit
        print(experiment_data)
        print(experiment_data.keys())

    # Example nested dictionary: each type has its own set of features,
    # and each feature's logits array may have a different number of samples.
    # Create a list to hold DataFrames per type.
        df_list = []

        for type_key, feature_dict in experiment_data.items():
            # Dictionary to accumulate series for the current type.
            df_dict = {}
            for feature_key, logits_arr in feature_dict.items():
            # Create a Series from the logits array.
                series = pd.Series(logits_arr)
                # The column label will be a tuple (type, feature, "logits")
                df_dict[(type_key, feature_key, "logits")] = series
            # Build a DataFrame for this type.
            df_type = pd.DataFrame(df_dict)
            df_list.append(df_type)

        # Combine all DataFrames along columns.
        # Outer join on the index: the final DataFrame's rows correspond to the union of indices.
        final_df = pd.concat(df_list, axis=1, sort=False)
        final_df.columns = pd.MultiIndex.from_tuples(
            final_df.columns, names=["type", "feature", "quantity"]
        )

        final_df.to_csv(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/ablation/ablation_data_M{model_iteration}_D{data_iteration}.csv")