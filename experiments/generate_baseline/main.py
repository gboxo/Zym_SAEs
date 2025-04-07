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
    model, tokenizer = load_esm_model(checkpoint, num_labels, mixed, full, deepspeed)
    non_frozen_params = torch.load(filepath)
    for param_name, param in model.named_parameters():
        if param_name in non_frozen_params:
            param.data = non_frozen_params[param_name].data
    return model, tokenizer



def generate_tl(model: HookedSAETransformer, prompt: str, max_new_tokens=256, n_samples=20):
    input_ids = model.to_tokens(prompt, prepend_bos=False)
    input_ids_batch = input_ids.repeat(n_samples, 1)

    outputs = model.generate(
        input_ids_batch, 
        top_k=9, #tbd
        max_new_tokens=max_new_tokens,
        eos_token_id=1,
        do_sample=True,
        )
    return outputs


def generate_t(model,tokenizer, prompt: str, max_new_tokens=256, n_samples=20, penalty=1.2):

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to("cuda")
    outputs = model.generate(
        input_ids, 
        top_k=9, #tbd
        repetition_penalty=penalty,
        max_length=max_new_tokens,
        eos_token_id=1,
        pad_token_id=0,
        do_sample=True,
        num_return_sequences=n_samples) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.
    return outputs




if __name__ == "__main__":

    model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"
    tokenizer, model = load_model(model_path)
    model = model.to("cuda")
    ec_label = "3.2.1.1"
    prompt = "3.2.1.1<sep><start>"


    # ============================================
    # Generate with transformers (penalty)
    # ============================================

    out_t_penalty = generate_t(model, tokenizer, prompt, max_new_tokens=800, n_samples=25, penalty=1.2)
    torch.cuda.empty_cache()
    out_t_no_penalty = generate_t(model, tokenizer, prompt, max_new_tokens=800, n_samples=25, penalty=1)
    torch.cuda.empty_cache()

    
    # ============================================
    # Generate with transformer_lens (penalty)
    # ============================================

    model = get_sl_model(model, model.config, tokenizer).to("cuda")
    out_tl = generate_tl(model, prompt, max_new_tokens=800, n_samples=25)
    torch.cuda.empty_cache()

    
    del model

    
    # ============================================
    # Score 
    # ============================================
    
    checkpoint = "/home/woody/b114cb/b114cb23/models/esm2_t33_650M_UR50D"
    oracle_model, oracle_tokenizer = load_oracle_model(
        checkpoint,
        "/home/woody/b114cb/b114cb23/Filippo/alpha_amylase_activity_predictor/LoRa_esm2_3B/esm_GB1_finetuned.pth",
        num_labels=1
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    oracle_model.to(device)
    oracle_model.eval()

    # ============
    
    seqs_out = tokenizer.batch_decode(out_t_penalty, skip_special_tokens=True)
    seqs_out = [s.replace("<|endoftext|>", "").replace(" ","").replace(".","") for s in seqs_out]
    tokenized_out = oracle_tokenizer(seqs_out, return_tensors="pt", padding=True)
    tokenized_out_t = tokenized_out.to(device)

    with torch.no_grad():
        t_logits_penalty = oracle_model(tokenized_out["input_ids"], attention_mask=tokenized_out["attention_mask"]).logits


    torch.cuda.empty_cache()
    print(sum([len(elem) for elem in seqs_out])/len(seqs_out))
    print(t_logits_penalty.shape)
    print("The mean of the activity of transformer generated sequences is (with penalty): ", t_logits_penalty.mean())
    print("The standard deviation of the activity of transformer generated sequences is (with penalty): ", t_logits_penalty.std())
    # ============
    
    seqs_out = tokenizer.batch_decode(out_t_no_penalty, skip_special_tokens=True)
    seqs_out = [s.replace("<|endoftext|>", "").replace(" ","").replace(".","") for s in seqs_out]
    tokenized_out = oracle_tokenizer(seqs_out, return_tensors="pt", padding=True)
    tokenized_out_t = tokenized_out.to(device)

    with torch.no_grad():
        t_logits_no_penalty = oracle_model(tokenized_out["input_ids"], attention_mask=tokenized_out["attention_mask"]).logits


    torch.cuda.empty_cache()
    print(sum([len(elem) for elem in seqs_out])/len(seqs_out))
    print(t_logits_no_penalty.shape)
    print("The mean of the activity of transformer generated sequences is (without penalty): ", t_logits_no_penalty.mean())
    print("The standard deviation of the activity of transformer generated sequences is (without penalty): ", t_logits_no_penalty.std())
    # ============

    seqs_out = tokenizer.batch_decode(out_tl, skip_special_tokens=True)
    seqs_out = [s.replace("<|endoftext|>", "").replace(" ","").replace(".","") for s in seqs_out]
    tokenized_out = oracle_tokenizer(seqs_out, return_tensors="pt", padding=True)
    tokenized_out = tokenized_out.to(device)

    with torch.no_grad():
        tl_logits = oracle_model(tokenized_out["input_ids"], attention_mask=tokenized_out["attention_mask"]).logits


    torch.cuda.empty_cache()
    print(sum([len(elem) for elem in seqs_out])/len(seqs_out))
    print(tl_logits.shape)
    print("The mean of the activity of transformer_lens generated sequences is: ", tl_logits.mean())
    print("The standard deviation of the activity of transformer_lens generated sequences is: ", tl_logits.std())