from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
from transformers import AutoConfig

import einops
import torch
import seaborn as sns
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens import HookedTransformer



def convert_GPT_weights(gpt, cfg: GPT2Config) -> dict:

    state_dict = {}

    state_dict["embed.W_E"] = gpt.transformer.wte.weight
    state_dict["pos_embed.W_pos"] = gpt.transformer.wpe.weight



    for l in range(cfg.n_layer):
        state_dict[f"blocks.{l}.ln1.w"] = gpt.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = gpt.transformer.h[l].ln_1.bias

        c_attn = gpt.transformer.h[l].attn.c_attn.weight.T
        W_Q = c_attn[:cfg.n_embd, :]
        W_K = c_attn[cfg.n_embd:2*cfg.n_embd, :]
        W_V = c_attn[2*cfg.n_embd:, :]




        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_head)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_head)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_head)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        attn_bias = gpt.transformer.h[l].attn.c_attn.bias
        b_Q = attn_bias[:cfg.n_embd]
        b_K = attn_bias[cfg.n_embd:2*cfg.n_embd]
        b_V = attn_bias[2*cfg.n_embd:]

        b_Q = einops.rearrange(
            b_Q,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_head,
        )

        b_K = einops.rearrange(
            b_K,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_head,
        )

        b_V = einops.rearrange(
            b_V,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_head,
        )

        state_dict[f"blocks.{l}.attn.b_Q"] = b_Q
        state_dict[f"blocks.{l}.attn.b_K"] = b_K
        state_dict[f"blocks.{l}.attn.b_V"] = b_V

        W_O = gpt.transformer.h[l].attn.c_proj.weight.T

        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_head)
        b_O = gpt.transformer.h[l].attn.c_proj.bias 
        b_O = einops.rearrange(b_O, "(n h)->n h ", n=cfg.n_head)
        b_O = einops.rearrange(
            b_O,
            "n_head d_head -> (n_head d_head)",
            n_head=cfg.n_head,
        )

        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = b_O
        state_dict[f"blocks.{l}.ln2.w"] = gpt.transformer.h[l].ln_2.weight
        state_dict[f"blocks.{l}.ln2.b"] = gpt.transformer.h[l].ln_2.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = gpt.transformer.h[l].mlp.c_fc.weight
        state_dict[f"blocks.{l}.mlp.b_in"] = gpt.transformer.h[l].mlp.c_fc.bias

        state_dict[f"blocks.{l}.mlp.W_out"] = gpt.transformer.h[l].mlp.c_proj.weight
        state_dict[f"blocks.{l}.mlp.b_out"] = gpt.transformer.h[l].mlp.c_proj.bias


    state_dict["ln_final.w"] = gpt.transformer.ln_f.weight
    state_dict["ln_final.b"] = gpt.transformer.ln_f.bias

    state_dict["unembed.W_U"] = gpt.lm_head.weight.T

    return state_dict


def get_ht_model(gpt:AutoModelForCausalLM,cfg: GPT2Config, tokenizer=None) -> HookedTransformer:
    state_dict = convert_GPT_weights(gpt, cfg)

    cfg_dict = {
        "eps": cfg.layer_norm_epsilon,
        "attn_only": False,
        "act_fn": "gelu_new",
        "d_model": cfg.n_embd,
        "d_head": cfg.n_embd // cfg.n_head,
        "n_heads": cfg.n_head,
        "n_layers": cfg.n_layer,
        "n_ctx": cfg.n_ctx,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
        "d_vocab": cfg.vocab_size,
        "use_attn_scale": True,
        "normalization_type": "LN",
        "positional_embedding_type": "standard",
        "tokenizer_prepends_bos": True,
        "default_prepend_bos": False,
        "use_normalization_before_and_after": False,
        "attention_dir": "causal"


    }



    cfg_ht = HookedTransformerConfig(**cfg_dict)



    model = HookedTransformer(cfg_ht,tokenizer=tokenizer)
    model.load_and_process_state_dict(
        state_dict,
        fold_ln = False,
        center_writing_weights = False,
        center_unembed=False,
        fold_value_biases = False,
        refactor_factored_attn_matrices = False,
    )
    return model








if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
    config_ht = AutoConfig.from_pretrained("nferruz/ProtGPT2")
    model_ht = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2",
                                                    attn_implementation="eager")

    cfg = model_ht.config
    cfg.d_mlp = 5120
    model_tl = get_ht_model(model_ht,cfg, tokenizer=tokenizer)



    sequence = "<|endoftext|>\nMGEAMGLTQPAVSRAVARLEERVGIRIFNRTARAITLTDEGRRFYEAVAPLLAGIEMHGY\nRVNVEGVAQLLELYARDILAEGRLVQLLPEWAD\n<|endoftext|>"

    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0) 

    logits_ht = model_ht.forward(input_ids)["logits"]
    logits_tl, cache = model_tl.run_with_cache(input_ids)

    assert torch.allclose(logits_tl,logits_ht,atol=1e-5), "Logits not equal"


