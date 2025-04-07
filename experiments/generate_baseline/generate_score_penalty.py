
import os
import torch
from transformer_lens import HookedTransformerKeyValueCache
from transformer_lens.utilities import devices
from transformer_lens import HookedTransformerConfig
from typing import Optional, Union, Literal
from transformer_lens.utils import sample_logits
from transformer_lens import utils
import transformer_lens.utils as utils
from sae_lens import HookedSAETransformer
from src.utils import load_model 
from functools import partial
from transformer_lens import HookedTransformerConfig
from typing import Optional, Union, Literal
from transformer_lens.utils import sample_logits
from src.utils import convert_GPT_weights

from sae_lens import HookedSAETransformer, SAE, SAEConfig
from src.utils import load_model, get_sl_model, load_sae
# %%
from transformer_lens import HookedTransformerConfig
from typing import Optional, Union, Literal
from transformer_lens.utils import sample_logits
import tqdm
from src.utils import convert_GPT_weights

from sae_lens import HookedSAETransformer, SAE, SAEConfig
from src.utils import load_model, get_sl_model, load_sae
import os
# %%

from transformers import AutoTokenizer, AutoModelForSequenceClassification
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


USE_DEFAULT_VALUE = False



class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float):
        if not (isinstance(penalty, float) and penalty > 0):
            raise ValueError(f"`penalty` must be a positive float, got {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Create a set of indices for each sequence in the batch
        batch_size = input_ids.shape[0]
        
        # Make a copy of the scores to modify
        scores_modified = scores.clone()
        
        # Apply penalty to each sequence in the batch
        for batch_idx in range(batch_size):
            # Get unique tokens in this sequence
            seq_tokens = input_ids[batch_idx]
            unique_tokens = torch.unique(seq_tokens)
            
            # Get scores for these tokens
            token_scores = scores_modified[batch_idx, unique_tokens]
            
            # Apply penalty based on sign of scores
            penalized_scores = torch.where(
                token_scores < 0,
                token_scores * self.penalty,
                token_scores / self.penalty
            )
            
            # Update scores
            scores_modified[batch_idx, unique_tokens] = penalized_scores
            
        return scores_modified

# %%
        

def sample_logits(
    final_logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
    freq_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample from the logits, in order to generate text

    final_logits has shape [batch, vocab_size]
    We divide the logits by temperature before softmaxing and sampling - high temperature = more uniform, low = more argmaxy. Temp = 0.0 is greedy sampling
    We apply top_k and top_p filtering to the logits, to encourage diversity. top_k = 10 means we only sample from the 10 most likely tokens. top_p = 0.9 means we only sample from the top 90% of tokens, and then renormalise the distribution. top_k and top_p are mutually exclusive. By default we apply neither and just sample from the full distribution.

    Frequency penalty is a penalty on the probability of a token, proportional to the number of times it has been generated so far. This encourages the model to generate new tokens, rather than repeating itself. It is a hyperparameter, and should be tuned. It is applied to the logits before sampling. If this is non-zero it is required to input the input_tokens

    Repetition penalty applies a multiplicative penalty to tokens that have already appeared in the sequence, making them less likely to be repeated. Values > 1.0 discourage repetition, while values < 1.0 encourage it.

    #! TODO: Finish testing all the edge cases here. Useful testing code:
    logits = torch.randn(4)
    print(logits)
    np.unique(np.array([sample_logits(logits, top_k=2).item() for i in range(1000)]), return_counts=True)
    """







    if temperature == 0.0:
        # Greedy sampling
        return final_logits.argmax(dim=-1)
    else:
        # Sample from the distribution
        final_logits = final_logits.clone()
        
        # Apply repetition penalty if needed
        if repetition_penalty != 1.0 and tokens is not None:
            batch_size = final_logits.shape[0]
            for batch_idx in range(batch_size):
                # Get unique tokens in this sequence
                seq_tokens = tokens[batch_idx]
                unique_tokens = torch.unique(seq_tokens)
                scores = final_logits[batch_idx].unsqueeze(0)
                
                # Get scores for these tokens
                score = torch.gather(scores, 1, input_ids)

                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)

                scores_processed = scores.scatter(1, input_ids, score)
                token_scores = final_logits[batch_idx, unique_tokens]
                
                # Apply penalty based on sign of scores
                penalized_scores = torch.where(
                    token_scores < 0,
                    token_scores * repetition_penalty,
                    token_scores / repetition_penalty
                )
                
                # Update scores
                final_logits[batch_idx, unique_tokens] = penalized_scores

        final_logits = final_logits / temperature
        
        if freq_penalty > 0:
            assert tokens is not None, "Must provide input_tokens if applying a frequency penalty"
            for batch_index in range(final_logits.shape[0]):
                # torch.bincount returns a tensor of length d_vocab, with the number of occurences of each token in the tokens.
                final_logits[batch_index] = final_logits[
                    batch_index
                ] - freq_penalty * torch.bincount(
                    tokens[batch_index], minlength=final_logits.shape[-1]
                )
        if top_k is not None:
            assert top_k > 0, "top_k has to be greater than 0"
            top_logits, top_idx = final_logits.topk(top_k, dim=-1)
            indices_to_remove = final_logits < top_logits[..., -1].unsqueeze(-1)
            final_logits = final_logits.masked_fill(indices_to_remove, -float("inf"))
        elif top_p is not None:
            assert 1.0 >= top_p > 0.0, "top_p has to be in (0, 1]"
            sorted_logits, sorted_indices = torch.sort(final_logits, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # We round up - we want prob >= top_p not <top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            final_logits = final_logits.masked_fill(indices_to_remove, -float("inf"))

        final_logits = final_logits.to(torch.float32)
        return torch.distributions.categorical.Categorical(logits=final_logits).sample()

# %%

# ====================

class MyModel(HookedSAETransformer):
    def __init__(self, cfg=None, tokenizer=None, **kwargs):
        # Pass all arguments to parent class
        super().__init__(cfg, tokenizer, **kwargs)

    def get_pos_offset(self, past_kv_cache, batch_size):
        # If we're doing caching, then we reuse keys and values from previous runs, as that's the
        # only way that past activations will affect the final logits. The cache contains those so
        # we don't need to recompute them. This is useful for generating text. As we have absolute
        # positional encodings, to implement this we have a `pos_offset` variable, defaulting to
        # zero, which says to offset which positional encodings are used (cached keys and values
        # were calculated with their own positional encodings).
        if past_kv_cache is None:
            pos_offset = 0
        else:
            (
                cached_batch_size,
                cache_ctx_length,
                num_heads_in_cache,
                d_head_in_cache,
            ) = past_kv_cache[0].past_keys.shape
            assert cached_batch_size == batch_size
            if self.cfg.n_key_value_heads is None:
                assert num_heads_in_cache == self.cfg.n_heads
            else:
                assert num_heads_in_cache == self.cfg.n_key_value_heads
            assert d_head_in_cache == self.cfg.d_head
            pos_offset = cache_ctx_length
        return pos_offset

    def get_residual(
        self,
        embed,
        pos_offset,
        prepend_bos=USE_DEFAULT_VALUE,
        attention_mask=None,
        tokens=None,
        return_shortformer_pos_embed=True,
        device=None,
    ):
        if device is None:
            device = devices.get_device_for_block_index(0, self.cfg)

        if tokens is None:
            # Because tokens only need for defining batch size and sequence length, we can simply synthesize them
            tokens = torch.ones((embed.size(0), embed.size(1))).int().to(device)

        if self.cfg.positional_embedding_type == "standard":
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset, attention_mask)
            )  # [batch, pos, d_model]
            residual = embed + pos_embed  # [batch, pos, d_model]
            shortformer_pos_embed = None
        elif self.cfg.positional_embedding_type == "shortformer":
            # If we're using shortformer style attention, we don't add the positional embedding to
            # the residual stream. See HookedTransformerConfig for details
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset, attention_mask)
            )  # [batch, pos, d_model]
            residual = embed
            shortformer_pos_embed = pos_embed
        elif self.cfg.positional_embedding_type == "rotary":
            # Rotary doesn't use positional embeddings, instead they're applied when dot producting
            # keys and queries. See HookedTransformerConfig for details
            residual = embed
            shortformer_pos_embed = None
        elif self.cfg.positional_embedding_type == "alibi":
            # ALiBi does not add positional embeddings to word embeddings,instead it biases QK attention scores.
            residual = embed
            shortformer_pos_embed = None
        else:
            raise ValueError(
                f"Invalid positional_embedding_type passed in {self.cfg.positional_embedding_type}"
            )

        if return_shortformer_pos_embed:
            return residual, shortformer_pos_embed
        else:
            return residual



    @torch.inference_mode()
    def generate_with_penalty(
        self,
        input: Union[str, list[str], torch.Tensor] = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = 1,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        freq_penalty: float = 0.0,
        use_past_kv_cache: bool = True,
        prepend_bos: Optional[bool] = False,
        padding_side: Optional[Literal["left", "right"]] = "left",
        return_type: Optional[str] = "input",
        verbose: bool = True,
    ) -> Union[
            str,
            list[str],
            torch.Tensor,
               ]:
        """Generate text with repetition penalty applied."""
        with utils.LocallyOverridenDefaults(
                self, prepend_bos=prepend_bos, padding_side=padding_side):

            assert isinstance(input, (str, torch.Tensor, list)) and (
                isinstance(input, list)
                and all(isinstance(i, str) for i in input)
                or not isinstance(input, list)
            ), "Input must be either string, torch.Tensor, or list[str]"

            assert return_type in [
                "input",
                "str",
                "tokens",
                "embeds",
            ], "return_type must be one of ['input', 'str', 'tokens', 'embeds']"

            if return_type == "input":
                if isinstance(input, (str, list)):
                    return_type = "str"
                elif input.ndim == 2:
                    return_type = "tokens"
                else:
                    return_type = "embeds"

            if isinstance(input, (str, list)):
                input_type = "str"
                # If text, convert to tokens (batch_size=1)
                assert (
                    self.tokenizer is not None
                ), "Must provide a tokenizer if passing a string to the model"
                input = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
            elif input.ndim == 2:
                input_type = "tokens"
            else:
                input_type = "embeds"


            input_tokens = input if input_type in ["str", "tokens"] else None
            batch_size, ctx_length = input.shape[0], input.shape[1]
            device = devices.get_device_for_block_index(0, self.cfg)
            input = input.to(device)
            if use_past_kv_cache:
                past_kv_cache = HookedTransformerKeyValueCache.init_cache(
                    self.cfg, self.cfg.device, batch_size
                )
            else:
                past_kv_cache = None
            
            shortformer_pos_embed = None
            embeds = input if input_type == "embeds" else self.embed(input)

            assert isinstance(embeds, torch.Tensor) and embeds.ndim == 3

            stop_tokens: list[int] = []
            eos_token_for_padding = 0
            assert self.tokenizer is not None
            # Initialize the output with the input tokens
            output_tokens = input.clone()
            batch_size = output_tokens.shape[0]


            if stop_at_eos:
            
                tokenizer_has_eos_token = (
                    self.tokenizer is not None and self.tokenizer.eos_token_id is not None
                )
                if eos_token_id is None:
                    assert (
                        tokenizer_has_eos_token
                    ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

                    eos_token_id = self.tokenizer.eos_token_id

                if isinstance(eos_token_id, int):
                    stop_tokens = [eos_token_id]
                    eos_token_for_padding = eos_token_id
                else:
                    # eos_token_id is a Sequence (e.g. list or tuple)
                    stop_tokens = eos_token_id
                    eos_token_for_padding = (
                        self.tokenizer.eos_token_id if tokenizer_has_eos_token else eos_token_id[0]
                    )

                # An array to track which sequences in the batch have finished.
                finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.cfg.device)

                # Currently nothing in HookedTransformer changes with eval, but this is here in case
                    
                self.eval()
                sampled_tokens_list = []
                for index in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
                    pos_offset = self.get_pos_offset(past_kv_cache, batch_size)

                    tokens = torch.zeros((embeds.size(0), embeds.size(1))).to(torch.int)
                    attention_mask = utils.get_attention_mask(
                        self.tokenizer, tokens, False if prepend_bos is None else prepend_bos
                    ).to(device)
                    residual, shortformer_pos_embed = self.get_residual(
                        embeds,
                        pos_offset,
                        return_shortformer_pos_embed=True,
                        device=device,
                        attention_mask=attention_mask,
                    )

                    # While generating, we keep generating logits, throw away all but the final logits,
                    # and then use those logits to sample from the distribution We keep adding the
                    # sampled tokens to the end of tokens.
                    start_at_layer = 0  # Make forward returns embeddings
                    if use_past_kv_cache:
                        # We just take the final tokens, as a [batch, 1] tensor
                        if index > 0:
                            logits = self.forward(
                                residual[:, -1:],
                                return_type="logits",
                                prepend_bos=prepend_bos,
                                padding_side=padding_side,
                                past_kv_cache=past_kv_cache,
                                start_at_layer=start_at_layer,
                                shortformer_pos_embed=shortformer_pos_embed,
                            )
                        else:
                            logits = self.forward(
                                residual,
                                return_type="logits",
                                prepend_bos=prepend_bos,
                                padding_side=padding_side,
                                past_kv_cache=past_kv_cache,
                                start_at_layer=start_at_layer,
                                shortformer_pos_embed=shortformer_pos_embed,
                            )
                    else:
                        # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
                        # the cache.
                        logits = self.forward(
                            residual,
                            return_type="logits",
                            prepend_bos=prepend_bos,
                            padding_side=padding_side,
                            start_at_layer=start_at_layer,
                            shortformer_pos_embed=shortformer_pos_embed,
                        )
                    final_logits = logits[:, -1, :]

                    if do_sample:
                        if input_type in [
                            "str",
                            "tokens",
                        ]:  # Those types of inputs support frequency penalty
                            sampled_tokens = sample_logits(
                                final_logits,
                                top_k=top_k,
                                top_p=top_p,
                                temperature=temperature,
                                freq_penalty=freq_penalty,
                                repetition_penalty=repetition_penalty,
                                tokens=torch.cat(
                                    (input_tokens, torch.cat(sampled_tokens_list, dim=1)), dim=1) if "sampled_tokens" in locals() else input_tokens,
                            ).to(devices.get_device_for_block_index(0, self.cfg))
                        else:
                            sampled_tokens = utils.sample_logits(
                                final_logits, top_k=top_k, top_p=top_p, temperature=temperature
                            ).to(devices.get_device_for_block_index(0, self.cfg))
                    else:
                        sampled_tokens = final_logits.argmax(-1).to(
                            devices.get_device_for_block_index(0, self.cfg)
                        )
                    sampled_tokens_list.append(sampled_tokens.unsqueeze(1))
                    if stop_at_eos:
                        # For all unfinished sequences, add on the next token. If a sequence was
                        # finished, throw away the generated token and add eos_token_for_padding
                        # instead.
                        sampled_tokens[finished_sequences] = eos_token_for_padding
                        finished_sequences.logical_or_(
                            torch.isin(
                                sampled_tokens.to(self.cfg.device),
                                torch.tensor(stop_tokens).to(self.cfg.device),
                            )
                        )

                    embeds = torch.hstack([embeds, self.embed(sampled_tokens.unsqueeze(-1))])

                    if stop_at_eos and finished_sequences.all():
                        break

                sampled_tokens = torch.cat(sampled_tokens_list, dim=1)
                if input_type in ["str", "tokens"]:
                    output_tokens = torch.cat((input_tokens, sampled_tokens), dim=1)
                else:
                    output_tokens = sampled_tokens

                if return_type == "str":
                    decoded_texts = [
                        self.tokenizer.decode(tokens, skip_special_tokens=True)
                        for tokens in output_tokens
                    ]
                    return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts
                elif return_type == "tokens":
                    return output_tokens
                else:
                    return embeds











if __name__ == "__main__":


    # Load the dataframe
    model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"


    tokenizer, model = load_model(model_path)
    cfg = model.config
    state_dict = convert_GPT_weights(model, cfg)

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



    model = MyModel(cfg_ht,tokenizer=tokenizer)
    model.load_and_process_state_dict(
        state_dict,
        fold_ln = False,
        center_writing_weights = False,
        center_unembed=False,
        fold_value_biases = False,
        refactor_factored_attn_matrices = False,
    )
    model.to("cuda")

    prompt = "3.2.1.1<sep><start>"


    n_samples = 25
    input_ids = model.to_tokens(prompt, prepend_bos=False)
    input_ids_batch = input_ids.repeat(n_samples, 1)
    

    output = model.generate_with_penalty(
                                        input_ids_batch,
                                        max_new_tokens=800,
                                        repetition_penalty=1.2,
                                        eos_token_id=1,
                                        top_k=9,
                                        do_sample=True)
        
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
    
    seqs_out = tokenizer.batch_decode(output, skip_special_tokens=True)
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
# %%
