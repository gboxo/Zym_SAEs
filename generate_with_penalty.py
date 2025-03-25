# %%
import torch
from transformer_lens import HookedTransformer  # Replace with actual import
import transformer_lens.utils as utils
from typing import Optional, Union, Literal
from transformer_lens.HookedTransformer import USE_DEFAULT_VALUE
from transformer_lens.HookedTransformer import devices
from transformer_lens.HookedTransformer import HookedTransformerKeyValueCache
from transformer_lens.utils import sample_logits
import tqdm

# %%

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
                
                # Get scores for these tokens
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

class MyModel(HookedTransformer):
    def __init__(self, cfg=None, tokenizer=None, **kwargs):
        # Pass all arguments to parent class
        super().__init__(cfg, tokenizer, **kwargs)
    
    def generate_with_penalty(
        self,
        input: Union[str, list[str], torch.Tensor] = "",
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        freq_penalty: float = 0.0,
        **kwargs
    ):
        """Generate text with repetition penalty applied."""
        # Process input to get tokens
        if isinstance(input, str):
            tokens = self.to_tokens(input, prepend_bos=True)
        elif isinstance(input, list) and all(isinstance(item, str) for item in input):
            tokens = self.to_tokens(input, prepend_bos=True)
        elif isinstance(input, torch.Tensor):
            tokens = input
        else:
            raise ValueError(f"Input {input} not recognized")
        
        # Initialize the output with the input tokens
        output_tokens = tokens.clone()
        batch_size = output_tokens.shape[0]
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Get logits from the model
            with torch.no_grad():
                logits = self(output_tokens)[:, -1, :]
            
            # Sample from the logits with all penalties applied
            next_token = sample_logits(
                logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                freq_penalty=freq_penalty,
                repetition_penalty=repetition_penalty,
                tokens=output_tokens
            )
            
            # Append the new token to the output
            next_token = next_token.unsqueeze(-1)
            output_tokens = torch.cat([output_tokens, next_token], dim=-1)
        
        return output_tokens


if __name__ == "__main__":
    model = MyModel.from_pretrained("gpt2")  # Initialize model
    # %%
    tokens = model.tokenizer.encode("Hello, how are you?", return_tensors="pt")
    output = model.generate_with_penalty(tokens, max_new_tokens=10, repetition_penalty=1.5, temperature=0.8,do_sample=True)
    
    # Decode and print the output
    print(model.tokenizer.decode(output[0]))

# %%
