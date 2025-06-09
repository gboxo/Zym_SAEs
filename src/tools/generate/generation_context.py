from typing import Dict, Any, Optional, List, Callable
import torch
from functools import partial
from transformer_lens import HookedTransformer
from sae_lens import SAE
import pickle
import os


class GenerationContext:
    """Context manager for tracking activations during generation with interventions"""
    
    def __init__(self):
        self.activations: Dict[str, Dict] = {}
        self.current_generation_idx: Optional[str] = None
        
    def reset(self):
        """Reset all activation tracking"""
        self.activations.clear()
        self.current_generation_idx = None
        
    def set_generation_idx(self, generation_idx: str):
        """Set the current generation index for tracking"""
        self.current_generation_idx = generation_idx
        if generation_idx not in self.activations:
            self.activations[generation_idx] = {}
            
    def get_ablation_hook(self, ablation_features: List[int]) -> Callable:
        """Get a partial function for ablation hook with context"""
        from src.tools.generate.steering_hooks import feature_ablation_with_activation_caching
        
        return partial(
            feature_ablation_with_activation_caching,
            ablation_feature=ablation_features,
            generation_idx=self.current_generation_idx,
            mask_activations=self.activations
        )
        
    def get_clampping_hook(self, clampping_features: List[int], clampping_value: List[float]) -> Callable:
        """Get a partial function for clampping hook with context"""
        from src.tools.generate.steering_hooks import clampping
        
        return partial(
            clampping,
            clampping_features=clampping_features,
            clampping_value=clampping_value,
            generation_idx=self.current_generation_idx,
            mask_activations=self.activations
        )
        
    def save_activations(self, output_dir: str):
        """Save all tracked activations to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/activations.pkl", "wb") as f:
            pickle.dump(self.activations, f)
            
        # For backwards compatibility
        with open(f"{output_dir}/ablation_activations.pkl", "wb") as f:
            pickle.dump(self.activations, f)


def generate_with_intervention(
    model: HookedTransformer, 
    sae: SAE, 
    prompt: str, 
    intervention_type: str,
    intervention_params: Dict[str, Any],
    max_new_tokens: int = 256, 
    n_samples: int = 10, 
    generation_idx: str = "default",
    n_batches: int = 3,
    hook_layer: str = 'blocks.25.hook_resid_pre.hook_sae_acts_post'
) -> tuple[List[List[str]], GenerationContext]:
    """
    Unified generation function for any intervention type
    """
    context = GenerationContext()
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)
    input_ids_batch = input_ids.repeat(n_samples, 1)
    
    all_outputs_batches = []
    
    for batch_idx in range(n_batches):
        batch_generation_idx = f"{generation_idx}_{batch_idx}"
        context.set_generation_idx(batch_generation_idx)
        
        # Create hook function based on intervention type
        if intervention_type == "ablation":
            hook_fn = context.get_ablation_hook(intervention_params["features"])
        elif intervention_type == "clampping":
            hook_fn = context.get_clampping_hook(
                intervention_params["features"], 
                intervention_params["value"]
            )
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")
        
        # Generate with hook
        with model.hooks(fwd_hooks=[(hook_layer, hook_fn)]):
            output = model.generate(
                input_ids_batch,
                top_k=9,
                max_new_tokens=max_new_tokens,
                eos_token_id=1,
                do_sample=True,
                verbose=False,
            )
        
        # Decode outputs
        all_outputs = model.tokenizer.batch_decode(output)
        all_outputs = [o.replace("<|endoftext|>", "") for o in all_outputs]
        
        # Process activation records to match sequence lengths
        _process_activation_records(context, batch_generation_idx, output, input_ids, 
                                  model, n_samples)
        
        all_outputs_batches.append(all_outputs)
    
    return all_outputs_batches, context


def _process_activation_records(context: GenerationContext, batch_generation_idx: str, 
                               output: torch.Tensor, input_ids: torch.Tensor, 
                               model: HookedTransformer, n_samples: int):
    """Helper function to process activation records to match sequence lengths"""
    prompt_length = input_ids.shape[1]
    
    for i in range(n_samples):
        sample_key = f"sample_{i}"
        if sample_key in context.activations[batch_generation_idx]:
            seq_length = (output[i] != model.tokenizer.pad_token_id).sum().item()
            generated_length = seq_length - prompt_length
            if generated_length > 0:
                context.activations[batch_generation_idx][sample_key] = \
                    context.activations[batch_generation_idx][sample_key][:generated_length]


def generate_with_ablation(
    model: HookedTransformer, 
    sae: SAE, 
    prompt: str, 
    ablation_features: List[int], 
    max_new_tokens: int = 256, 
    n_samples: int = 10, 
    generation_idx: str = "default",
    n_batches: int = 3
) -> tuple[List[List[str]], GenerationContext]:
    """Generate with feature ablation intervention"""
    return generate_with_intervention(
        model, sae, prompt, 
        intervention_type="ablation",
        intervention_params={"features": ablation_features},
        max_new_tokens=max_new_tokens,
        n_samples=n_samples,
        generation_idx=generation_idx,
        n_batches=n_batches
    )


def generate_with_clampping(
    model: HookedTransformer, 
    sae: SAE, 
    prompt: str, 
    clampping_features: List[int], 
    clampping_value: List[float], 
    max_new_tokens: int = 512, 
    n_samples: int = 10, 
    generation_idx: str = "default",
    n_batches: int = 3
) -> tuple[List[List[str]], GenerationContext]:
    """Generate with feature clampping intervention"""
    return generate_with_intervention(
        model, sae, prompt,
        intervention_type="clampping", 
        intervention_params={"features": clampping_features, "value": clampping_value},
        max_new_tokens=max_new_tokens,
        n_samples=n_samples,
        generation_idx=generation_idx,
        n_batches=n_batches
    )


# Backwards compatibility aliases