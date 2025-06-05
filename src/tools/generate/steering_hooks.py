import torch
from transformer_lens.hook_points import HookPoint


def feature_ablation_with_activation_caching(
    activations: torch.Tensor, 
    hook: HookPoint, 
    ablation_feature: list[int], 
    generation_idx: str = None,
    mask_activations: dict = None
) -> torch.Tensor:
    """
    Feature ablation hook that sets specified features to zero and tracks activation states.
    
    Args:
        activations: The activations from the model
        hook: The hook point from which the activations are obtained
        ablation_feature: List of feature indices to ablate
        generation_idx: Generation step identifier for tracking
        mask_activations: Dictionary to store activation tracking data
    
    Returns:
        Modified activations with specified features set to zero
    """
    # Skip intervention during prompt processing (multiple tokens)
    if activations.shape[1] > 1:
        return activations
        
    # Track which features would have been activated before ablation
    if generation_idx is not None and mask_activations is not None:
        pre_ablation_mask = activations[:, :, ablation_feature] > 0
        mask_cpu = pre_ablation_mask.cpu().detach().numpy()
        
        for i in range(mask_cpu.shape[0]):
            sample_key = f"sample_{i}"
            if sample_key not in mask_activations[generation_idx]:
                mask_activations[generation_idx][sample_key] = []
            
            # Store activation status
            if len(ablation_feature) > 1:
                mask_activations[generation_idx][sample_key].append(mask_cpu[i, 0, :].tolist())
            else:
                mask_activations[generation_idx][sample_key].append(mask_cpu[i, 0].tolist())
    
    # Perform ablation
    activations[:, :, ablation_feature] = 0
    return activations


def clampping(
    activations: torch.Tensor, 
    hook: HookPoint, 
    clampping_features: list[int], 
    clampping_value: list[float], 
    generation_idx: str = None,
    mask_activations: dict = None
) -> torch.Tensor:
    """
    Feature clampping hook that sets active features to specified values and tracks states.
    
    Args:
        activations: The activations from the model
        hook: The hook point from which the activations are obtained
        clampping_features: List of feature indices to clamp
        clampping_value: List of values to clamp each feature to (must match length of clampping_features)
        generation_idx: Generation step identifier for tracking
        mask_activations: Dictionary to store activation tracking data
    
    Returns:
        Modified activations with specified features clampped
    """
    # Skip intervention during prompt processing (multiple tokens)
    if activations.shape[1] > 1:
        return activations
    
    # Ensure clampping_value is a list with same length as clampping_features
    if isinstance(clampping_value, (int, float)):
        clampping_value = [clampping_value] * len(clampping_features)
    elif len(clampping_value) != len(clampping_features):
        raise ValueError(f"Length of clampping_value ({len(clampping_value)}) must match clampping_features ({len(clampping_features)})")
        
    # Identify which features are active (> 0) and apply clampping per feature
    for i, (feature_idx, clamp_val) in enumerate(zip(clampping_features, clampping_value)):
        feature_mask = activations[:, :, feature_idx] > 0
        
        # Track activation states if requested
        if generation_idx is not None and mask_activations is not None:
            mask_cpu = feature_mask.cpu().detach().numpy()
            
            for batch_idx in range(mask_cpu.shape[0]):
                sample_key = f"sample_{batch_idx}"
                if sample_key not in mask_activations[generation_idx]:
                    mask_activations[generation_idx][sample_key] = []
                
                # For first feature, initialize the list for this position
                if i == 0:
                    mask_activations[generation_idx][sample_key].append([])
                
                # Add this feature's activation state to the current position
                mask_activations[generation_idx][sample_key][-1].append(bool(mask_cpu[batch_idx, 0]))
        
        # Apply clampping to this specific feature
        activations[:, :, feature_idx] = torch.where(
            feature_mask, clamp_val, activations[:, :, feature_idx]
        )
    
    return activations


def dense_steering_hook(
    activations: torch.Tensor, 
    hook: HookPoint, 
    steering_vector: torch.Tensor, 
    strength: float
) -> torch.Tensor:
    """
    Dense steering hook that adds a steering vector to activations during generation.
    
    Args:
        activations: The activations from the model
        hook: The hook point from which the activations are obtained
        steering_vector: The vector to steer the activations with
        strength: The strength of the steering
    
    Returns:
        Modified activations with steering vector applied
    """
    # Only apply during generation (single token)
    if activations.shape[1] == 1:
        activations = activations + steering_vector * strength
    
    return activations



