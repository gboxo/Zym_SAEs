import yaml
import os
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def merge_configs(base_config, override_config):
    for key, value in override_config.items():
        if isinstance(value, dict) and key in base_config:
            base_config[key] = merge_configs(base_config[key], value)
        else:
            base_config[key] = value
    return base_config

def load_experiment_config(experiment_config_path):
    # Load the experiment-specific configuration
    experiment_config = load_yaml(experiment_config_path)
    folder_path = os.path.dirname(experiment_config_path)
    
    # Load the base configuration specified in the experiment config
    base_config_path = experiment_config.pop('base_config')
    base_config = load_yaml(os.path.join(folder_path, base_config_path))
    
    # Merge the base and experiment-specific configurations
    complete_config = merge_configs(base_config, experiment_config)
    
    return complete_config


def convert_to_sae_config(config):
    """
    Convert a configuration loaded from YAML to a format compatible with the SAE training.
    Extracts only the fields that are actually used by the training code.
    
    Args:
        config: The configuration dictionary loaded from YAML
        
    Returns:
        A dictionary with SAE configuration parameters
    """
    # Create an empty SAE config
    sae_config = {}
    
    # Extract fields from the config
    # From 'base' section
    if 'base' in config:
        sae_config['seed'] = config['base'].get('seed')
        sae_config['model_name'] = config['base'].get('model_name')
    
    # From 'sae' section
    if 'sae' in config:
        sae_config['sae_type'] = config['sae'].get('sae_type')
        sae_config['site'] = config['sae'].get('site')
        sae_config['layer'] = config['sae'].get('layer')
        sae_config['act_size'] = config['sae'].get('act_size')
        sae_config['dict_size'] = config['sae'].get('dict_size')
        sae_config['input_unit_norm'] = config['sae'].get('input_unit_norm')
        sae_config['device'] = config['sae'].get('device')
        
        # Handle dtype conversion
        if 'dtype' in config['sae']:
            dtype_str = config['sae']['dtype']
            if dtype_str == 'float32':
                import torch
                sae_config['dtype'] = torch.float32
            elif dtype_str == 'float16':
                import torch
                sae_config['dtype'] = torch.float16
            elif dtype_str == 'bfloat16':
                import torch
                sae_config['dtype'] = torch.bfloat16
    
    # From 'training' section
    if 'training' in config:
        sae_config['dataset_path'] = config['training'].get('dataset_path')
        sae_config['batch_size'] = config['training'].get('batch_size')
        sae_config['model_batch_size'] = config['training'].get('model_batch_size')
        sae_config['lr'] = config['training'].get('lr')
        sae_config['num_tokens'] = config['training'].get('num_tokens')
        sae_config['l1_coeff'] = config['training'].get('l1_coeff')
        sae_config['beta1'] = config['training'].get('beta1')
        sae_config['beta2'] = config['training'].get('beta2')
        sae_config['max_grad_norm'] = config['training'].get('max_grad_norm')
        sae_config['seq_len'] = config['training'].get('seq_len')
        sae_config['wandb_project'] = config['training'].get('wandb_project')
        sae_config['wandb_dir'] = config['training'].get('wandb_dir')
        sae_config['n_batches_to_dead'] = config['training'].get('n_batches_to_dead')
        sae_config['checkpoint_dir'] = config['training'].get('checkpoint_dir')
        sae_config['perf_log_freq'] = config['training'].get('perf_log_freq')
        sae_config['checkpoint_freq'] = config['training'].get('checkpoint_freq')
        sae_config['num_batches_in_buffer'] = config['training'].get('num_batches_in_buffer')
        
        # BatchTopKSAE specific parameters
        sae_config['top_k'] = config['training'].get('top_k')
        sae_config['top_k_aux'] = config['training'].get('top_k_aux')
        sae_config['aux_penalty'] = config['training'].get('aux_penalty')
        sae_config['bandwidth'] = config['training'].get('bandwidth')
    
    # Remove None values from the config
    sae_config = {k: v for k, v in sae_config.items() if v is not None}
    
    # Compute derived fields
    import transformer_lens.utils as utils
    if 'site' in sae_config and 'layer' in sae_config:
        sae_config['hook_point'] = utils.get_act_name(sae_config['site'], sae_config['layer'])
        
        # Compute name if all required fields are present
        required_fields = ['model_name', 'dict_size', 'sae_type', 'top_k', 'lr']
        if all(field in sae_config for field in required_fields):
            sae_config['name'] = (f"{sae_config['model_name']}_{sae_config['hook_point']}_"
                                f"{sae_config['dict_size']}_{sae_config['sae_type']}_"
                                f"{sae_config['top_k']}_{sae_config['lr']}")
    
    return sae_config
