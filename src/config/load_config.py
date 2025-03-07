import yaml

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
    
    # Load the base configuration specified in the experiment config
    base_config_path = experiment_config.pop('base_config')
    base_config = load_yaml(base_config_path)
    
    # Merge the base and experiment-specific configurations
    complete_config = merge_configs(base_config, experiment_config)
    
    return complete_config

# Load complete configuration for an experiment
complete_config = load_experiment_config('experiment_1_config.yaml')
print(complete_config)

