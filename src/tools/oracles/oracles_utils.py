import yaml

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

