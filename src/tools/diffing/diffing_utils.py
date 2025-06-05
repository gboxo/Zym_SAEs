
import yaml

def load_config(cfg_path):
    """
    Load configuration from a YAML file.
    
    Args:
        cfg_path (str): Path to the configuration file.
        
    Returns:
        dict: Configuration parameters.
    """
    with open(cfg_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
