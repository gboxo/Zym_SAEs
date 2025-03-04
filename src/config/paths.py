import os
import platform
import socket
import yaml
import argparse

# Default paths configuration
DEFAULT_PATHS = {
    "local": {
        "model_path": "/users/nferruz/gboxo/models/ZymCTRL/",
        "checkpoints_dir": "/users/nferruz/gboxo/ZymCTRL/checkpoints/",
        "data_dir": "/users/nferruz/gboxo/ZymCTRL_Dataset/",
        "wandb_dir": "/users/nferruz/gboxo/wandb/"
    },
    "compute": {
        "model_path": "/home/woody/b114cb/b114cb23/models/ZymCTRL/",
        "checkpoints_dir": "/home/woody/b114cb/b114cb23/boxo/ZymCTRL/checkpoints/",
        "data_dir": "/home/woody/b114cb/b114cb23/boxo/ZymCTRL_Dataset/",
        "wandb_dir": "/home/woody/b114cb/b114cb23/boxo/wandb/"
    }
}

def detect_environment():
    """
    Auto-detect which environment we're running in (local or compute)
    """
    hostname = socket.gethostname()
    
    # Check if running on a compute cluster
    if 'alex' in hostname:
        return "compute"
    else:
        return "local"

def get_path_config(override_env=None, config_file=None):
    """
    Get path configuration based on priority:
    1. Command line arguments
    2. Environment variables
    3. Config file
    4. Auto-detected environment default

    Args:
        override_env: Manually specify "local" or "compute"
        config_file: Path to custom config file
    """
    # Start with default paths
    paths = {}
    
    # 4. Use auto-detected environment if not overridden
    env = override_env or os.environ.get("SAE_ENV") or detect_environment()
    paths.update(DEFAULT_PATHS.get(env, DEFAULT_PATHS["local"]))
    
    # 3. Load from config file if specified
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            custom_config = yaml.safe_load(f)
            if 'paths' in custom_config:
                paths.update(custom_config['paths'])
    
    # 2. Override with environment variables if they exist
    env_var_map = {
        "SAE_MODEL_PATH": "model_path",
        "SAE_CHECKPOINTS_DIR": "checkpoints_dir",
        "SAE_DATA_DIR": "data_dir",
        "SAE_WANDB_DIR": "wandb_dir"
    }
    
    for env_var, path_key in env_var_map.items():
        if os.environ.get(env_var):
            paths[path_key] = os.environ.get(env_var)
    
    return paths

def add_path_args(parser):
    """
    Add path-related arguments to an ArgumentParser
    """
    parser.add_argument("--model_path", type=str, help="Path to the language model")
    parser.add_argument("--checkpoints_dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--data_dir", type=str, help="Directory containing datasets")
    parser.add_argument("--wandb_dir", type=str, help="Directory for wandb logs")
    parser.add_argument("--env", type=str, choices=["local", "compute"], 
                        help="Override environment detection")
    parser.add_argument("--config_file", type=str, help="Path to custom config file")
    
    return parser

def resolve_paths(args):
    """
    Resolve paths from args, falling back to auto-detection
    """
    # Get base paths from environment detection and config
    paths = get_path_config(
        override_env=args.env if hasattr(args, 'env') and args.env else None,
        config_file=args.config_file if hasattr(args, 'config_file') and args.config_file else None
    )
    
    # 1. Override with command line arguments if they exist
    for key in paths:
        if hasattr(args, key) and getattr(args, key) is not None:
            paths[key] = getattr(args, key)
    
    return paths 