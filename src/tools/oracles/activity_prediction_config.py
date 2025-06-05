from dataclasses import dataclass
import yaml


@dataclass
class ActivityPredictionConfig:


    cfg_path: str
    batch_size: int
    seq_source: str
    out_dir: str
    label: str


    oracle_path1: str
    checkpoint_path1: str
    oracle_path2: str
    checkpoint_path2: str


    @classmethod
    def from_yaml(cls, cfg_path: str, **overrides):
        """Load config from YAML with optional overrides from CLI args."""
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
        
        paths = config["paths"]
        return cls(
            cfg_path=cfg_path,
            batch_size=overrides.get('batch_size', 16),
            seq_source=paths["seqs_path"],
            out_dir=paths["output_path"], 
            label=config["label"],
            oracle_path1=paths["oracle_path1"],
            checkpoint_path1=paths["checkpoint_path1"],
            oracle_path2=paths["oracle_path2"],
            checkpoint_path2=paths["checkpoint_path2"]
        )



