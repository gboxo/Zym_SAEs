from dataclasses import dataclass, field



@dataclass
class LatentScoringConfig:
    hook_point: str
    is_DMS: bool
    label: str
    model_name: str


    # Dataframe columns
    seq_col_id: str
    pred_col_id: str
    col_id: str

    # Paths
    df_path: str
    model_path: str
    sae_path: str
    out_dir: str

    # Scoring parameters
    prefix_tokens: int
    percentiles: list
    min_rest_fraction: list











