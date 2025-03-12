
from src.training.training import resume_training
from src.training.logs import init_wandb
from src.utils import get_ht_model
from src.utils import load_model
from src.config.load_config import load_experiment_config, convert_to_sae_config
import torch
from types import SimpleNamespace
EVALUATE = False

if EVALUATE:
    tokenizer, model = load_model("AI4PD/ZymCTRL")
    config = model.config
    config.attn_implementation = "eager"
    config.d_model = 10240

    model = get_ht_model(model,config, tokenizer)


    # === EVALUATE ===
    from src.training.activation_store import ActivationsStore
    from src.utils import load_sae
    torch.cuda.empty_cache()

    # === POST TRAINING ===
    checkpoint_dir = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_06_03_25_hhook_resid_pre_1280_batchtopk_100_0.0005_resumed/"

    cfg,post_sae = load_sae(checkpoint_dir)
    post_sae.eval()


    cfg["use_wandb"] = False
    cfg["model_batch_size"] = 128
    cfg["batch_size"] = 64
    cfg["n_iters"] = 10
    cfg["dataset_path"] = "Data/Diffing/tokenized_eval_dataset_iteration1_rounds0to10"

    activation_store = ActivationsStore(model,cfg)
    batch = activation_store.next_batch()

    loss = post_sae(batch)["loss"]
    print("Post-training loss: ", loss)
"""

    # === PRE TRAINING ===
    checkpoint_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000/"
    cfg,sae_pre = load_sae(checkpoint_path)
    sae_pre.eval()
    cfg = loaded_cfg
    cfg["use_wandb"] = False
    cfg["model_batch_size"] = 128
    cfg["batch_size"] = 64
    cfg["n_iters"] = 10
    cfg["dataset_path"] = "Data/Diffing/tokenized_eval_dataset_iteration1"


    activation_store = ActivationsStore(model,cfg)
    batch = activation_store.next_batch()
    loss = sae_pre(batch)["loss"]
    print("Pre-training loss: ", loss)
"""