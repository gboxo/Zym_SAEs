## YAML Configs

### Workstation

  seed: 49
  batch_size: 4096
  lr: 0.0003
  num_tokens: 1000000000
  l1_coeff: 0
  beta1: 0.9
  beta2: 0.99
  max_grad_norm: 100000
  seq_len: 128
  dtype: torch.float32
  model_name: ZymCTRL
  model_path: AI4PD/ZymCTRL
  site: resid_pre
  layer: 26
  act_size: 1280
  dict_size: 10240
  device: cuda:0
  model_batch_size: 256
  dataset_path: /users/nferruz/gboxo/ZymCTRL_Dataset/final_dataset_big/
  wandb_project: ZymCTRL_SAE
  input_unit_norm: true
  perf_log_freq: 1000
  sae_type: BatchTopKSAE
  checkpoint_freq: 10000
  n_batches_to_dead: 5
  checkpoint_dir: /users/nferruz/gboxo/ZymCTRL/checkpoints/
  wandb_dir: /users/nferruz/gboxo/wandb/
  top_k: 100 
  top_k_aux: 512
  aux_penalty: 0.01  # 1/32
  bandwidth: 0.001


### Alex

  seed: 49
  batch_size: 4096
  lr: 0.0003
  num_tokens: 1000000000
  l1_coeff: 0
  beta1: 0.9
  beta2: 0.99
  max_grad_norm: 100000
  seq_len: 256
  dtype: torch.float32
  model_name: ZymCTRL
  model_path: /home/woody/b114cb/b114cb23/models/ZymCTRL/
  site: resid_pre
  layer: 26
  act_size: 1280
  dict_size: 10240
  device: cuda:0
  model_batch_size: 512
  dataset_path: /home/woody/b114cb/b114cb23/boxo/final_dataset_big/
  wandb_project: ZymCTRL_SAE
  input_unit_norm: true
  perf_log_freq: 1000
  sae_type: BatchTopKSAE
  checkpoint_freq: 10000
  n_batches_to_dead: 5
  checkpoint_dir: /home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/
  wandb_dir: /home/woody/b114cb/b114cb23/boxo/
  top_k: 100
  top_k_aux: 512
  aux_penalty: 0.01 # 1/100
  bandwidth: 0.001

## Base Config
"seed": 49,
"batch_size": 4096,
"lr": 3e-4,
"num_tokens": int(1e9),
"l1_coeff": 0,
"beta1": 0.9,
"beta2": 0.99,
"max_grad_norm": 100000,
"seq_len": 128,
"dtype": torch.float32,
"model_name": "gpt2-small",
"site": "resid_pre",
"layer": 8,
"act_size": 768,
"dict_size": 12288,
"device": "cuda:0",
"model_batch_size": 512,
"num_batches_in_buffer": 10,
"dataset_path": "Skylion007/openwebtext",
"wandb_project": "sparse_autoencoders",
"wandb_dir": "/home/woody/b114cb/b114cb23/boxo/",
"input_unit_norm": True,
"perf_log_freq": 1000,
"sae_type": "topk",
"checkpoint_freq": 10000,
"n_batches_to_dead": 5,
"checkpoint_dir": "",
"top_k": 32,
"top_k_aux": 512,
"aux_penalty": (1/32),
"bandwidth": 0.001,
"hook_point":
"name":



## Model Diffing

config = "configs/workstation.yaml"
checkpoint_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_04_03_25_hhook_resid_pre_1280_batchtopk_100_0.0003_resumed/checkpoint_latest.pt"
dataset_path = "Data/Diffing/tokenized_train_dataset_iteration1_rounds0to10"
cfg["use_wandb"] = True
cfg["model_type"] 
cfg["model_name"]
loaded_cfg["name"] = "Model_Diffing_M0_D9"
loaded_cfg["dtype"] = torch.float32
loaded_cfg["wandb_project"] = "Model_Diffing_RL" 
loaded_cfg["batch_size"] = 512
loaded_cfg["top_k"] = 100
loaded_cfg["aux_penalty"] = 0
loaded_cfg["top_k_aux"] = 0
loaded_cfg["model_batch_size"] = 128 
loaded_cfg["dataset_path"] = dataset_path
loaded_cfg["perf_log_freq"] = 1
loaded_cfg["model_name"] = cfg["model_name"]
loaded_cfg["num_batches_in_buffer"] = 5 
loaded_cfg["n_iters"] = 80*9
loaded_cfg["model_type"] = "BatchTopKSAE"
loaded_cfg["use_wandb"] = True
loaded_cfg["num_tokens"] = 256*152*9
loaded_cfg["checkpoint_dir"] = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/Diffing/checkpoints/"
loaded_cfg["threshold_compute_freq"] = 1
loaded_cfg["threshold_num_batches"] = 40 
loaded_cfg["lr"] = 5e-4
cfg["wandb_dir"]
cfg["wandb_project"]
cfg["wandb_run_id"]
cfg["resume_from"] 
cfg["resume_history"]
cfg["model_type"] = "BatchTopK"
cfg["model_path"]





