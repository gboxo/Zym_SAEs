# Overview

We've implemented a pipeline to train and analyze Sparse Autoencoders for the protein Language Model ZymCTRL. 
The pipeline includes:

1. **Sequence Generation**: Generating sequences for multiple checkpoints of a DPO process to make ZymCTRL output highly active alpha-amylases (3.2.1.1)
2. **Protein Folding**: Folding the generated sequences using ESMFold
3. **Activity Prediction**: Predicting activity using an oracle model
4. **SAE Training**: Fine-tuning Sparse Autoencoders on these sequences
5. **Feature Analysis**: Analyzing the evolution and fitness of features

## Experiment Files Documentation

### ESM_Fold.py
- **Purpose**: Folds protein sequences using ESMFold model
- **Input**:
  - Iteration number (--iteration_num)
  - EC label (--label, e.g. "3.2.1.1")
  - Procedure type (--procedure: "diffing", "steering", or "ablation")
- **Output**:
  - PDB files for each folded sequence
  - Files stored in: `/home/woody/b114cb/b114cb23/boxo/outputs_{procedure}/output_iterations{iteration_num}/PDB/`
- **Key Features**:
  - Loads ESMFold model
  - Processes sequences in batches
  - Handles different procedure types (diffing, steering, ablation)
  - Saves PDB files with proper naming conventions
  - Manages CUDA memory efficiently
  - Handles sequence formatting and preprocessing

### activity_prediction.py  
- **Purpose**: Predicts protein activity using ESM models
- **Input**:
  - Iteration number (--iteration_num)
  - EC label (--label)
  - Sequence data from: `/home/woody/b114cb/b114cb23/boxo/seq_gens/seq_gen_{ec_label}_iteration{iteration_num}.fasta`
- **Output**:
  - Activity predictions file: `/home/woody/b114cb/b114cb23/boxo/activity_predictions/activity_prediction_iteration{iteration_num}.txt`
- **Key Features**:
  - Uses two ESM models (esm2_t33_650M_UR50D and esm1v_t33_650M_UR90S_1)
  - Averages predictions from both models
  - Saves predictions with sequence IDs
  - Handles CUDA memory management
  - Processes sequences in batches
  - Implements custom SequenceDataset class
  - Uses LoRA adapters for efficient fine-tuning

### generate_with_ablation.py
- **Purpose**: Generates sequences with feature ablation
- **Input**:
  - Model iteration number
  - Data iteration number  
  - Top correlated features from: `/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/correlations/top_correlations_M{model_iteration}_D{data_iteration}.pkl`
- **Output**:
  - Ablated sequences for each feature
  - Files stored in: `/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/ablation/M{model_iteration}_D{data_iteration}/`
- **Key Features**:
  - Ablates top correlated features
  - Generates multiple samples per feature
  - Uses SAE model for feature manipulation
  - Handles CUDA memory cleanup
  - Implements custom ablation hook
  - Supports batch generation
  - Manages model checkpoint loading

### generate_dataframe.py
- **Purpose**: Creates unified dataframe from analysis data
- **Input**:
  - Iteration number (--iteration_num)
  - EC label (--label)
  - Sequence data from: `/home/woody/b114cb/b114cb23/boxo/seq_gens/seq_gen_3.2.1.1_iteration{iteration_num-1}.fasta`
  - Activity predictions from: `/home/woody/b114cb/b114cb23/boxo/activity_predictions/activity_prediction_iteration{iteration_num-1}.txt`
  - TM scores from: `/home/woody/b114cb/b114cb23/boxo/outputs/TM_scores/TM_scores_{label}_iteration{iteration_num-1}`
  - pLDDT scores from PDB files
- **Output**:
  - CSV file: `/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/dataframe_iteration{iteration_num-1}.csv`
- **Key Features**:
  - Merges sequence, activity, and structure data
  - Computes average pLDDT scores
  - Handles missing data
  - Creates comprehensive analysis dataframe
  - Implements custom pLDDT extraction
  - Validates input data
  - Handles file I/O efficiently

### sequence_distribution.py
- **Purpose**: Analyzes sequence distribution using embeddings
- **Input**:
  - Sequence embeddings from ESM model
  - Sequence data from FASTA files
- **Output**:
  - TSNE plots and density visualizations
  - Files saved in: `/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/figures/`
- **Key Features**:
  - Computes sequence embeddings
  - Performs dimensionality reduction
  - Creates 2D density plots
  - Visualizes sequence clusters
  - Generates interactive plots
  - Implements Gaussian KDE
  - Handles large datasets efficiently
  - Produces publication-quality figures

### latent_scoring.py
- **Purpose**: Analyzes latent space features
- **Input**:
  - Iteration number (--iteration_num)
  - EC label (--label)
  - Sequence features from SAE
  - CS values from: `/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/all_cs.pt`
- **Output**:
  - Feature correlations and visualizations
  - Files saved in: `/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/figures/`
- **Key Features**:
  - Computes feature correlations with metrics
  - Analyzes firing rates
  - Creates interactive 2D/3D visualizations
  - Performs statistical analysis
  - Generates heatmaps and density plots
  - Implements multiple hypothesis correction
  - Produces interactive HTML visualizations
  - Handles large feature matrices efficiently

### create_diffing_dataset.py
- **Purpose**: Creates datasets for SAE training
- **Input**:
  - Iteration number (--iteration_num)
  - EC label (--label)
  - Sequence data from FASTA files
- **Output**:
  - Tokenized dataset in HuggingFace format
  - Saved to: `/home/woody/b114cb/b114cb23/boxo/diffing_datasets/dataset_iteration{iteration_num}`
- **Key Features**:
  - Formats sequences with EC labels
  - Shuffles and splits data
  - Creates train/eval splits
  - Saves dataset in HuggingFace format
  - Handles sequence formatting
  - Implements custom tokenization
  - Manages dataset versioning
  - Ensures reproducibility with fixed random seed

### analysis_seqs.py
- **Purpose**: Analyzes ablated sequences
- **Input**:
  - Ablated sequence files from: `/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/ablation/`
- **Output**:
  - Sequence statistics and visualizations
- **Key Features**:
  - Computes sequence length distributions
  - Visualizes sequence properties
  - Analyzes ablation effects
  - Generates KDE plots
  - Handles multiple ablation features
  - Produces publication-quality figures
  - Implements efficient sequence processing

### alpha_amylase.py
- **Purpose**: Computes cosine similarities between SAE weights
- **Input**:
  - SAE weight matrices from checkpoints
- **Output**:
  - Cosine similarity metrics
  - Saved to: `/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/all_cs.pt`
- **Key Features**:
  - Compares weights across iterations
  - Computes stage-wise similarities
  - Filters stable features
  - Generates violin plots
  - Handles large weight matrices
  - Implements efficient similarity computation
  - Produces publication-quality visualizations

### run_folding.sh
- **Purpose**: Runs protein folding using ESMFold
- **Key Features**:
  - Processes array of iterations (6-9)
  - Uses A40 GPU
  - 24 hour time limit
  - Handles proxy settings
  - Manages CUDA/cuDNN environment
  - Sets up Python environment
  - Handles SLURM job array
  - Manages output/error logging

### SW_MD_RL_slurm.sh
- **Purpose**: Runs SAE training for RL models
- **Key Features**:
  - Processes iterations 1-10
  - Uses A100 GPU
  - 10 hour time limit
  - Handles model path switching
  - Creates config files dynamically
  - Manages checkpoints and resuming
  - Sets up environment variables
  - Handles proxy configuration
  - Manages GPU resources

### SW_MD_BM_slurm.sh
- **Purpose**: Runs SAE training for baseline models
- **Key Features**:
  - Processes iterations 1-10
  - Uses A100 GPU
  - 10 hour time limit
  - Handles model path switching
  - Creates config files dynamically
  - Manages checkpoints and resuming
  - Sets up environment variables
  - Handles proxy configuration
  - Manages GPU resources

### run_activity_prediction.sh
- **Purpose**: Uses an oracle to predict the activity of generated sequences
- **Input**:
  - Sequence FASTA files from: `/home/woody/b114cb/b114cb23/boxo/seq_gens/seq_gen_3.2.1.1_iteration{iteration_num}.fasta`
  - Pretrained ESM models:
    - esm2_t33_650M_UR50D
    - esm1v_t33_650M_UR90S_1
- **Output**:
  - Activity predictions file: `/home/woody/b114cb/b114cb23/boxo/activity_predictions/activity_prediction_iteration{iteration_num}.txt`
- **Key Features**:
  - Uses two ESM models for ensemble prediction
  - Averages predictions from both models
  - Handles CUDA memory management
  - Processes sequences in batches
  - Saves predictions with sequence IDs
  - Runs on A100 GPU with 10 hour time limit
  - Sets up environment variables
  - Handles proxy configuration
  - Manages GPU resources

### create_dataframe.sh
- **Purpose**: Creates unified dataframes from analysis data
- **Key Features**:
  - Processes iterations 1-6
  - Sets up proxy and environment
  - Calls generate_dataframe.py for each iteration
  - Handles environment configuration
  - Manages file I/O
  - Ensures proper data formatting

### run_generation.sh
- **Purpose**: Generates protein sequences using ZymCTRL
- **Key Features**:
  - Runs sequence generation for iterations 2-30
  - Uses A40 GPU
  - 24 hour time limit
  - Handles proxy settings for external connections
  - Sets up Python environment
  - Manages CUDA/cuDNN environment
  - Handles SLURM job array
  - Manages output/error logging




## Additional Scripts
- `bash scripts/create_dataset_diffing.sh`
- `bash scripts/run_train_sae_2b.sh`
- `python3 -m experiments.Diffing_Analysis.alpha_amylase`
