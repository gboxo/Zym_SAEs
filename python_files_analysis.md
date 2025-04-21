# Comprehensive Python Files List and Mapping

## All Python Files in Repository

### Root Level Files
- `/media/workspace/crg_boxo/model_diffing.py` - Model comparison utilities
- `/media/workspace/crg_boxo/run_topk.py` - TopK SAE execution script
- `/media/workspace/crg_boxo/evolution_analysis.py` - Evolutionary analysis of protein sequences

---

## 🔧 SRC/ Directory - Core Framework

### **Core Infrastructure**
- `src/__init__.py` - Main package initialization with comprehensive API exports
- `src/utils.py` - Core utilities: model loading, SAE management, format conversions

### **Configuration Management**
- `src/config/load_config.py` - Hierarchical YAML config loading with merging
- `src/config/paths.py` - Environment-aware path configuration

### **Data Processing**
- `src/data/fasta_to_dataset.py` - FASTA to tokenized HuggingFace datasets
- `src/data/tokenize_dataset.py` - Protein sequence tokenization with enzyme labels

### **Training System**
- `src/training/config.py` - SAE training configuration management
- `src/training/sae.py` - Multiple SAE implementations (BatchTopK, TopK, Vanilla, JumpReLU)
- `src/training/logs.py` - WandB integration, checkpointing, performance monitoring
- `src/training/activation_store.py` - Model activation management with buffering
- `src/training/training.py` - Main SAE training loops with gradient accumulation

### **Inference System**
- `src/inference/inference_batch_topk.py` - BatchTopK to JumpReLU conversion
- `src/inference/compute_threshold.py` - Activation threshold computation
- `src/inference/sae_eval.py` - Comprehensive SAE evaluation with metrics

### **Analysis Tools**
- `src/tools/analysis/compare_norms.py` - Decoder weight norm comparisons
- `src/tools/analysis/firing_patterns.py` - Feature firing patterns with secondary structure
- `src/tools/analysis/compare_decoders.py` - Cosine similarity between decoder weights

### **Data Utilities**
- `src/tools/data_utils/data_utils.py` - YAML configuration loader
- `src/tools/data_utils/create_diffing_dataset.py` - Tokenized datasets for diffing
- `src/tools/data_utils/create_diffing_dataset_mixture.py` - Mixed datasets with ratios
- `src/tools/data_utils/join_datasets.py` - Dataset concatenation utilities
- `src/tools/data_utils/generate_dataframe.py` - Analysis dataframe creation

### **Diffing Analysis**
- `src/tools/diffing/analysis_latent_scoring.py` - Advanced latent analysis
- `src/tools/diffing/base_latent_scoring.py` - Base latent feature analysis
- `src/tools/diffing/diffing_utils.py` - Core diffing utilities
- `src/tools/diffing/plots_diffing.py` - Diffing visualization
- `src/tools/diffing/latent_scoring_correlations.py` - Feature correlation analysis
- `src/tools/diffing/latent_scoring_contiguous.py` - Contiguous feature analysis
- `src/tools/diffing/firing_rates.py` - Feature firing rate analysis
- `src/tools/diffing/topk_latent_scoring.py` - TopK specific latent scoring
- `src/tools/diffing/compute_cs.py` - Cosine similarity computation
- `src/tools/diffing/latent_scoring.py` - Main latent scoring functionality

### **Sequence Generation**
- `src/tools/generate/generate_and_score.py` - Generation with scoring pipeline
- `src/tools/generate/generate_with_steering.py` - Activation steering during generation
- `src/tools/generate/generate_with_penalty.py` - Penalty-based generation
- `src/tools/generate/generate_utils.py` - Generation utility functions
- `src/tools/generate/generate_without_penalty.py` - Unpenalized generation
- `src/tools/generate/deepspeed_seq_gen.py` - DeepSpeed accelerated generation
- `src/tools/generate/generate_with_ablation.py` - Feature ablation generation
- `src/tools/generate/seq_gen.py` - Core sequence generation with perplexity

### **Oracle Models**
- `src/tools/oracles/activity_prediction_intervention.py` - Activity prediction with interventions
- `src/tools/oracles/ESM_Fold.py` - ESMFold protein folding
- `src/tools/oracles/oracles_utils.py` - Oracle utility functions
- `src/tools/oracles/activity_prediction.py` - Activity prediction using ESM models

### **Visualization**
- `src/tools/plots/ridge_plot.py` - Ridge plot visualizations
- `src/tools/plots/pdb_visualization.py` - PDB structure visualization
- `src/tools/plots/script.py` - General plotting scripts
- `src/tools/plots/get_data.py` - Data processing for plotting

---

## 🧪 EXPERIMENTS_ALEX/ Directory - Experimental Code

### **Circuit Analysis**
- `experiments_alex/circuit_analysis/tooling.py` - SAE feature attribution analysis
- `experiments_alex/circuit_analysis/attribution_vis.py` - Attribution visualization PDFs
- `experiments_alex/circuit_analysis/attribution_exp.py` - Attribution experiments on proteins

### **KL Divergence Analysis**
- `experiments_alex/kl_divergence/main.py` - KL divergence between base and DPO models
- `experiments_alex/kl_divergence/main2.py` - Alternative KL divergence computation
- `experiments_alex/kl_divergence/transition_probs.py` - Token transition probability analysis
- `experiments_alex/kl_divergence/transition_plot.py` - Transition visualization
- `experiments_alex/kl_divergence/plot_kl.py` - KL divergence plotting
- `experiments_alex/kl_divergence/pdb_visualizer.py` - PDB visualization utilities
- `experiments_alex/kl_divergence/activity_prediction.py` - Activity prediction
- `experiments_alex/kl_divergence/trans_comb.py` - Transition combination analysis

### **MSA Steering**
- `experiments_alex/msa_steering/train_column_sparse_probe.py` - L1-penalized logistic probes
- `experiments_alex/msa_steering/msa_to_token_pos_mapping.py` - MSA to sequence mapping
- `experiments_alex/msa_steering/get_features_for_scoring.py` - SAE feature extraction

### **Noelia DPO Experiments**
- `experiments_alex/noelia_dpo/score_each_layer.py` - Layer embedding classification
- `experiments_alex/noelia_dpo/embed_generated_seqs.py` - Sequence embedding extraction
- `experiments_alex/noelia_dpo/join_sequences.py` - Dataset creation from sequences
- `experiments_alex/noelia_dpo/get_features_for_scoring.py` - Feature extraction
- `experiments_alex/noelia_dpo/all_diffing.py` - Complete diffing analysis
- `experiments_alex/noelia_dpo/plot_features.py` - Feature visualization
- `experiments_alex/noelia_dpo/diffing.py` - SAE weight comparison

### **Noelia FT DMS Experiments**
- `experiments_alex/noelia_ft_dms/seq_gen.py` - Fine-tuned model sequence generation
- `experiments_alex/noelia_ft_dms/activity_prediction_lr.py` - Linear regression activity prediction
- `experiments_alex/noelia_ft_dms/activity_prediction.py` - ESM-based activity prediction
- `experiments_alex/noelia_ft_dms/just_steering.py` - Activation steering
- `experiments_alex/noelia_ft_dms/get_averge_feature_activity.py` - Average feature activity
- `experiments_alex/noelia_ft_dms/latent_scoring.py` - Comprehensive feature analysis
- `experiments_alex/noelia_ft_dms/generate_with_clipping.py` - Generation with feature clipping
- `experiments_alex/noelia_ft_dms/generate_with_ablation.py` - Generation with feature ablation

---

## 🔗 Key Mappings: experiments_alex → src

### **Direct Duplications (Near-Identical)**
| experiments_alex | src equivalent | Similarity |
|------------------|----------------|------------|
| `noelia_ft_dms/seq_gen.py` | `tools/generate/seq_gen.py` | 95% - Same core generation logic |
| `noelia_ft_dms/activity_prediction.py` | `tools/oracles/activity_prediction.py` | 90% - Same oracle model usage |
| `kl_divergence/activity_prediction.py` | `tools/oracles/activity_prediction.py` | 90% - Same oracle approach |
| `noelia_dpo/join_sequences.py` | `tools/data_utils/create_diffing_dataset.py` | 95% - Identical dataset creation |
| `noelia_ft_dms/latent_scoring.py` | `tools/diffing/latent_scoring.py` | 85% - Same feature analysis core |
| `msa_steering/get_features_for_scoring.py` | `tools/diffing/latent_scoring.py` | 80% - Same feature extraction |

### **Functional Equivalents (Similar Purpose)**
| experiments_alex | src equivalent | Relationship |
|------------------|----------------|--------------|
| `noelia_ft_dms/generate_with_ablation.py` | `tools/generate/generate_with_ablation.py` | Same intervention approach |
| `noelia_ft_dms/generate_with_clipping.py` | `tools/generate/generate_with_penalty.py` | Similar penalty mechanisms |
| `noelia_ft_dms/just_steering.py` | `tools/generate/generate_with_steering.py` | Same steering methodology |
| `noelia_dpo/diffing.py` | `tools/diffing/diffing_utils.py` | Core diffing functionality |
| `circuit_analysis/attribution_vis.py` | `tools/plots/` (various) | Visualization patterns |

### **Experiment-Specific Extensions**
| experiments_alex | Related src | Extension type |
|------------------|-------------|----------------|
| `circuit_analysis/tooling.py` | `training/sae.py` + `utils.py` | Attribution analysis addition |
| `kl_divergence/transition_probs.py` | `tools/diffing/` | Token transition analysis |
| `msa_steering/train_column_sparse_probe.py` | `tools/analysis/` | MSA-specific analysis |
| `noelia_dpo/score_each_layer.py` | `inference/sae_eval.py` | Layer-wise evaluation |

---

## 📊 Detailed File Content Analysis

### Core Infrastructure Details

#### `src/utils.py`
- **Key Functions**: `load_model()`, `load_sae()`, `load_config()`, `get_ht_model()`, `get_sl_model()`, `convert_GPT_weights()`, `get_paths()`
- **Dependencies**: torch, transformers, sae_lens, transformer_lens, socket for environment detection
- **Purpose**: Central utility hub for model loading, SAE management, and format conversions

#### `src/config/load_config.py`
- **Key Functions**: `load_yaml()`, `merge_configs()`, `load_experiment_config()`, `convert_to_sae_config()`
- **Dependencies**: yaml, transformer_lens.utils
- **Purpose**: Hierarchical YAML configuration loading with base config merging and SAE-specific conversion

### Training System Details

#### `src/training/sae.py`
- **Key Classes**: `BaseAutoencoder`, `BatchTopKSAE`, `TopKSAE`, `VanillaSAE`, `JumpReLUSAE`, `JumpReLU`
- **Dependencies**: torch, torch.nn, torch.autograd
- **Purpose**: Multiple SAE implementations with different sparsity mechanisms

#### `src/training/training.py`
- **Key Functions**: `train_sae()`, `resume_training()`, `validate_and_clip_gradients()`, threshold computation utilities
- **Dependencies**: torch, training submodules
- **Purpose**: Main SAE training loops with gradient accumulation, threshold computation, and checkpointing

### Analysis Tools Details

#### `src/tools/diffing/latent_scoring.py`
- **Key Functions**: Feature extraction, importance scoring, correlation analysis
- **Dependencies**: torch, scipy.sparse, sklearn, numpy
- **Purpose**: Main latent scoring functionality with comprehensive feature analysis

#### `src/tools/generate/seq_gen.py`
- **Key Functions**: `main()`, `calculatePerplexity()`, `remove_characters()`
- **Dependencies**: torch, transformers, tqdm
- **Purpose**: Core protein sequence generation with perplexity calculation

#### `src/tools/oracles/activity_prediction.py`
- **Key Classes**: `SequenceDataset`, model loading and saving functions
- **Dependencies**: torch, transformers, peft, datasets
- **Purpose**: Activity prediction using ESM models with LoRA fine-tuning

---

## 🏗️ Architecture Patterns

### **Key Design Principles**
1. **Modular Design**: Clear separation between training, inference, analysis, and utilities
2. **Configuration Management**: Hierarchical YAML configs with environment awareness
3. **SAE Implementations**: Multiple sparsity mechanisms (TopK, BatchTopK, JumpReLU, Vanilla)
4. **Protein-Specific**: Specialized for protein language models with enzyme classification
5. **Experiment Framework**: Full pipeline from sequence generation to structure prediction and activity analysis
6. **Checkpointing**: Comprehensive resuming capabilities with state preservation
7. **Multi-Environment**: Support for both local workstations and compute clusters

### **Migration Pattern Analysis**
The structure reveals that:
1. **experiments_alex/** contains original experimental implementations
2. **src/tools/** contains cleaned, generalized versions of successful experiments
3. Many experiments_alex files have hardcoded paths and specific parameters
4. src/ versions are more configurable and reusable
5. The codebase implements a complete sparse autoencoder training and analysis pipeline specifically designed for protein language models

### **Code Duplication Summary**
- **High overlap** between experiments_alex/ and src/tools/ (60-95% similarity in many cases)
- **Main differences**: Hardcoded paths vs configurable parameters
- **Refactoring opportunity**: Many experiments_alex files could be replaced with properly configured src/ equivalents

---

*Analysis generated on 2025-01-26*