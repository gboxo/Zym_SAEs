import torch
from utils import load_sae, load_model, get_ht_model
from activation_store import ActivationsStore
from training import train_sae
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def compute_position_wise_loss(sae, model, cfg, num_samples=100, seq_length=256):
    """
    Compute mean loss at each position in the sequence.
    
    Args:
        sae: The sparse autoencoder model
        model: The language model
        cfg: Configuration dictionary
        num_samples: Number of sequences to average over
        seq_length: Length of sequences to analyze
        
    Returns:
        numpy array of shape (seq_length,) containing mean losses
    """
    position_losses = np.zeros(seq_length)
    counts = np.zeros(seq_length)
    
    # Create activation store for getting samples
    activation_store = ActivationsStore(model, cfg)
    
    sae.eval()
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Computing position-wise loss"):
            # Get batch of activations
            batch = activation_store.next_batch()
            batch = batch.reshape(-1, seq_length, cfg["act_size"])
            
            # Compute SAE output for each position
            for pos in range(seq_length):
                pos_activations = batch[:, pos, :]
                sae_output = sae(pos_activations)
                position_losses[pos] += sae_output["l2_loss"].item() * batch.shape[0]
                counts[pos] += batch.shape[0]
    
    # Compute means
    mean_losses = position_losses / counts
    return mean_losses

def plot_position_wise_loss(losses, save_path):
    """
    Plot and save the position-wise loss graph.
    
    Args:
        losses: numpy array of losses per position
        save_path: path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(losses)), losses, '-', linewidth=2)
    plt.xlabel('Sequence Position')
    plt.ylabel('Mean Reconstruction Loss')
    plt.title('SAE Reconstruction Loss by Sequence Position')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
    # Also save the raw data
    np.save(save_path.replace('.png', '.npy'), losses)

def finetune_sae(args):
    # Existing initialization code...
    cfg, sae = load_sae(args.sae_path)
    
    # Update config with new parameters
    cfg.update({
        "num_tokens": args.num_tokens,
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "dataset_path": args.dataset_path,
        "wandb_project": f"{cfg['wandb_project']}_finetuned",
    })

    # Load the model and tokenizer
    tokenizer, model_ht = load_model(args.model_path)
    
    # Convert to HookedTransformer format
    config = model_ht.config
    config.d_mlp = 5120  # Adjust based on your model
    model = get_ht_model(model_ht, config, tokenizer=tokenizer)
    
    # Compute initial position-wise loss
    print("Computing initial position-wise loss...")
    initial_losses = compute_position_wise_loss(sae, model, cfg)
    plot_position_wise_loss(initial_losses, 
                          f"{args.sae_path}/position_wise_loss_before_finetuning.png")
    
    # Create activation store for new dataset
    activation_store = ActivationsStore(model, cfg)
    
    # Fine-tune the SAE
    train_sae(sae, activation_store, model, cfg)
    
    # Compute final position-wise loss
    print("Computing final position-wise loss...")
    final_losses = compute_position_wise_loss(sae, model, cfg)
    plot_position_wise_loss(final_losses, 
                          f"{args.sae_path}/position_wise_loss_after_finetuning.png")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(initial_losses)), initial_losses, '-', 
             label='Before Fine-tuning', alpha=0.7)
    plt.plot(range(len(final_losses)), final_losses, '-', 
             label='After Fine-tuning', alpha=0.7)
    plt.xlabel('Sequence Position')
    plt.ylabel('Mean Reconstruction Loss')
    plt.title('SAE Reconstruction Loss by Sequence Position')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.sae_path}/position_wise_loss_comparison.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune a pre-trained SAE')
    parser.add_argument('--sae_path', type=str, required=True,
                      help='Path to the pre-trained SAE checkpoint directory')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the language model')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the new dataset')
    parser.add_argument('--num_tokens', type=int, default=1_000_000,
                      help='Number of tokens to use for fine-tuning')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                      help='Learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=4096,
                      help='Batch size for training')
    args = parser.parse_args()
    finetune_sae(args) 