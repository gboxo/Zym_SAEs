import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pkl
import torch
import seaborn as sns


def compute_neuron_metrics(data, num_neurons=15360):
    """
    Compute various firing metrics for neurons across sequences.
    
    Args:
        data: List of tensors containing neuron activation data
        num_neurons: Number of neurons to analyze
        
    Returns:
        Dictionary of computed metrics
    """
    device = data[0].device if data else torch.device('cpu')

    # Initialize metrics
    total_firings = torch.zeros(num_neurons, device=device)
    sequence_counts = torch.zeros(num_neurons, device=device)
    max_position = torch.zeros(num_neurons, device=device)
    min_position = torch.zeros(num_neurons, device=device)
    total_strikes = torch.zeros(num_neurons, device=device)
    total_strike_lengths = torch.zeros(num_neurons, device=device)
    firing_positions = [[] for _ in range(num_neurons)]
    all_strikes = [[[] for _ in range(num_neurons)] for _ in range(len(data))]

    for seq_idx, seq_tensor in tqdm(enumerate(data)):
        seq_len = seq_tensor.size(0)
        active_pos, active_neurons = torch.where(seq_tensor > 0)

        # Update total firings (metric 1)
        total_firings += torch.bincount(active_neurons, minlength=num_neurons)

        # Update sequence counts (metric 2)
        unique_neurons = torch.unique(active_neurons)
        sequence_counts[unique_neurons] += 1

        # Update max position (for metric 4)
        for neuron in unique_neurons:
            neuron_mask = (active_neurons == neuron)
            current_max = active_pos[neuron_mask].max().item() if neuron_mask.any() else -1
            current_min = active_pos[neuron_mask].min().item() if neuron_mask.any() else -1
            if current_max > max_position[neuron]:
                max_position[neuron] = current_max

            if current_min < min_position[neuron] or min_position[neuron] == 0:
                min_position[neuron] = current_min
                
        # Process firing positions and strikes
        if len(active_neurons) == 0:
            continue

        # Sort by neuron to group positions
        sorted_indices = torch.argsort(active_neurons)
        active_neurons_sorted = active_neurons[sorted_indices]
        active_pos_sorted = active_pos[sorted_indices]

        # Split into neuron groups
        unique_neurons, inverse_indices = torch.unique(active_neurons_sorted, return_inverse=True)
        counts = torch.bincount(inverse_indices)
        split_pos = torch.split(active_pos_sorted, counts.tolist())

        for i, neuron in enumerate(unique_neurons):
            neuron = neuron.item()
            positions = split_pos[i].cpu().tolist()
            
            # Collect firing positions (metric 3)
            firing_positions[neuron].extend(positions)
            
            # Calculate strikes
            if not positions:
                continue
                
            positions.sort()
            strikes = []
            current_start = positions[0]
            current_end = positions[0]
            
            for pos in positions[1:]:
                if pos == current_end + 1:
                    current_end = pos
                else:
                    strikes.append((current_start, current_end))
                    current_start = pos
                    current_end = pos
            strikes.append((current_start, current_end))

            # Update strike metrics (metrics 5 & 6)
            num_strikes = len(strikes)
            total_strikes[neuron] += num_strikes
            total_strike_lengths[neuron] += sum(end - start + 1 for start, end in strikes)
            all_strikes[seq_idx][neuron].extend(strikes)

    # Calculate derived metrics
    percentage_firings = (sequence_counts / len(data)) * 100  # Metric 2
    avg_strike_length = torch.zeros_like(total_strike_lengths)
    mask = total_strikes > 0
    avg_strike_length[mask] = total_strike_lengths[mask] / total_strikes[mask]  # Metric 5
    avg_strikes_per_seq = total_strikes / len(data)  # Metric 6
    neurons_first_10 = torch.where(max_position < 10)[0].tolist()  # Metric 4

    return {
        "total_firings": total_firings,          # Metric 1
        "percentage_firings": percentage_firings,
        "firing_positions": firing_positions,    # Metric 3 (raw position data)
        "neurons_first_10": neurons_first_10,
        "avg_strike_length": avg_strike_length,
        "avg_strikes_per_seq": avg_strikes_per_seq,
        "max_position": max_position,
        "min_position": min_position,
        "all_strikes": all_strikes,
    }


def load_data(path):
    """
    Load and preprocess data from a pickle file.
    
    Args:
        path: Path to the pickle file
        
    Returns:
        List of tensor data
    """
    with open(path, "rb") as f:
        data = pkl.load(f)
    
    return [torch.tensor(dat.todense(), dtype=torch.float32) for dat in data]


def plot_percentage_firings(metrics):
    """Plot distribution of firing percentage across neurons."""
    percentage_firings = metrics["percentage_firings"]
    percentage_firings = percentage_firings[percentage_firings > 0]
    
    plt.figure(figsize=(12, 6))
    sns.histplot(percentage_firings, bins=100)
    plt.xlabel("Percentage of sequences fired")
    plt.ylabel("Number of neurons")
    plt.title("Distribution of firing percentage across neurons")
    plt.show()


def plot_max_position(metrics):
    """Plot distribution of maximum firing positions."""
    max_position = metrics["max_position"]
    max_position = max_position[max_position > 0]
    
    plt.figure(figsize=(12, 6)) 
    plt.hist(max_position, bins=100)
    plt.xlabel("Max position of firing")
    plt.ylabel("Number of neurons")
    plt.title("Distribution of max firing position across neurons")
    plt.show()


def plot_min_position(metrics):
    """Plot distribution of minimum firing positions."""
    min_position = metrics["min_position"]
    min_position = min_position[min_position > 0]
    
    plt.figure(figsize=(12, 6)) 
    plt.hist(min_position, bins=100)
    plt.xlabel("Min position of firing")
    plt.ylabel("Number of neurons")
    plt.title("Distribution of min firing position across neurons")
    plt.show()


def plot_avg_strike_length(metrics, min_length=2):
    """Plot distribution of average strike lengths."""
    avg_strike_length = metrics["avg_strike_length"]
    avg_strike_length = avg_strike_length[avg_strike_length > min_length]
    
    plt.figure(figsize=(12, 6)) 
    plt.hist(avg_strike_length, bins=100)
    plt.xlabel("Average strike length")
    plt.ylabel("Number of neurons")
    plt.title("Distribution of average strike length")
    plt.show()


def plot_avg_strikes_per_seq(metrics, min_strikes=1):
    """Plot distribution of average strikes per sequence."""
    avg_strikes_per_seq = metrics["avg_strikes_per_seq"]
    avg_strikes_per_seq = avg_strikes_per_seq[avg_strikes_per_seq > min_strikes]
    
    plt.figure(figsize=(12, 6)) 
    plt.hist(avg_strikes_per_seq, bins=100)
    plt.xlabel("Average strikes per sequence")
    plt.ylabel("Number of neurons")
    plt.title("Distribution of number of strikes")
    plt.show()


def get_neurons_in_first_n_positions(metrics, n=10):
    """Get neurons that fire only in the first n positions."""
    max_position = metrics["max_position"]
    g_zero = max_position > 0
    l_n = max_position < n
    filter_mask = g_zero & l_n
    return torch.where(filter_mask)[0], max_position[filter_mask]


def analyze_neuron_firing_patterns(data_path, num_neurons=15360, save_path=None):
    """
    Main function to analyze neuron firing patterns.
    
    Args:
        data_path: Path to the data file
        num_neurons: Number of neurons to analyze
        save_path: Optional path to save the computed metrics
    
    Returns:
        Dictionary of computed metrics
    """
    data = load_data(data_path)
    metrics = compute_neuron_metrics(data, num_neurons)
    
    if save_path:
        save_metrics(metrics, save_path)
        
    return metrics


def visualize_all_metrics(metrics):
    """Generate all visualization plots for the metrics."""
    plot_percentage_firings(metrics)
    plot_max_position(metrics)
    plot_min_position(metrics)
    plot_avg_strike_length(metrics)
    plot_avg_strikes_per_seq(metrics)


if __name__ == "__main__":
    data_path = "features_M0_D17.pkl"
    output_path = "neuron_metrics.pkl"  # Path to save metrics
    
    metrics = analyze_neuron_firing_patterns(data_path, save_path=output_path)
    visualize_all_metrics(metrics)
    
    # Example of getting neurons that fire only in first 10 positions
    early_neurons, early_positions = get_neurons_in_first_n_positions(metrics, 10)
    print(f"Found {len(early_neurons)} neurons that fire only in the first 10 positions")

