import pandas as pd
import pickle
import numpy as np



def map_seq_to_msa(original_seq, msa_seq):
    """
    Maps positions from the original sequence to the MSA sequence.
    
    Args:
        original_seq (str): The original sequence without gaps
        msa_seq (str): The MSA sequence with gaps (represented by '-')
        
    Returns:
        list: A mapping array where each index corresponds to an MSA position.
              The value is the corresponding position in the original sequence,
              or -1 if the MSA position is a gap.
    """
    mapping = []
    orig_pos = 0
    
    for msa_char in msa_seq:
        if msa_char == '-':
            mapping.append(-1)
        else:
            # Verify the characters match (sanity check)
            if orig_pos < len(original_seq) and original_seq[orig_pos] != msa_char:
                print(f"Warning: Character mismatch at position {orig_pos}. "
                      f"Original: {original_seq[orig_pos]}, MSA: {msa_char}")
            mapping.append(orig_pos)
            orig_pos += 1
    
    return mapping

def map_positions_to_msa(positions, original_seq, msa_seq):
    """
    Maps an array of positions from the original sequence to MSA positions.
    
    Args:
        positions (list): List of positions in the original sequence
        original_seq (str): The original sequence without gaps
        msa_seq (str): The MSA sequence with gaps
        
    Returns:
        list: Positions mapped to the MSA coordinate system, with -1 for unmapped positions
    """
    # Create the mapping from original sequence to MSA
    seq_to_msa_map = {}
    orig_pos = 0
    
    for msa_pos, msa_char in enumerate(msa_seq):
        if msa_char != '-':
            seq_to_msa_map[orig_pos] = msa_pos
            orig_pos += 1
    
    # Map each position using the mapping
    msa_positions = []
    for pos in positions:
        msa_positions.append(seq_to_msa_map.get(pos, -1))
    
    return msa_positions


def load_msa(path):
    msa_seqs = {} 
    with open(path, "r") as f:
        current_seq = ""
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq and current_id:
                    msa_seqs[current_id] = current_seq
                current_id = line[1:].split(",")[0]
                current_seq = ""
            else:
                current_seq += line
        if current_seq and current_id:
            msa_seqs[current_id] = current_seq
    return msa_seqs



if __name__ == "__main__":

    features_path = "/home/woody/b114cb/b114cb23/boxo/msa_steering/latent_scoring_base/features/features.pkl"
    df = pd.read_csv("/home/woody/b114cb/b114cb23/boxo/msa_steering/activity_predictions_no_penalty.csv")
    msa_path = "/home/woody/b114cb/b114cb23/boxo/msa_steering/msa_ft.fasta"
    msa_seqs = load_msa(msa_path)

    with open(features_path, "rb") as f:
        features = pickle.load(f)
    print(features["3.2.1.1_48_12_ZC_FT"].todense().shape)

    # --- Collect and reindex features based on MSA columns ---
    reindexed_features_sparse = {}


    
    


    for key in msa_seqs.keys():
        if key not in features.keys():
            print(f"Warning: Key {key} not found in features")
            continue
        if key not in df["index"].tolist():
            print(f"Warning: Key {key} not found in CSV sequences")
            continue

        seq = df[df["index"] == key]["sequence"].values[0]
        msa_seq = msa_seqs[key]
        mapping = map_seq_to_msa(seq, msa_seq)
        feature_matrix = features[key]  # keep as sparse
        L, D = feature_matrix.shape

        # Convert to CSR for efficient row slicing
        feature_matrix = feature_matrix.tocsr()

        # Store as {msa_pos: feature}
        msa_features_dict = {}
        for msa_pos, seq_pos in enumerate(mapping):
            if seq_pos == -1:
                msa_features_dict[msa_pos] = None  # or a sparse zero vector if you prefer
            else:
                feature_idx = seq_pos + 9
                if feature_idx < L:
                    msa_features_dict[msa_pos] = feature_matrix[feature_idx]
                else:
                    print(f"Warning: feature_idx {feature_idx} out of bounds for key {key}")
                    msa_features_dict[msa_pos] = None
        reindexed_features_sparse[key] = msa_features_dict

    # Save the sparse reindexed features
    with open("/home/woody/b114cb/b114cb23/boxo/msa_steering/reindexed_features_by_msa_sparse.pkl", "wb") as f:
        pickle.dump(reindexed_features_sparse, f)


    


