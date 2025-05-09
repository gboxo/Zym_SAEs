import pandas as pd
import numpy as np
import os



"""

SUMMARY STATISTICS of the sequences at AA level


Metrics to track:
- Length of the sequences
- Amino acid composition
- Aggergate by:
    - Hydrophobicity
    - Polar uncharged
    - Positively charged
    - Negatively charged
- GRAVY Score
- Aliphatic index
"""

hydropathy_index = {
    'I': 4.5,
    'V': 4.2,
    'L': 3.8,
    'F': 2.8,
    'C': 2.5,
    'M': 1.9,
    'A': 1.8,
    'G': -0.4,
    'T': -0.7,
    'W': -0.9,
    'S': -0.8,
    'Y': -1.3,
    'P': -1.6,
    'H': -3.2,
    'E': -3.5,
    'Q': -3.5,
    'D': -3.5,
    'N': -3.5,
    'K': -3.9,
    'R': -4.5
}





class SummaryStatistics:
    def __init__(self, sequences_dict):
        # Store the dictionary {ID: sequence}
        self.sequences_dict = sequences_dict
        # Create a Series with IDs as index for convenience in other methods
        self._sequence_series = pd.Series(sequences_dict)

    def get_length(self):
        # Return the number of sequences
        return len(self.sequences_dict)

    def get_amino_acid_composition(self):
        """
        Calculates the amino acid composition for each sequence individually.

        Returns:
            pd.Series: A Series indexed by sequence ID, where each value is a
                       dictionary mapping amino acids to their counts for that sequence.
        """
        def calculate_composition(sequence):
            # Use pd.Series().value_counts() to get counts, then convert to dict
            if not sequence: # Handle empty sequences
                return {}
            return pd.Series(list(sequence)).value_counts().to_dict()

        # Apply the function to the Series (already indexed by ID)
        return self._sequence_series.apply(calculate_composition)

    def get_hydrophobicity(self):
        # Apply the function to the Series (already indexed by ID)
        # Use .get for safety, defaulting to 0 if AA not in hydropathy_index
        return self._sequence_series.apply(lambda x: sum(hydropathy_index.get(aa, 0) for aa in x if x)) # Check if x is not empty

    def get_gravy_score(self):
        # Apply the function to the Series (already indexed by ID)
        # Note: This seems identical to get_hydrophobicity based on current implementation
        # Maybe the calculation should be sum / length? Assuming sum for now.
        # Use .get for safety, defaulting to 0 if AA not in hydropathy_index
        return self._sequence_series.apply(lambda x: sum(hydropathy_index.get(aa, 0) for aa in x if x)) # Check if x is not empty

    def get_aliphatic_index(self):
        """
        Calculates the aliphatic index for each sequence.

        The aliphatic index is defined as: X(Ala) + a*X(Val) + b*X(Leu) + b*X(Ile)
        where X is the mole fraction and a=2.9, b=3.9.
        """
        a = 2.9
        b = 3.9

        def calculate_index(sequence):
            length = len(sequence)
            if length == 0:
                return 0.0 # Return 0 for empty sequences

            # Count occurrences of relevant amino acids
            count_ala = sequence.count('A')
            count_val = sequence.count('V')
            count_leu = sequence.count('L')
            count_ile = sequence.count('I')

            # Calculate mole fractions (X)
            x_ala = count_ala / length
            x_val = count_val / length
            x_leu = count_leu / length
            x_ile = count_ile / length

            # Apply the formula
            aliphatic_index = x_ala + a * x_val + b * (x_leu + x_ile)
            return aliphatic_index

        # Apply the calculation to the Series (already indexed by ID)
        return self._sequence_series.apply(calculate_index)
    
    
    

def read_sequence_from_file(path):
    """
    Reads **all** sequences in a .txt (multiple FASTA‐style entries).
    Each header line is of the form:
      >ID,… <sep> <start> A B C … <end>

    Returns:
      A dictionary mapping sequence IDs to sequence strings (with spaces removed)
    """
    seqs = {}
    curr = ""
    curr_id = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # If we were collecting a sequence, save it before starting a new one
                if curr and curr_id:
                    seqs[curr_id] = curr
                    curr = ""
                # Get the ID from the header line
                curr_id = line[1:].split(",")[0]
                # If this header has the entire seq on the same line:
                if "<start>" in line and "<end>" in line:
                    block = line.split("<start>", 1)[1].split("<end>", 1)[0]
                    curr = block.replace(" ", "")
                continue

            # Continue accumulating sequence tokens
            curr += line.replace(" ", "")

    # At EOF, flush last sequence
    if curr and curr_id:
        seqs[curr_id] = curr

    return seqs

if __name__ == "__main__":
    path = "/home/woody/b114cb/b114cb23/boxo/strong_steering/seq_gen/seq_gen_3.2.1.1_ZC_FT.fasta"
    output_dir = "/home/woody/b114cb/b114cb23/boxo/strong_steering/summary_statistics/base"
    os.makedirs(output_dir, exist_ok=True)
    sequences = read_sequence_from_file(path)
    # Pass the dictionary directly
    summary_stats = SummaryStatistics(sequences)

    df = pd.DataFrame(summary_stats.get_hydrophobicity(), columns=["hydrophobicity"])
    df["length"] = summary_stats.get_length()
    df["aliphatic_index"] = summary_stats.get_aliphatic_index()
    df["gravy_score"] = summary_stats.get_gravy_score()
    df["amino_acid_composition"] = summary_stats.get_amino_acid_composition()
    print(df.head())
    df.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
