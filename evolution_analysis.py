# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = "/users/nferruz/gboxo/diffing_sapi_multi_iterations_from_DMS/joined_dataframes"
output_path = "/users/nferruz/gboxo/diffing_sapi_multi_iterations_from_DMS/alignments/"



# %%
files = os.listdir(path)


# %%
all_data = []
for file in files:
    it = file.split("_")[2].split("iteration")[1].split(".")[0]

    df = pd.read_csv(os.path.join(path, file))
    df["iteration"] = int(it)
    all_data.append(df)

all_data = pd.concat(all_data)

# %%

plt.figure(figsize=(10, 6))
sns.violinplot(data=all_data, x="iteration", y="alntmscore")
plt.xlabel("Iteration")
plt.ylabel("AlnTMscore")
plt.title("Distribution of AlnTMscore Across Iterations")
plt.show()

# %%
print(all_data.shape)

"""
 FILTERING STAGE
 1) We filter out the structures with alnTMscore < 0.5
 2) We filter out sequences with length > 600 or < 400
"""

all_data = all_data[all_data["alntmscore"] > 0.5]
all_data = all_data[all_data["sequence"].apply(len) > 400]
all_data = all_data[all_data["sequence"].apply(len) < 600]



# %%


from Bio import SeqIO, Seq
from Bio.SeqRecord import SeqRecord

records = []

for i in range(6):
    df_iter = all_data[all_data["iteration"] == i]
    for _, row in df_iter.iterrows():
        label = f"iter{i}_{row['label']}"
        seq = row["sequence"]
        records.append(SeqRecord(Seq.Seq(seq), id=label, description=""))

# Save to file
SeqIO.write(records, f"{output_path}/all_sequences.fasta", "fasta")




print(all_data.shape)
# %%
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# Function to save sequences by iteration
def save_fasta_by_iteration(df, base_filename="iteration"):
    for i in df["iteration"].unique():
        df_iter = df[df["iteration"] == i]
        print(df_iter)
        records = [
            SeqRecord(Seq(row["sequence"]), id=str(idx), description=f"iteration{i}")
            for idx, row in df_iter.iterrows()
        ]
        SeqIO.write(records, f"{path}/{base_filename}{i}.fasta", "fasta")

# Apply
save_fasta_by_iteration(all_data)

# %%



alignment_path = "/users/nferruz/gboxo/diffing_sapi_multi_iterations_from_DMS/alignments"
import pandas as pd

df_align = pd.read_csv(os.path.join(alignment_path, "iter0_vs_1.m8"), sep="\t", header=None)
df_align.columns = ["query", "target", "identity", "aln_len", "mismatches", "gaps",
                    "qstart", "qend", "tstart", "tend", "evalue", "bit_score"]


# %%

from Bio import AlignIO
import numpy as np
import pandas as pd

alignment = AlignIO.read(os.path.join(output_path, "all_aligned.fasta"), "fasta")
ref_seq = str(alignment[0].seq)  # use first sequence as reference
n_positions = len(ref_seq)
mut_matrix = np.zeros(n_positions)

for record in alignment[1:]:
    for i in range(n_positions):
        if record.seq[i] != ref_seq[i] and ref_seq[i] != '-' and record.seq[i] != '-':
            mut_matrix[i] += 1

# Normalize by number of sequences
mut_matrix /= len(alignment) - 1

# Convert to DataFrame for plotting
df_mut = pd.DataFrame({
    "position": np.arange(1, n_positions + 1),
    "mutation_frequency": mut_matrix
})

# %%
# Save to CSV
df_mut.to_csv(os.path.join(output_path, "position_mutation_matrix.csv"), index=False)


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 4))
plt.plot(df_mut["position"], df_mut["mutation_frequency"], lw=1)
plt.xlabel("Residue Position (MSA)")
plt.ylabel("Mutation Frequency")
plt.title("Position-wise Mutation Frequency Across Iterations")
plt.tight_layout()
plt.show()


# %%
# Define a threshold: e.g., top 5% mutation frequency
threshold = df_mut["mutation_frequency"].quantile(0.99)

plt.figure(figsize=(16, 4))
plt.plot(df_mut["position"], df_mut["mutation_frequency"], label="Mutation frequency")
plt.axhline(threshold, color="red", linestyle="--", label=f"99th percentile = {threshold:.2f}")
plt.fill_between(df_mut["position"], df_mut["mutation_frequency"], threshold, 
                 where=(df_mut["mutation_frequency"] > threshold), color="red", alpha=0.2)
plt.xlabel("Residue Position (MSA-aligned)")
plt.ylabel("Mutation Frequency")
plt.legend()
plt.title("Hotspot Identification from Mutation Matrix")
plt.tight_layout()
plt.show()




# %%
#!/usr/bin/env python3
from Bio import AlignIO
from collections import Counter
import numpy as np

# 1) Load the alignment
alignment = AlignIO.read(os.path.join("/users/nferruz/gboxo/diffing_sapi_multi_iterations_from_DMS/alignments", "all_aligned.fasta"), "fasta")

# 2) Group sequences by iteration
sequences_by_iter = {}
for rec in alignment:
    # Assumes record.id ends with something like "iteration0", "iteration1", ...
    iter_label = rec.id.rsplit("_", 1)[-1]
    sequences_by_iter.setdefault(iter_label, []).append(str(rec.seq))

# 3) Select the two iterations you want to compare:
iter_a = "iteration1"
iter_b = "iteration2"

seqs_a = sequences_by_iter.get(iter_a)
seqs_b = sequences_by_iter.get(iter_b)
if seqs_a is None or seqs_b is None:
    raise KeyError(f"One of {iter_a}, {iter_b} not found in alignment IDs: {list(sequences_by_iter)}")

# 4) Turn each set into a (N × L) array of characters
arr_a = np.array([list(s) for s in seqs_a])
arr_b = np.array([list(s) for s in seqs_b])

# 5) Compute consensus + conservation fraction per column
def consensus_and_freq(arr):
    cons = []
    freq = []
    for col in arr.T:
        counts = Counter(col)
        counts.pop('-', None)        # ignore gaps
        if counts:
            aa, count = counts.most_common(1)[0]
            cons.append(aa)
            freq.append(count / len(col))
        else:
            cons.append('-')
            freq.append(0.0)
    return np.array(cons), np.array(freq)

cons_a, freq_a = consensus_and_freq(arr_a)
cons_b, freq_b = consensus_and_freq(arr_b)

# 6) Find and report all positions where consensus differs
diff_positions = np.where(cons_a != cons_b)[0]
print(f"Found {len(diff_positions)} differing positions between {iter_a} and {iter_b}:\n")
for pos in diff_positions:
    print(f"  Alignment pos {pos+1:4d}: "
          f"{iter_a} = {cons_a[pos]!r} ({freq_a[pos]*100:5.1f}% conserved), "
          f"{iter_b} = {cons_b[pos]!r} ({freq_b[pos]*100:5.1f}% conserved)")



# %%
from Bio import AlignIO
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# User parameters: change if needed
iter_a = "iteration0"
iter_b = "iteration5"
alignment_file = "all_aligned.fasta"
output_path = "/users/nferruz/gboxo/diffing_sapi_multi_iterations_from_DMS/alignments2/"

# Load the MAFFT alignment
alignment = AlignIO.read(os.path.join(output_path, "all_aligned.fasta"), "fasta")


# Group sequences by iteration
seqs_by_iter = {}
for rec in alignment:
    iter_label = rec.id.rsplit("_", 1)[-1]
    seqs_by_iter.setdefault(iter_label, []).append(str(rec.seq))


seqs_a = seqs_by_iter.get(iter_a)
seqs_b = seqs_by_iter.get(iter_b)
if seqs_a is None or seqs_b is None:
    raise ValueError(f"Iterations '{iter_a}' or '{iter_b}' not found. Available: {list(seqs_by_iter.keys())}")

# Convert to numpy arrays
arr_a = np.array([list(s) for s in seqs_a])
arr_b = np.array([list(s) for s in seqs_b])

# Compute consensus residues and conservation frequencies
def consensus_and_freq(arr):
    cons = []
    freq = []
    for col in arr.T:
        counts = Counter(col)
        counts.pop('-', None)
        if counts:
            aa, count = counts.most_common(1)[0]
            cons.append(aa)
            freq.append(count / len(col))
        else:
            cons.append('-')
            freq.append(0.0)
    return np.array(cons), np.array(freq)

cons_a, freq_a = consensus_and_freq(arr_a)
cons_b, freq_b = consensus_and_freq(arr_b)

# Identify positions where consensus differs
diff_positions = np.where(cons_a != cons_b)[0] + 1  # 1-based positions

# Plot 1: Conservation frequency for each iteration
plt.figure()
plt.plot(freq_a, label=f"{iter_a} conservation")
plt.plot(freq_b, label=f"{iter_b} conservation")
plt.xlabel("Alignment position")
plt.ylabel("Conservation fraction")
plt.title("Sequence Conservation per Position")
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Highlight consensus changes
plt.figure()
plt.vlines(diff_positions, ymin=0, ymax=1)
plt.xlabel("Alignment position")
plt.ylabel("Consensus change indicator")
plt.title("Positions with Consensus Changes")
plt.tight_layout()
plt.show()

# %%

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# PARAMETERS: adjust these if your iteration tags differ
iter_a = "iteration1"
iter_b = "iteration28"
alignment_file = os.path.join(output_path, "all_aligned.fasta")

# 1) Simple FASTA parser
def parse_fasta(path):
    ids, seqs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                ids.append(line[1:])
                seqs.append("")
            else:
                seqs[-1] += line
    return ids, seqs

ids, seqs = parse_fasta(alignment_file)

# 2) Group by iteration
groups = {}
for rid, seq in zip(ids, seqs):
    tag = rid.rsplit("_",1)[-1]
    groups.setdefault(tag, []).append(seq)

seqs_a = groups[iter_a]
seqs_b = groups[iter_b]

# 3) Consensus+freq function
def consensus_and_freq(seqs):
    arr = np.array([list(s) for s in seqs])
    cons, freq = [], []
    for col in arr.T:
        c = Counter(col)
        c.pop("-", None)
        if c:
            aa, ct = c.most_common(1)[0]
            cons.append(aa)
            freq.append(ct/len(col))
        else:
            cons.append("-")
            freq.append(0.0)
    return np.array(cons), np.array(freq)

cons_a, _ = consensus_and_freq(seqs_a)
cons_b, _ = consensus_and_freq(seqs_b)

# 4) Gather substitution counts
subs = Counter(zip(cons_a, cons_b))
aas = list("ACDEFGHIKLMNPQRSTVWY")  # standard AA order
mat = np.zeros((20,20), int)
for i, aa1 in enumerate(aas):
    for j, aa2 in enumerate(aas):
        mat[i,j] = subs.get((aa1, aa2), 0)

# 5) Plot heatmap
plt.figure(figsize=(8,6))
plt.imshow(mat, aspect='equal')
plt.xticks(range(20), aas, rotation=90)
plt.yticks(range(20), aas)
plt.xlabel(f"{iter_b} consensus")
plt.ylabel(f"{iter_a} consensus")
plt.title("Consensus AA substitutions: iteration1 → iteration2")
plt.colorbar(label="Count of positions")
plt.tight_layout()
plt.show()

# %%


# %%


