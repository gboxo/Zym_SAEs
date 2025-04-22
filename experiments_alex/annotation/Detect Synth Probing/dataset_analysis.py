# %%
import json
from src.utils import get_paths
# %%

paths = get_paths()
brenda_path = paths.mini_brenda
with open(brenda_path, "r") as f:
    data = f.read()


data = data.split("\n")
data = [seq.strip("<pad>") for seq in data]
data = [elem for seq in data for elem in seq.split("<|endoftext|>")]
ec_label = [seq.split("<sep>")[0] for seq in data if len(seq.split("<sep>")) > 1]
sequences = [seq.split("<sep>")[1].strip("<sep>").strip("<start>").strip("<end>") for seq in data if len(seq.split("<sep>")) > 1]
sequences = [seq.strip("<|endoftext|").strip("<pad>") for seq in sequences]
dict_data = {}
for id,seq in zip(ec_label,sequences):
    if "-" in id or len(id) == 0:
        continue
    if id not in dict_data:
        dict_data[id] = []
    dict_data[id].append(seq)

filter_data = {}
for id,seq in dict_data.items():
    if len(seq) > 100:
        filter_data[id] = seq[:100]


# %%
with open("Data/Detect_Synth_Data/natural_sequences_by_ec.json", "w") as f:
    json.dump(filter_data, f)










