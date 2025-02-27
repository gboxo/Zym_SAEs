# %%
from inference_batch_topk import convert_to_jumprelu
from utils import load_sae, load_model, get_ht_model
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import scipy.cluster.hierarchy as hierarchy

# %%
tokenizer, model = load_model("AI4PD/ZymCTRL")
model = get_ht_model(model, model.config).to("cuda")
sae_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000/"
cfg, sae = load_sae(sae_path)
thresholds = torch.load(sae_path+"/percentiles/feature_percentile_99.pt")
thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
sae.to("cuda")
jump_relu = convert_to_jumprelu(sae, thresholds)
jump_relu.eval()
del sae

# %%

with open("micro_brenda.txt", "r") as f:
    test_set = f.read()
test_set = test_set.split("\n")
test_set = [seq.strip("<pad>") for seq in test_set]
test_set = [elem for seq in test_set for elem in seq.split("<|endoftext|>")]
ec_numbers = [elem.split("<sep>")[0] for elem in test_set if len(elem.split("<sep>")) > 1]
ec_numbers = [elem for elem in ec_numbers if len(elem) > 0]

ec_numbers  = list(set(ec_numbers))

# %%

all_acts = []
for ec_number in ec_numbers:
    with torch.no_grad():
        tokenized_ec_number = tokenizer.encode(ec_number, padding=False, truncation=True, return_tensors="pt", max_length=256)
        dot_pos = tokenized_ec_number == torch.tensor(431)
        names_filter = lambda x: x in "blocks.26.hook_resid_pre"
        logits, cache = model.run_with_cache(tokenized_ec_number.to("cuda"), names_filter=names_filter)
        acts = cache["blocks.26.hook_resid_pre"]
        all_acts.append(acts)



# Import utils for sparse tensors
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
all_sparse_tensors = []
for acts in all_acts:
    feature_acts = jump_relu.forward(acts, use_pre_enc_bias=True)["feature_acts"]
    feature_acts = feature_acts.sum(dim=1)
    all_sparse_tensors.append(coo_matrix(feature_acts[0].detach().cpu().numpy()))



# %%
all_features = [act.todense() for act in all_sparse_tensors]
features_agg = np.array(all_features)[:,0].sum(axis=0)
top_features = np.argsort(features_agg)[-100:]
indices = np.where(features_agg > 0)[0]

# Create a feature expression matrix and display the most important features






# %%

indices_sort = sorted(ec_numbers, key = lambda x: (x.split(".")))
indices_sort = [ec_numbers.index(elem) for elem in indices_sort]
sorted_ec_numbers = [ec_numbers[i] for i in indices_sort]

# %%

x = np.array(all_features)[:,0]

x_filtered = x[:, top_features]
x_filtered = x_filtered[indices_sort]
import pandas as pd
df = pd.DataFrame(x_filtered)
df.columns = top_features
df.index = sorted_ec_numbers
sns.heatmap(df, cmap="RdBu_r")
plt.show()


# %%


sns.clustermap(df, cmap="RdBu_r", row_cluster=False)
plt.show()

# %%





