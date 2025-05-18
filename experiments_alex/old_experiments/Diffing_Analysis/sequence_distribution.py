import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
GET_EMBEDDINGS = False
if GET_EMBEDDINGS:
    path = "/users/nferruz/gboxo/Alpha Amylase/seq_gens/"

    files = os.listdir(path)
    files = [i for i in files if i.endswith(".fasta")]
    all_dfs = {} 
    for file in files:
        with open(os.path.join(path, file), "r") as f:
            data = f.read()
            data = data.split(">")
            data = [i for i in data if i != ""]
            ids = [i.split("\n")[0].split("\t")[0] for i in data]
            seqs = [i.split("\n")[1] for i in data]
            df = pd.DataFrame({"id": ids, "seq": seqs})
            all_dfs[file] = df


    checkpoint = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
        output_hidden_states=True
    )
    model.to("cuda")
    model.eval()

    embeddings = {}
    for file in files:
        df = all_dfs[file]
        seqs = df["seq"].tolist()
        all_embeddings = []
        for seq in tqdm(seqs):
            tokens = tokenizer(seq, return_tensors="pt", padding=True).to("cuda")
            embedding = model(tokens["input_ids"])["hidden_states"][0][-1]
            all_embeddings.append(embedding.to("cpu"))
        embeddings[file] = all_embeddings
        torch.cuda.empty_cache()
        del all_embeddings

    with open("Data/Diffing_Analysis_Data/embeddings.pkl", "wb") as f:
        pkl.dump(embeddings, f)


with open("Data/Diffing_Analysis_Data/embeddings.pkl", "rb") as f:
    embeddings = pkl.load(f)

# Get all the embeddings and calculate centroids
all_embeddings = []
centroids = {}  # Store centroids for each file
for file in embeddings.keys():
    file_embeddings = [emb.mean(0).reshape(1, -1) for emb in embeddings[file]]
    centroid = torch.stack(file_embeddings).mean(0).detach().cpu().numpy()  # Calculate centroid for this file
    centroids[file] = centroid
    all_embeddings.extend(file_embeddings)

all_embeddings = torch.cat(all_embeddings, dim=0).detach().cpu().numpy()


# Dimensionality reduction TSNE
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# First, standardize the embeddings
scaler = StandardScaler()
all_embeddings_scaled = scaler.fit_transform(all_embeddings)

# Then apply t-SNE with some adjusted parameters
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=10,  # Adjust this based on your dataset size
    n_iter=1000,
    init='pca'
)
all_embeddings_2d = tsne.fit_transform(all_embeddings_scaled)


# KDE plot of the 2D embeddings
# Gaussian KDE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Create a grid of points to evaluate the KDE
x_min, x_max = all_embeddings_2d[:, 0].min() - 1, all_embeddings_2d[:, 0].max() + 1
y_min, y_max = all_embeddings_2d[:, 1].min() - 1, all_embeddings_2d[:, 1].max() + 1
x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
positions = np.vstack([x.ravel(), y.ravel()])

# Compute kernel density estimate
values = np.vstack([all_embeddings_2d[:, 0], all_embeddings_2d[:, 1]])
kernel = gaussian_kde(values)
z = np.reshape(kernel(positions).T, x.shape)

# Plot the KDE with centroids
plt.figure(figsize=(10, 8))
plt.imshow(z.T, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
plt.colorbar(label='Density')

# Transform and plot centroids
centroids_array = np.stack(list(centroids.values()))[:,0]
centroids_scaled = scaler.transform(centroids_array)
centroids_2d = tsne.fit_transform(centroids_scaled)

# Plot centroids with different markers and labels
for idx, (file_name, _) in enumerate(centroids.items()):
    plt.scatter(centroids_2d[idx, 0], centroids_2d[idx, 1], 
               c='red', marker='*', s=20, label=f'Centroid {file_name}')

plt.xlabel('TSNE-1')
plt.ylabel('TSNE-2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("Data/Diffing_Analysis_Data/tsne_with_centroids.png", dpi=300, bbox_inches='tight')
plt.close()
