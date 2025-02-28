# %%
import torch
import torch.utils.data
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, vstack
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# %%
# Convert to torch tensors


# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def convert_to_sparse_coo_tensor(sparse_coo_np):

    indices = np.vstack((sparse_coo_np.row, sparse_coo_np.col))  # COO indices
    values = sparse_coo_np.data                                  # COO values
    shape = sparse_coo_np.shape                                  # Tensor shape

    sparse_coo_torch = torch.sparse_coo_tensor(
        indices=torch.tensor(indices, dtype=torch.long),
        values=torch.tensor(values, dtype=torch.float),
        size=shape
    )
    return sparse_coo_torch



def model_train(dataloader):


# Initialize model, loss, and optimizer
    model = LogisticRegression(input_dim=10240).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # L1 regularization strength
    lambda_l1 = 0.1

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_x, batch_y in tqdm(dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y.float())

            # Add L1 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + lambda_l1 * l1_norm

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training complete!")
    return model


def sparse_collate_fn(batch):
    """Custom collate function for sparse tensors"""
    x_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]
    
    # For sparse tensors, we need to concatenate them differently
    # Convert sparse tensors to dense for batching
    x_dense = torch.stack([x.to_dense() for x in x_list])
    y_tensor = torch.stack(y_list)
    
    return x_dense, y_tensor


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    natural_train = np.load("features/natural_features_train.npz", allow_pickle=True)["arr_0"].tolist()
    natural_test = np.load("features/natural_features_test.npy", allow_pickle=True).tolist()
    synth_train = np.load("features/synth_features_train.npz", allow_pickle=True)["arr_0"].tolist()
    synth_test = np.load("features/synth_features_test.npy", allow_pickle=True).tolist()

    X_train = vstack((natural_train , synth_train))
    X_test = vstack((vstack(natural_test) , vstack(synth_test)))

    y_train = np.concatenate((np.zeros(natural_train.shape[0]), np.ones(synth_train.shape[0])), axis=0)
    y_test = np.concatenate((np.zeros(vstack(natural_test).shape[0]), np.ones(vstack(synth_test).shape[0])), axis=0)


    X_train_sparse = convert_to_sparse_coo_tensor(X_train)
    X_test_sparse = convert_to_sparse_coo_tensor(X_test)
    X_train_sparse = X_train_sparse.to(device)
    X_test_sparse = X_test_sparse.to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_sparse, y_train), 
        batch_size=512, 
        shuffle=True,
        collate_fn=sparse_collate_fn
    )

    model = model_train(dataloader)



    
    
    
    
    


