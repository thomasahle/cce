import pandas as pd
import torch
import math
from torch import nn
from torch.optim import Adam
from surprise import Dataset
from torch.utils.data import Dataset as TorchDataset, DataLoader

import cce


def make_embedding(vocab, num_params, dimension, method):
    n_chunks = 4
    if method != 'simple':
        chunk_dim = dimension // n_chunks
        assert n_chunks * chunk_dim == dimension, f"Dimension not divisible by {n_chunks}"
    if method == 'robe':
        log_size = int(math.log2(num_params))
        return cce.RobeEmbedding(
            size=num_params,
            chunk_size=chunk_dim,
            multi_hash=cce.MultiHash(num_hashes=n_chunks, output_bits=log_size),
        )
    if method == 'ce':
        rows = num_params // dimension
        return cce.CompositionalEmbedding(
            rows=rows,
            chunk_size=chunk_dim,
            hash=cce.MultiHash(num_hashes=n_chunks, output_bits=int(math.log2(rows))),
        )
    elif method == 'simple':
        num_embeddings = num_params // dimension
        return cce.SimpleEmbedding(
            num_embeddings, dimension,
            hash=cce.SingleHash(output_bits=int(math.log2(num_embeddings))),
        )
    elif method == 'cce':
        n_chunks = 4
        num_embeddings = num_params // dimension // n_chunks
        # TODO: The CCEmbedding should take a hash function so we don't have
        # to give the exact vocab size.
        return cce.CCEmbedding(
            vocab=vocab, rows=num_embeddings, chunk_size=dimension//n_chunks, n_chunks=n_chunks,
        )

# Load and process the data. We predict whether the user rated something >= 3.
data = Dataset.load_builtin('ml-100k')
df = pd.DataFrame(data.raw_ratings, columns=["user", "item", "rate", "id"])
df["rate"] = df["rate"].apply(lambda x: 1 if float(x) >= 3 else 0)
df["user"] = df["user"].astype("category").cat.codes.values
df["item"] = df["item"].astype("category").cat.codes.values

class RatingDataset(TorchDataset):
    def __init__(self, df): self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        return self.df.iloc[idx, 0], self.df.iloc[idx, 1], self.df.iloc[idx, 2]

class RecommenderNet(nn.Module):
    def __init__(self, n_users, n_items, num_params, dim, method):
        super().__init__()
        self.method = method
        self.user_embedding = make_embedding(n_users, num_params, dim, method)
        self.item_embedding = make_embedding(n_items, num_params, dim, method)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        return self.mlp(user_emb * item_emb).view(-1)

# Instantiate the model and define the loss function and optimizer
#num_params = 2**26
num_params = 300 * 64
train_loader = DataLoader(RatingDataset(df), batch_size=64)
n_users = df["user"].nunique()
n_items = df["item"].nunique()
print(f'Unique users: {n_users}, Unique items: {n_items}, #params: {num_params}')
model = RecommenderNet(n_users, n_items, num_params, dim=64, method='cce')
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    model.train()
    total_loss = 0
    for user, item, label in train_loader:
        optimizer.zero_grad()
        prediction = model(user.long(), item.long())
        loss = criterion(prediction, label.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch: ", epoch, "Loss: ", total_loss/len(train_loader))
    if model.method == 'cce':
        model.user_embedding.cluster(verbose=False)
        model.item_embedding.cluster(verbose=False)

