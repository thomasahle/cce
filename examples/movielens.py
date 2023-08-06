import pandas as pd
import torch
import math
from torch import nn
from surprise import Dataset
from torch.utils.data import Dataset as TorchDataset, DataLoader
import argparse

import cce
torch.manual_seed(0xcce)


def make_embedding(vocab, num_params, dimension, method):
    n_chunks = 4
    chunk_dim = dimension // n_chunks
    assert n_chunks * chunk_dim == dimension, f"Dimension not divisible by {n_chunks}"

    if method == 'robe':
        log_size = int(math.log2(num_params))
        return cce.RobeEmbedding(
            size=num_params,
            chunk_size=chunk_dim,
            hash=cce.MultiHash(num_hashes=n_chunks, output_bits=log_size),
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
        # Divide by two, since the CCE embedding will use two tables with each `row` rows.
        rows = num_params // dimension // 2
        return cce.CCEmbedding(
            vocab=vocab, rows=rows, chunk_size=dimension//n_chunks, n_chunks=n_chunks,
        )
    elif method == 'full':
        emb = nn.Embedding(vocab, dimension)
        nn.init.uniform_(emb.weight, -(dimension**-0.5), dimension**-0.5)
        return emb

class RatingDataset(TorchDataset):
    def __init__(self, df): self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        return self.df.iloc[idx, 0], self.df.iloc[idx, 1], self.df.iloc[idx, 2]

class RecommenderNet(nn.Module):
    """ A simple DLRM style model """
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
        bs, dim = user_emb.shape
        return self.mlp(user_emb * item_emb).view(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--last-cluster', type=int, default=7, help='Stop reclusering after this many epochs.')
    parser.add_argument('--dim', type=int, default=32, help='Dimension of embeddings')
    args = parser.parse_args()

    # Load and process the data. We predict whether the user rated something >= 3.
    data = Dataset.load_builtin('ml-100k')
    df = pd.DataFrame(data.raw_ratings, columns=["user", "item", "rate", "id"])
    df["rate"] = df["rate"].apply(lambda x: 1 if float(x) >= 3 else 0)
    df["user"] = df["user"].astype("category").cat.codes.values
    df["item"] = df["item"].astype("category").cat.codes.values

    # Instantiate the model and define the loss function and optimizer
    dim = args.dim
    num_params = 200 * dim
    n_users = df["user"].nunique()
    n_items = df["item"].nunique()
    print(f'Unique users: {n_users}, Unique items: {n_items}, #params: {num_params}')
    model = RecommenderNet(n_users, n_items, num_params, dim=dim, method=args.method)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters())

    # Create DataLoader
    train = df.sample(frac=0.8, random_state=0)
    valid = df.drop(train.index)
    train_tensor = RatingDataset(train)
    valid_tensor = RatingDataset(valid)
    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_tensor, batch_size=64, shuffle=False)

    # Train the model
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for user, item, label in train_loader:
            optimizer.zero_grad()
            prediction = model(user.long(), item.long())
            loss = criterion(prediction, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss/len(train_loader)

        # Validate the model
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for user, item, label in valid_loader:
                user, item, label = user.long(), item.long(), label.float()
                prediction = model(user, item)
                loss = criterion(prediction, label)
                total_loss += loss.item()
            valid_loss = total_loss/len(valid_loader)

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.3}, Validation Loss: {valid_loss:.3}")

        if model.method == 'cce' and epoch < args.last_cluster:
            print('Clustering...')
            model.user_embedding.cluster(verbose=False)
            model.item_embedding.cluster(verbose=False)

if __name__ == '__main__':
    main()
