import pandas as pd
import torch
import math
from torch import nn
import numpy as np
# from surprise import Dataset
from torch.utils.data import Dataset as TorchDataset, DataLoader
import argparse
import time
from sklearn.metrics import roc_auc_score
import tqdm

import data

import cce


methods = ['robe', 'ce', 'simple', 'cce', 'full', 'tt', 'cce_robe', 'dhe', 'hash', 'hnet']

def make_embedding(vocab, num_params, dimension, method):
    n_chunks = 4
    chunk_dim = dimension // n_chunks
    assert n_chunks * chunk_dim == dimension, f"Dimension not divisible by {n_chunks}"

    if method == 'robe':
        hash = cce.PolyHash(num_hashes=n_chunks, output_range=num_params)
        return cce.RobeEmbedding(
            size=num_params,
            chunk_size=chunk_dim,
            hash=hash
        )
    if method == 'ce':
        rows = num_params // dimension
        hash = cce.PolyHash(num_hashes=n_chunks, output_range=rows)
        return cce.CompositionalEmbedding(rows=rows, chunk_size=chunk_dim, hash=hash)
    elif method == 'hash':
        rows = num_params // dimension
        # TODO: When we index multiple (k) pointers into the same space, do we need to use
        # k-independent hashing instead of 2 independent hashing?
        hash = cce.PolyHash(num_hashes=n_chunks, output_range=rows)
        return cce.HashEmbedding(rows, dimension, hash)
    elif method == 'hnet':
        hash = cce.PolyHash(num_hashes=dimension, output_range=num_params)
        return cce.HashNetEmbedding(num_params, hash)
    elif method == 'simple':
        rows = num_params // dimension
        hash = cce.PolyHash(num_hashes=1, output_range=rows)
        return cce.CompositionalEmbedding(rows=rows, chunk_size=dimension, hash=hash)
    elif method == 'cce':
        # Divide by two, since the CCE embedding will use two tables with each `row` rows.
        rows = num_params // dimension // 2
        return cce.CCEmbedding(
            vocab=vocab, rows=rows, chunk_size=dimension//n_chunks, n_chunks=n_chunks,
        )
    elif method == 'cce_robe':
        size = num_params // 2
        return cce.CCERobembedding(
            vocab=vocab, size=size, chunk_size=dimension//n_chunks, n_chunks=n_chunks,
        )
    elif method == 'full':
        emb = nn.Embedding(vocab, dimension)
        nn.init.uniform_(emb.weight, -(dimension**-0.5), dimension**-0.5)
        return emb
    elif method == 'tt':
        # For TT we use a QR "Hash" which doesn't have collisions.
        # This hash makes sense for TT-Rec more than the other methods, since
        # TT-Rec tends to have much smaller ranges than the other methods.
        output_range = int(math.ceil(vocab ** (1 / n_chunks)))
        hash = cce.QRHash(num_hashes=n_chunks, output_range=output_range)

        # Find largest allowable rank
        tt, rank = None, 1
        while True:
            tt_new = cce.TensorTrainEmbedding(rank, dimension, hash=hash, split_dim=True)
            if tt_new.size() > num_params:
                break
            tt = tt_new
            rank += 1
        print(f"Notice: Using {tt.size()} params, rather than {num_params}. {rank=}, {hash.range=}")
        return tt
    elif method == 'dhe':
        # Find largest allowable rank
        emb, rank = None, 1
        while True:
            hash = cce.MultiHash(num_hashes=rank, output_range=2**62)
            n_hidden = int(math.ceil(rank**(1/n_chunks)))
            emb_new = cce.DeepHashEmbedding(rank, dimension, n_hidden, hash)
            if emb_new.size() > num_params:
                break
            emb, rank = emb_new, rank+1
        print(f"Notice: Using {emb.size()} params, rather than {num_params}. {rank=}, {n_hidden=}")
        return emb
    raise Exception(f'{method=} not supported.')


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
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        bs, dim = user_emb.shape
        #mix = torch.relu(user_emb + item_emb)
        mix = user_emb * item_emb
        return self.mlp(mix).view(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=methods)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--last-cluster', type=int, default=7, help='Stop reclusering after this many epochs.')
    parser.add_argument('--dim', type=int, default=32, help='Dimension of embeddings')
    parser.add_argument('--ppd', type=int, default=200, help='Parameters per dimension')
    parser.add_argument('--dataset', type=str, default='ml-100k', choices=['ml-100k', 'ml-1m', 'ml-20m', 'ml-25m'])
    parser.add_argument('--seed', type=int, default=0xcce)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    # Seed for reproducability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cpu"
    # MPS has some bugs related to broadcasting scatter_add
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    if torch.cuda.is_available():
        device = "cuda:0"
    print(f'Device: {device}')

    # Load and process the data. We predict whether the user rated something >= 3.
    # data = Dataset.load_builtin(args.dataset)
    # df = pd.DataFrame(data.raw_ratings, columns=["user", "item", "rate", "id"])
    # df["rate"] = df["rate"].apply(lambda x: 1 if float(x) >= 3 else 0)
    # df["user"] = df["user"].astype("category").cat.codes.values
    # df["item"] = df["item"].astype("category").cat.codes.values
    #train = torch.load('data/ml-1m.train')
    #valid = torch.load('data/ml-1m.val')
    train, valid = data.prepare_movielens(args.dataset)
    train[:, 2] = (train[:, 2] >= 3).to(torch.int)
    valid[:, 2] = (valid[:, 2] >= 3).to(torch.int)

    # Instantiate the model and define the loss function and optimizer
    dim = args.dim
    num_params = args.ppd * dim
    n_users = max(train[:, 0].max(), valid[:, 0].max()) + 1
    n_items = max(train[:, 1].max(), valid[:, 1].max()) + 1
    # n_users = df["user"].nunique()
    # n_items = df["item"].nunique()
    print(f'Unique users: {n_users}, Unique items: {n_items}, #params: {num_params}')

    model = RecommenderNet(n_users, n_items, num_params, dim=dim, method=args.method).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters())

    # Create DataLoader
    # train = df.sample(frac=0.8, random_state=args.seed)
    # valid = df.drop(train.index)
    # train_tensor = RatingDataset(train)
    # valid_tensor = RatingDataset(valid)
    # train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    # valid_loader = DataLoader(valid_tensor, batch_size=64, shuffle=False, num_workers=args.num_workers, persistent_workers=True)


    # Train the model
    for epoch in tqdm.tqdm(range(args.epochs)):
        start = time.time()
        model.train()
        total_loss = 0
        for batch in tqdm.tqdm(train.split(args.batch_size), leave=False):
            user, item, label = batch.T.to(device)
            optimizer.zero_grad()
            prediction = model(user, item)
            loss = criterion(prediction, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss * args.batch_size / len(train)

        # Validate the model
        model.eval()
        total_loss = 0
        y_true, y_pred = [], []  # To collect the true labels and the predicted scores
        with torch.no_grad():
            for batch in tqdm.tqdm(valid.split(args.batch_size), leave=False):
                user, item, label = batch.T.to(device)
                prediction = model(user, item)
                loss = criterion(prediction, label.float())
                total_loss += loss.item()
                # Save values for AUC computation
                y_true += label.cpu().numpy().tolist()
                y_pred += prediction.cpu().numpy().tolist()
            valid_loss = total_loss * args.batch_size / len(valid)
            valid_auc = roc_auc_score(y_true, y_pred)

        print(f"Epoch: {epoch}, Time: {time.time() - start:.3}s, Train Loss: {train_loss:.3}, Validation Loss: {valid_loss:.3}, AUC: {valid_auc:.3}")

        if model.method in ('cce', 'cce_robe') and epoch < args.last_cluster:
            start = time.time()
            model.user_embedding.cluster(verbose=False)
            model.item_embedding.cluster(verbose=False)
            print(f'Clustering. Time: {time.time() - start:.3}s')


if __name__ == '__main__':
    main()
