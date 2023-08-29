import pandas as pd
import torch
import math
from torch import nn
import numpy as np
from torch.utils.data import Dataset as TorchDataset, DataLoader
import argparse
import time
from sklearn.metrics import roc_auc_score
import tqdm
import sys
from itertools import takewhile, count

import dataset

import cce


methods = ['robe', 'ce', 'simple', 'cce', 'full', 'tt', 'cce_robe', 'dhe', 'bloom', 'hemb', 'hnet', 'whemb', 'ldim']

def make_embedding(vocab, num_params, dimension, method, sparse, seed):
    n_chunks = 4
    chunk_dim = dimension // n_chunks
    assert n_chunks * chunk_dim == dimension, f"Dimension not divisible by {n_chunks}"

    if method == 'robe':
        hash = cce.PolyHash(num_hashes=n_chunks, output_range=num_params)
        return cce.RobeEmbedding(size=num_params, chunk_size=chunk_dim, hash=hash, sparse=sparse)
    if method == 'ce':
        rows = num_params // dimension
        hash = cce.PolyHash(num_hashes=n_chunks, output_range=rows)
        return cce.CompositionalEmbedding(rows=rows, chunk_size=chunk_dim, hash=hash, sparse=sparse)
    elif method == 'bloom':
        rows = num_params // dimension
        hash = cce.PolyHash(num_hashes=n_chunks, output_range=rows)
        return cce.BloomEmbedding(rows, dimension, hash)
    elif method == 'whemb':
        return cce.WeightedHashEmbedding(num_params // dimension, dimension, n_chunks, sparse=sparse)
    elif method == 'hemb':
        return cce.HashEmbedding(num_params, dimension)
    elif method == 'hnet':
        hash = cce.PolyHash(num_hashes=dimension, output_range=num_params)
        return cce.HashNetEmbedding(num_params, hash, sparse=sparse)
    elif method == 'simple':
        rows = num_params // dimension
        hash = cce.PolyHash(num_hashes=1, output_range=rows)
        return cce.CompositionalEmbedding(rows=rows, chunk_size=dimension, hash=hash, sparse=sparse)
    elif method == 'cce':
        # We divide rows by two, since the CCE embedding will use two tables
        return cce.CCEmbedding(vocab=vocab, rows=num_params // dimension // 2, chunk_size=dimension//n_chunks, n_chunks=n_chunks, seed=seed)
    elif method == 'cce_robe':
        return cce.CCERobembedding(vocab=vocab, size=num_params//2, chunk_size=dimension//n_chunks, n_chunks=n_chunks)
    elif method == 'full':
        emb = nn.Embedding(vocab, dimension, sparse=sparse)
        nn.init.uniform_(emb.weight, -(dimension**-0.5), dimension**-0.5)
        return emb
    # Some methods require some more complicated sizing logic
    elif method in ['tt', 'dhe', 'ldim']:
        def make(rank):
            if method == 'tt':
                output_range = int(math.ceil(vocab ** (1 / n_chunks)))
                hash = cce.QRHash(num_hashes=n_chunks, output_range=output_range)
                return cce.TensorTrainEmbedding(rank, dimension, hash)
            if method == 'dhe':
                hash = cce.MultiHash(num_hashes=rank, output_range=2**62)
                n_hidden = int(math.ceil(rank**(1/n_chunks)))
                return cce.DeepHashEmbedding(rank, dimension, n_hidden, hash)
            if method == 'ldim':
                return cce.LowDimensionalEmbedding(vocab, rank, dimension, sparse)
        # It might be that even the lowest rank uses too many parameters.
        if make(1).size() > num_params:
            print(f"Error: Too few parameters ({num_params=}) to initialize model.")
            sys.exit()
        rank = max(takewhile((lambda r: make(r).size() < num_params), count(1)))
        emb = make(rank)
        print(f"Notice: Using {emb.size()} params, rather than {num_params}. {rank=}")
        return emb
    raise Exception(f'{method=} not supported.')


class GMF(nn.Module):
    """ A simple Generalized Matrix Factorization model """
    def __init__(self, n_users, n_items, num_params, dim, method, sparse, seed):
        super().__init__()
        self.method = method
        self.user_embedding = make_embedding(n_users, num_params, dim, method, sparse, seed)
        self.item_embedding = make_embedding(n_items, num_params, dim, method, sparse, seed)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        mix = user_emb * item_emb
        return torch.sigmoid(mix.sum(-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=methods)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--last-cluster', type=int, default=7, help='Stop reclusering after this many epochs.')
    parser.add_argument('--dim', type=int, default=32, help='Dimension of embeddings')
    parser.add_argument('--ppd', type=int, default=200, help='Parameters per dimension')
    parser.add_argument('--dataset', type=str, default='ml-100k')
    parser.add_argument('--seed', type=int, default=0xcce)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sparse', action='store_true')
    args = parser.parse_args()

    # Seed for reproducability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cpu"
    # MPS has some bugs related to broadcasting scatter_add
    #if torch.backends.mps.is_available():
    #    device = torch.device("mps")
    if torch.cuda.is_available():
        device = "cuda:0"
    print(f'Device: {device}')

    # Load and process the data. We predict whether the user rated something >= 3.
    if args.dataset.startswith('syn'):
        _, v, n = args.dataset.split('-')
        train, valid = dataset.make_synthetic(int(n), int(v))
    else:
        train, valid = dataset.prepare_movielens(args.dataset)

    # Instantiate the model and define the loss function and optimizer
    dim = args.dim
    num_params = args.ppd * dim
    max_user = max(train[:, 0].max(), valid[:, 0].max())
    max_item = max(train[:, 1].max(), valid[:, 1].max())
    n_users = torch.unique(torch.cat([train[:,0], valid[:,0]])).size()
    n_items = torch.unique(torch.cat([train[:,1], valid[:,1]])).size()
    print(f"Max user id: {max_user}, Max item id: {max_item}, #params: {num_params}")
    print(f"Unique users: {n_users}, Unique items: {n_items}")
    print("1 ratios:", train[:,2].to(float).mean().numpy(), valid[:,2].to(float).mean().numpy())

    model = GMF(max_user+1, max_item+1, num_params, dim=dim, method=args.method, sparse=args.sparse, seed=args.seed).to(device)
    criterion = nn.BCELoss()
    if args.sparse:
        print("Notice: Sparsity is only supported by some embeddings, and is generally only useful for vocabs >= 100_000")
        optimizer = torch.optim.SparseAdam(model.parameters())
    else:
        optimizer = torch.optim.AdamW(model.parameters())

    # For early stopping
    old_valid_loss = 10**10
    old_auc = 0

    # Train the model
    for epoch in tqdm.tqdm(range(args.epochs)):
        start = time.time()
        model.train()
        total_loss = 0
        train = train[torch.randperm(train.shape[0])]  # Shuffle
        for batch in tqdm.tqdm(train.split(args.batch_size), leave=False):
            user, item, label = batch.T.to(device)
            optimizer.zero_grad()
            prediction = model(user, item)
            loss = criterion(prediction, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss * args.batch_size / len(train)
        train_time = time.time() - start

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

        print(f"Epoch: {epoch}, Time: {train_time:.3}s, Train Loss: {train_loss:.3}, Validation Loss: {valid_loss:.3}, AUC: {valid_auc:.3}")

        if valid_loss > old_valid_loss * 1.1 and valid_auc * 1.1 < old_auc:
            print('Early stopping')
            break
        old_valid_loss = min(old_valid_loss, valid_loss)
        old_auc = max(old_auc, valid_auc)

        if model.method in ('cce', 'cce_robe') and epoch < args.last_cluster:
            start = time.time()
            model.user_embedding.cluster(verbose=False)
            model.item_embedding.cluster(verbose=False)
            cluster_time = time.time() - start
            print(f'Clustering. Time: {cluster_time:.3}s')
            if cluster_time > train_time:
                # Switch to faiss
                cce.cce.use_sklearns = False


if __name__ == '__main__':
    main()
