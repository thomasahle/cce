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

import data

import cce


methods = ['robe', 'ce', 'simple', 'cce', 'full', 'tt', 'cce_robe', 'dhe', 'hash', 'hnet', 'whemb']

def make_embedding(vocab, num_params, dimension, method):
    n_chunks = 4
    chunk_dim = dimension // n_chunks
    assert n_chunks * chunk_dim == dimension, f"Dimension not divisible by {n_chunks}"

    if method == 'robe':
        hash = cce.PolyHash(num_hashes=n_chunks, output_range=num_params)
        return cce.RobeEmbedding(size=num_params, chunk_size=chunk_dim, hash=hash)
    if method == 'ce':
        rows = num_params // dimension
        hash = cce.PolyHash(num_hashes=n_chunks, output_range=rows)
        return cce.CompositionalEmbedding(rows=rows, chunk_size=chunk_dim, hash=hash)
    elif method == 'hash':
        rows = num_params // dimension
        hash = cce.PolyHash(num_hashes=n_chunks, output_range=rows)
        return cce.HashEmbedding(rows, dimension, hash)
    elif method == 'whemb':
        rows = num_params // dimension // 2
        return cce.WeightedHashEmbedding(rows, dimension, n_chunks)
    elif method == 'hnet':
        hash = cce.PolyHash(num_hashes=dimension, output_range=num_params)
        return cce.HashNetEmbedding(num_params, hash)
    elif method == 'simple':
        rows = num_params // dimension
        hash = cce.PolyHash(num_hashes=1, output_range=rows)
        return cce.CompositionalEmbedding(rows=rows, chunk_size=dimension, hash=hash)
    elif method == 'cce':
        # We divide rows by two, since the CCE embedding will use two tables
        return cce.CCEmbedding(vocab=vocab, rows=num_params // dimension // 2, chunk_size=dimension//n_chunks, n_chunks=n_chunks)
    elif method == 'cce_robe':
        return cce.CCERobembedding(vocab=vocab, size=num_params//2, chunk_size=dimension//n_chunks, n_chunks=n_chunks)
    elif method == 'full':
        emb = nn.Embedding(vocab, dimension)
        nn.init.uniform_(emb.weight, -(dimension**-0.5), dimension**-0.5)
        return emb
    # Some methods require some more complicated sizing logic
    elif method in ['tt', 'dhe']:
        def make(rank):
            if method == 'tt':
                output_range = int(math.ceil(vocab ** (1 / n_chunks)))
                hash = cce.QRHash(num_hashes=n_chunks, output_range=output_range)
                return cce.TensorTrainEmbedding(rank, dimension, hash)
            if method == 'dhe':
                hash = cce.MultiHash(num_hashes=rank, output_range=2**62)
                n_hidden = int(math.ceil(rank**(1/n_chunks)))
                return cce.DeepHashEmbedding(rank, dimension, n_hidden, hash)
        # It might be that even the lowest rank uses too many parameters.
        if make(1).size() > num_params:
            print(f"Error: Too few parameters ({num_params=}) to initialize model.")
            sys.exit()
        rank = max(takewhile((lambda r: make(r).size() < num_params), count(1)))
        emb = make(rank)
        print(f"Notice: Using {emb.size()} params, rather than {num_params}. {rank=}, {emb.hash.range=}")
        return emb
    raise Exception(f'{method=} not supported.')


class GMF(nn.Module):
    """ A simple Generalized Matrix Factorization model """
    def __init__(self, n_users, n_items, num_params, dim, method):
        super().__init__()
        self.method = method
        self.user_embedding = make_embedding(n_users, num_params, dim, method)
        self.item_embedding = make_embedding(n_items, num_params, dim, method)

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
    #if torch.backends.mps.is_available():
    #    device = torch.device("mps")
    if torch.cuda.is_available():
        device = "cuda:0"
    print(f'Device: {device}')

    # Load and process the data. We predict whether the user rated something >= 3.
    train, valid = data.prepare_movielens(args.dataset)

    # Instantiate the model and define the loss function and optimizer
    dim = args.dim
    num_params = args.ppd * dim
    n_users = max(train[:, 0].max(), valid[:, 0].max()) + 1
    n_items = max(train[:, 1].max(), valid[:, 1].max()) + 1
    print(f'Unique users: {n_users}, Unique items: {n_items}, #params: {num_params}')
    print(f'Unique users: {torch.unique(train[:,0]).size()}, Unique items: {torch.unique(train[:,1]).size()}')

    model = GMF(n_users, n_items, num_params, dim=dim, method=args.method).to(device)
    criterion = nn.BCELoss()
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

        if valid_loss > old_valid_loss and valid_auc < old_auc:
            print('Early stopping')
            break
        old_valid_loss = valid_loss
        old_auc = valid_auc

        if model.method in ('cce', 'cce_robe') and epoch < args.last_cluster:
            start = time.time()
            model.user_embedding.cluster(verbose=False)
            model.item_embedding.cluster(verbose=False)
            print(f'Clustering. Time: {time.time() - start:.3}s')


if __name__ == '__main__':
    main()
