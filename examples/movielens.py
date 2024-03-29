import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset as TorchDataset, DataLoader, TensorDataset
import argparse
import time
from sklearn.metrics import roc_auc_score
import tqdm

import dataset
import cce


class GMF(nn.Module):
    """A simple Generalized Matrix Factorization model"""

    def __init__(self, n_users, n_items, num_params, dim, method, n_chunks, sparse):
        super().__init__()
        self.method = method
        self.user_embedding = cce.make_embedding(n_users, num_params, dim, method, n_chunks, sparse)
        self.item_embedding = cce.make_embedding(n_items, num_params, dim, method, n_chunks, sparse)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        mix = user_emb * item_emb
        return torch.sigmoid(mix.sum(-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=methods)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--last-cluster", type=int, default=-1, help="Stop reclusering after this many epochs.")
    parser.add_argument("--dim", type=int, default=32, help="Dimension of embeddings")
    parser.add_argument("--ppd", type=int, default=200, help="Parameters per dimension")
    parser.add_argument("--dataset", type=str, default="ml-100k")
    parser.add_argument("--seed", type=int, default=0xCCE)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--n-chunks", type=int, default=4)
    args = parser.parse_args()

    # Seed for reproducability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cpu"
    if not args.cpu:
        # MPS has some bugs related to broadcasting scatter_add
        # if torch.backends.mps.is_available():
        # device = torch.device("mps")
        if torch.cuda.is_available():
            device = "cuda:0"
    print(f"Device: {device}")

    # Load and process the data. We predict whether the user rated something >= 3.
    if args.dataset.startswith("syn"):
        _, v, n = args.dataset.split("-")
        train, valid = dataset.make_synthetic(int(n), int(v))
    else:
        train, valid = dataset.prepare_movielens(args.dataset)

    # Instantiate the model and define the loss function and optimizer
    dim = args.dim
    num_params = args.ppd * dim
    max_user = max(train[:, 0].max(), valid[:, 0].max())
    max_item = max(train[:, 1].max(), valid[:, 1].max())
    n_users = torch.unique(torch.cat([train[:, 0], valid[:, 0]])).size()
    n_items = torch.unique(torch.cat([train[:, 1], valid[:, 1]])).size()
    print(f"Max user id: {max_user}, Max item id: {max_item}, #params: {num_params}")
    print(f"Unique users: {n_users}, Unique items: {n_items}")
    print(
        "1 ratios:",
        train[:, 2].to(float).mean().numpy(),
        valid[:, 2].to(float).mean().numpy(),
    )

    model = (
        GMF(
            max_user + 1,
            max_item + 1,
            num_params,
            dim=dim,
            method=args.method,
            n_chunks=args.n_chunks,
            sparse=args.sparse,
        )
        .float()
        .to(device)
    )
    criterion = nn.BCELoss()
    if args.sparse:
        print(
            "Notice: Sparsity is only supported by some embeddings, and is generally only useful for vocabs >= 100_000"
        )
        optimizer = torch.optim.SparseAdam(model.parameters())
    else:
        optimizer = torch.optim.AdamW(model.parameters())

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

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

        scheduler.step(valid_loss)

        print(
            f"Epoch: {epoch}, "
            f"Time: {train_time:.3}s, "
            f"Train Loss: {train_loss:.3}, "
            f"Validation Loss: {valid_loss:.3}, "
            f"AUC: {valid_auc:.3}"
        )

        if valid_loss > old_valid_loss * 1.01 and valid_auc * 1.01 < old_auc:
            print("Early stopping")
            break
        old_valid_loss = min(old_valid_loss, valid_loss)
        old_auc = max(old_auc, valid_auc)

        last_cluster = args.last_cluster
        if last_cluster == -1:
            last_cluster = int(0.75 * args.epochs)
        if hasattr(model.user_embedding, "cluster") and epoch < last_cluster:
            start = time.time()
            model.user_embedding.cluster(verbose=False, max_time=train_time / 2)
            model.item_embedding.cluster(verbose=False, max_time=train_time / 2)
            cluster_time = time.time() - start
            print(f"Clustering. Time: {cluster_time:.3}s")
            if cluster_time > train_time and cce.cce.use_sklearn:
                print("Switching to faiss for clustering")
                cce.cce.use_sklearn = False


if __name__ == "__main__":
    main()
