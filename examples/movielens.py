import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torchdata.datapipes as dp
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
import random
from pytorch_lightning.loggers import WandbLogger
from functools import partial


torch.set_float32_matmul_precision('medium')


def negative_sample(_, n_users, n_movies):
    return (
        torch.tensor(random.randrange(n_users), dtype=torch.int64),
        torch.tensor(random.randrange(n_users), dtype=torch.int64),
        torch.tensor(0, dtype=torch.float32)
    )


def process_data(data):
    user, movie, rating, _ = data
    return (
        torch.tensor(int(user) - 1, dtype=torch.int64),
        torch.tensor(int(movie) - 1, dtype=torch.int64),
        torch.tensor(float(float(rating) >= 3), dtype=torch.float32)
    )


class MovieLensDataModule(pl.LightningDataModule):
    def __init__(self, file_path, n_users, n_movies,
                 batch_size=128, num_workers=1, negative_sampling=True):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_users = n_users
        self.n_movies = n_movies
        self.negative_sampling = negative_sampling

    def prepare_data(self):
        self.datapipe = (
            dp.iter.FileOpener([self.file_path], mode='rt')
            .parse_csv(delimiter=',', skip_lines=1)
            .map(process_data)
            # mix in negative samples
            .mux_longest(
                dp.iter.IterableWrapper(range(25_000_000))
                .map(partial(negative_sample, n_users=self.n_users, n_movies=self.n_movies))
            )
            .shuffle(buffer_size=1_000_000)
        )

    def setup(self, stage=None):
        self.train_data, self.val_data = self.datapipe.random_split(
            total_length=25000096,
            # The right way to do this split is to take the last movie watched
            # by each user (according to the timestamps) and put that in validation.
            weights={"train": 0.9, "valid": 0.1},
            seed=0
        )
        # Add sharding filter
        self.train_data = self.train_data.sharding_filter(self.num_workers)
        self.val_data = self.val_data.sharding_filter(self.num_workers)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)


class MovieLensModel(pl.LightningModule):
    def __init__(self, n_users, n_movies, dim=50, lr=0.01):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, dim)
        self.movie_embedding = nn.Embedding(n_movies, dim)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.lr = lr

    def forward(self, user, movie):
        user_embed = self.user_embedding(user)
        movie_embed = self.movie_embedding(movie)
        return (user_embed * movie_embed).mean(axis=1)
        # return self.cos(user_embed, movie_embed)

    def training_step(self, batch, batch_idx):
        user, movie, rating = batch
        prediction = self(user, movie)
        loss = F.binary_cross_entropy_with_logits(prediction, rating)
        # loss = F.mse_loss(prediction, rating)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user, movie, rating = batch
        prediction = self(user, movie)
        # loss = F.mse_loss(prediction, rating)
        loss = F.binary_cross_entropy_with_logits(prediction, rating)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument("--fast-dev-run", action='store_true')
    args = parser.parse_args()

    # Data
    data_module = MovieLensDataModule(args.path, args.batch_size, args.num_workers)

    # Model
    model = MovieLensModel(n_users=164_000, n_movies=210_000, dim=args.dim, lr=args.lr)

    # Create a wandb logger
    try:
        logger = WandbLogger(project='movielens', log_model='all')
    except ModuleNotFoundError:
        logger = None

    # Trainer
    trainer = pl.Trainer(max_epochs=args.epochs, fast_dev_run=args.fast_dev_run, logger=logger)

    # Find the maximum batch size
    tuner = pl.tuner.tuning.Tuner(trainer)
    tuner.lr_find(model, datamodule=data_module)
    #tuner.scale_batch_size(model, datamodule=data_module)

    # Training
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()

