import torch
import torch.nn as nn
import math

from cce import hash


class DeepHashEmbedding(nn.Module):
    # The paper suggests using nn.Mish activation, but in our experiments plain ReLU
    # worked a lot better.
    def __init__(
        self,
        rank: int,
        dim: int,
        n_hidden: int,
        hash: hash.MultiHash,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.hash = hash
        self.dim = dim
        assert n_hidden >= 1

        layers = [nn.Linear(hash.num_hashes, rank)]
        for _ in range(n_hidden - 1):
            layers += [activation(), nn.Linear(rank, rank)]
        layers += [activation(), nn.Linear(rank, dim)]
        self.model = nn.Sequential(*layers)

    def size(self):
        return sum(
            lin.weight.numel() + lin.bias.numel()
            for lin in self.model
            if isinstance(lin, nn.Linear)
        )

    def forward(self, x):
        hs = self.hash(x) / self.hash.range  # [0, 1)
        hs = (2 * hs - 1) / self.dim**0.5  # Normalize to the usual
        return self.model(hs)
