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
        #activation=nn.Mish,
    ):
        super().__init__()
        assert n_hidden >= 1
        self.rank = rank
        self.dim = dim

        self.hash = hash.MultiHash(num_hashes=rank, output_range=2**62)

        layers = []
        for _ in range(n_hidden):
            layers += [nn.Linear(rank, rank)]
        layers += [nn.Linear(rank, dim)]
        self.layers = nn.ModuleList(layers)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.uniform_(layer.weight, -self.rank**(-.5), self.rank**(-.5))

    def size(self):
        return sum(
            lin.weight.numel() + lin.bias.numel()
            for lin in self.layers
            if isinstance(lin, nn.Linear)
        )

    def forward(self, x):
        hs = self.hash(x) / self.hash.range  # [0, 1)
        hs = (2 * hs - 1) / self.hash.num_hashes**0.5  # Normalize to the usual
        #hs = hs / self.hash.num_hashes**0.5  # Normalize to the usual
        for i, layer in enumerate(self.layers[:-1]):
            #hs = (layer(torch.relu(hs)) + hs) / 2
            hs = torch.relu(layer(hs))
        hs = self.layers[-1](hs)
        return hs
