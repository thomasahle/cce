import torch
import torch.nn as nn
from cce import hash


class LowDimensionalEmbedding(nn.Module):
    def __init__(
        self,
        vocab: int,
        dim0: int,
        dim1: int,
        sparse=False,
        bias=True,
    ):
        super().__init__()
        self.dim1 = dim1
        self.table = nn.Embedding(vocab, dim0, sparse=sparse)
        self.bias = bias
        self.upscale = nn.Linear(dim0, dim1, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        dim0, dim1 = self.upscale.weight.T.shape
        nn.init.uniform_(self.table.weight, -(dim0**-0.5), dim0**-0.5)
        nn.init.uniform_(
            self.upscale.weight, -((dim1 / dim0) ** (-0.5)), (dim1 / dim0) ** (-0.5)
        )

    def size(self):
        return (
            self.upscale.weight.numel()
            + (self.upscale.bias.numel() if self.bias else 0)
            + self.table.weight.numel()
        )

    def forward(self, x):
        return self.upscale(self.table(x))
