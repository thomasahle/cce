import torch
import torch.nn as nn
from cce import hash


class SimpleEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        hash: hash.SingleHash,
    ):
        super().__init__()
        self.hash = hash
        self.emb = nn.Embedding(num_embeddings, dim)
        self.reset_parameters()

    def reset_parameters(self):
        dim = self.emb.embedding_dim
        nn.init.uniform_(self.emb.weight, -dim**-0.5, dim**-0.5)

    def forward(self, input_tensor):
        hash_values = self.hash(input_tensor) % self.emb.num_embeddings
        return self.emb(hash_values)
