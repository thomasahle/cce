import torch
import torch.nn as nn
from cce import hash


class SimpleEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        hash: hash.SingleHash,
    ):
        super().__init__()
        self.hash = hash
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input_tensor):
        hash_values = self.hash(input_tensor) % self.emb.num_embeddings
        return self.emb(hash_values)

