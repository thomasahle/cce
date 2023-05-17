import torch
import torch.nn as nn
from cce import hash


class RobeEmbedding(nn.Module):
    def __init__(
        self,
        size: int,
        chunk_size: int,
        multi_hash: hash.MultiHash,
    ):
        """ Output dimension = chunk_size * multi_hash.n_hashes """
        super().__init__()
        self.chunk_size = chunk_size
        self.multi_hash = multi_hash
        self.data = nn.Parameter(torch.empty(size))
        nn.init.uniform_(self.data)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape
        hash_values = self.multi_hash(input_tensor)  # (batch_size, num_hashes)
        slices = self.data[
            hash_values[..., None]
            + torch.arange(
                self.chunk_size,
                device=hash_values.device
            ) % len(self.data)
        ]  # (batch_size, num_hashes, chunk_size)
        return slices.reshape(*batch_size, -1)

