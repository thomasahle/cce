import torch
import torch.nn as nn
from cce import hash

def get_slices(table, indices, chunk_size):
    """
    This function fetches multiple slices from a tensor, `table`. Each slice is determined
    by the start index from 'indices' and has a length of 'chunk_size'. In case a slice
    exceeds the length of the table, it wraps around using modulo arithmetic.

    Parameters:
    - table (torch.Tensor): The source tensor from which slices are to be retrieved.
      Dimension: (table_size,)
    - indices (torch.Tensor): Tensor of starting indices for each slice. Can have any shape,
      but the last dimension is the number of slices to retrieve.
      Dimension: (*, num_slices)
    - chunk_size (int): The length of each slice to retrieve.

    Returns:
    - torch.Tensor: Tensor containing the slices. Dimension is the same as `indices` with
      an additional last dimension of size 'chunk_size'.
      Dimension: (*, num_slices, chunk_size)

    Example:
    If table = [0, 1, 2, 3, 4] and indices = [[0, 3], [2, 1]], and chunk_size = 2:
    Result will be [[[0, 1], [3, 4]], [[2, 3], [1, 2]]]
    """
    return table[(indices[..., None] + torch.arange(chunk_size, device=indices.device)) % len(table)]

class RobeEmbedding(nn.Module):
    def __init__(
        self,
        size: int,
        chunk_size: int,
        hash: hash.MultiHash,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.hash = hash
        self.table = nn.Parameter(torch.empty(size))
        self.reset_parameters()

    def reset_parameters(self):
        dim = self.chunk_size * self.hash.num_hashes
        nn.init.uniform_(self.table, -(dim**-0.5), dim**-0.5)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape
        hash_values = self.hash(input_tensor)  # (batch_size, num_hashes)
        slices = get_slices(self.table, hash_values, self.chunk_size) # (batch_size, num_hashes, chunk_size)
        return slices.flatten(start_dim=-2)
