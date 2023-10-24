import math
import torch.nn as nn
from itertools import takewhile, count

from .robe import RobeEmbedding
from .ce import CompositionalEmbedding
from .bloom import BloomEmbedding
from .hemb import (
    HashEmbedding,
    HashEmbedding2,
    RobeWeightedHashEmbedding,
)
from .hashnet import HashNetEmbedding
from .tt import TensorTrainEmbedding
from .cce import CCEmbedding
from .cce_robe import CCERobembedding
from .dhe import DeepHashEmbedding
from .low_dimensional import LowDimensionalEmbedding
from .hash import MultiHash, QRHash, PolyHash
from .sparse import SparseCodingEmbedding, SparseCodingEmbedding2


methods = [
    "robe",
    "ce",
    "simple",
    "cce",
    "full",
    "tt",
    "cce_robe",
    "dhe",
    "bloom",
    "hemb_original",
    "hemb_flat",
    "hemb_indep",
    "hemb_optional",
    "rhemb",
    "hnet",
    "ldim",
    "sparse",
    "sparse2",
]


def make_embedding(vocab, num_params, dimension, method, n_chunks, sparse):
    """
    Helper function to initialize compressed embeddings.
    The API follows torch.nn.Embedding, but has some extra options.

    This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.

    Parameters:
    -----------
    vocab : int
        The size of the vocabulary, which is the total number of IDs supported.
    num_params : int
        The number of parameters to use for the embedding.
    dimension : int
        The dimensionality of the embeddings. Each ID will be represented as a vector of this many dimensions.
    method : str
        The method to use for generating the embeddings. Supported methods are: 'robe', 'ce', 'cce', 'cce_robe', etc.
    n_chunks : int
        Many methods use some kind of "chunking" of the dimension space. Such as the number of hash functions in the Bloom Embedding, the number of columns in CCE, or the number of layers in DHE.
    sparse : bool
        Some methods support sparse gradients.

    Notes:
    ------
    All embeddings are initialized to give embedding vectors of roughly unit norm.
    This is different from nn.Embedding, which defaults to norm sqrt(d).
    However, DLRM uses unit norm, and from our experiements it works better.

    Examples:
    ---------
    >>> emb = make_embedding(vocab=10**4, num_params=1024, dimension=32, method='cce', n_chunks=4, sparse=False)
    >>> input = torch.randint(10**4, size=(100,))
    >>> assert emb(input).shape == (100, 32)
    """

    # Concatenative methods
    if method in ["robe", "ce", "cce", "cce_robe"]:
        chunk_dim = dimension // n_chunks
        assert n_chunks * chunk_dim == dimension, f"Dimension not divisible by {n_chunks}"
    if method == "robe":
        hash = PolyHash(num_hashes=n_chunks, output_range=num_params)
        return RobeEmbedding(size=num_params, chunk_size=chunk_dim, hash=hash, sparse=sparse)
    if method == "ce":
        rows = num_params // dimension
        hash = PolyHash(num_hashes=n_chunks, output_range=rows)
        return CompositionalEmbedding(rows=rows, chunk_size=chunk_dim, hash=hash, sparse=sparse)
    elif method == "cce":
        return CCEmbedding(vocab=vocab, num_params=num_params, chunk_size=chunk_dim, n_chunks=n_chunks)
    elif method == "cce_robe":
        return CCERobembedding(vocab=vocab, num_params=num_params, chunk_size=chunk_dim, n_chunks=n_chunks)
    elif method == "hnet":
        hash = PolyHash(num_hashes=dimension, output_range=num_params)
        return HashNetEmbedding(num_params, hash, sparse=sparse)

    # Additive methods
    elif method == "bloom":
        rows = num_params // dimension
        hash = PolyHash(num_hashes=n_chunks, output_range=rows)
        return BloomEmbedding(rows, dimension, hash)
    elif method == "sparse":
        return SparseCodingEmbedding(num_params, vocab, dimension, n_chunks, sparse=sparse)
    elif method == "sparse2":
        return SparseCodingEmbedding2(num_params, vocab, dimension, n_chunks, sparse=sparse)
    elif method == "rhemb":
        return RobeWeightedHashEmbedding(num_params, dimension, n_chunks, sparse=sparse)
    elif method == "hemb_original":
        return HashEmbedding(num_params, dimension, n_chunks, method=hemb.METHOD_ORIGINAL)
    elif method == "hemb_flat":
        return HashEmbedding(num_params, dimension, n_chunks, method=hemb.METHOD_FLAT)
    elif method == "hemb_indep":
        return HashEmbedding(num_params, dimension, n_chunks, method=hemb.METHOD_INDEP)
    elif method == "hemb_optional":
        return HashEmbedding2(num_params, dimension, n_chunks)

    # Other methods
    elif method == "simple":
        rows = num_params // dimension
        hash = PolyHash(num_hashes=1, output_range=rows)
        return CompositionalEmbedding(rows=rows, chunk_size=dimension, hash=hash, sparse=sparse)
    elif method == "full":
        emb = nn.Embedding(vocab, dimension, sparse=sparse)
        nn.init.uniform_(emb.weight, -(dimension**-0.5), dimension**-0.5)
        return emb

    # Some methods require some more complicated sizing logic
    elif method in ["tt", "dhe", "ldim"]:

        def make(rank):
            if method == "tt":
                output_range = int(math.ceil(vocab ** (1 / n_chunks)))
                hash = QRHash(num_hashes=n_chunks, output_range=output_range)
                return TensorTrainEmbedding(rank, dimension, hash)
            if method == "dhe":
                n_hidden = int(math.ceil(rank ** (1 / n_chunks)))
                return DeepHashEmbedding(rank, dimension, n_hidden)
            if method == "ldim":
                return LowDimensionalEmbedding(vocab, rank, dimension, sparse)

        # It might be that even the lowest rank uses too many parameters.
        if make(1).size() > num_params:
            raise ValueError(f"Error: Too few parameters ({num_params=}) to initialize model.")

        rank = max(takewhile((lambda r: make(r).size() < num_params), count(1)))
        emb = make(rank)
        print(f"Notice: Using {emb.size()} params, rather than {num_params}. {rank=}")
        return emb
    raise NotImplementedError(f"{method=} not supported.")
