# Clustered Compositional Embeddings
This repository contains the code for the paper ["Clustering the Sketch: Dynamic Compression for Embedding Tables"](https://arxiv.org/abs/2210.05974).

## Abstract
Embedding tables are used by machine learning systems to work with categorical features. In modern Recommendation Systems, these tables can be very large, necessitating the development of new methods for fitting them in memory, even during training. We suggest Clustered Compositional Embeddings (CCE) which combines clustering-based compression like quantization to codebooks with dynamic methods like The Hashing Trick and Compositional Embeddings [Shi et al., 2020]. Experimentally CCE achieves the best of both worlds: The high compression rate of codebook-based quantization, but dynamically like hashing-based methods, so it can be used during training. Theoretically, we prove that CCE is guaranteed to converge to the optimal codebook and give a tight bound for the number of iterations required.

## Example code

```python
import cce

class GMF(nn.Module):
    """ A simple Generalized Matrix Factorization model """
    def __init__(self, n_users, n_items, dim, num_params):
        super().__init__()
        self.user_embedding = cce.CCEmbedding(vocab=n_users, rows=num_params // dim // 2, chunk_size=dim // 4, n_chunks=4)
        self.item_embedding = cce.CCEmbedding(vocab=n_users, rows=num_params // dim // 2, chunk_size=dim // 4, n_chunks=4)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        return torch.sigmoid((user_emb * item_emb).sum(-1))
```

Instead of the Clustered Compositional Embedding, the library also contain many other compressed embedding methods, such as cce.RobeEmbedding, cce.CompositionalEmbedding, cce.TensorTrainEmbedding and cce.DeepHashEmbedding.
See https://github.com/thomasahle/cce/blob/main/examples/movielens.py#L22 for examples on how to initialize them.

## Install

Install with
```bash
pip install -e .
```

Then run
```bash
python examples/movielens.py --method cce
```
To test run CCE on a simple movielens 100K dataset.
You can also use `--method robe` to use the Robe method, or `ce` for compositional embeddings.

## Citation

```
@article{tsang2022clustering,
  title={Clustering Embedding Tables, Without First Learning Them},
  author={Tsang, Henry Ling-Hei and Ahle, Thomas Dybdahl},
  journal={arXiv preprint arXiv:2210.05974},
  year={2022}
}
```

## Other implemented algorithms

- [Hashing Trick [ICML '09]](https://arxiv.org/abs/0902.2206)
- [HashedNets [ICML '15]](https://arxiv.org/abs/1504.04788)
- [Hash Embeddings [NeurIPS '17]](https://arxiv.org/abs/1709.03933)
- [Compositional Embeddings (QR) [KDD '20]](https://arxiv.org/abs/1909.02107)
- [Deep Hash Embedding (DHE) [KDD '21]](https://arxiv.org/abs/2010.10784)
- [Tensor Train (TT-Rec) [MLSys '21]](https://arxiv.org/abs/2101.11714)
- [Random Offset Block Embedding (ROBE) [MLSys '22]](https://proceedings.mlsys.org/paper_files/paper/2022/file/1eb34d662b67a14e3511d0dfd78669be-Paper.pdf)

## Results

For a simple 32 dim model, varying the number of parameters:

| Size | MovieLens | Synthetic |
| --- | --- | --- |
| 1M | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/ml-1m-auc.png" width="100%"/> |  <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/syn-1m.png" width="100%"/> |
| 10M | . | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/syn-10m.png" width="100%"/> |
| 20M | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/ml-20-auc.png" width="100%"/> | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/syn-20-auc.png" width="100%"/> |
| 25M | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/ml-25-auc.png" width="100%"/> | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/syn-25-auc.png" width="100%"/> |
| 100M |  | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/syn-100-auc.png" width="100%"/> |


Similarly a 16 dimensional model on Criteo Kaggle:
<img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/criteo2.png" alt="Criteo Kaggle" width="70%"/>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clustering-embedding-tables-without-first/click-through-rate-prediction-on-criteo)](https://paperswithcode.com/sota/click-through-rate-prediction-on-criteo?p=clustering-embedding-tables-without-first)

## Parameters

Many of the methods which utilize hashing allow a `n_chunks` parameter, which defines how many sub-vectors are combined to get the final embedding.
Increasing `n_chunks` nearly always give better results (as evidenced in the plot below), but it also increases time usage.
In our results above we always used a default of `n_chunks=4`.
| CE | CCE |
| --- | --- |
| <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/ml1-ce-splits.png" width="100%" /> | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/ml25-cce-splits.png" width="100%" /> |
