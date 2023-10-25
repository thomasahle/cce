<p align="center" margin="0">
    <img src="https://raw.githubusercontent.com/thomasahle/cce/main/docs/banner.png" />
<i>We train embedding tables with fewer parameters by combining multiple sketches of the same data, iteratively.</i></p>

# Clustered Compositional Embeddings
This repository contains the code for the paper ["Clustering the Sketch: Dynamic Compression for Embedding Tables"](https://arxiv.org/abs/2210.05974).

## Abstract
Embedding tables are used by machine learning systems to work with categorical features. In modern Recommendation Systems, these tables can be very large, necessitating the development of new methods for fitting them in memory, even during training. We suggest **Clustered Compositional Embeddings** (CCE) which combines clustering-based compression like quantization to codebooks with dynamic methods like [Hashing Trick [ICML '09]](https://arxiv.org/abs/0902.2206) and [Compositional Embeddings (QR) [KDD '20]](https://arxiv.org/abs/1909.02107). Experimentally CCE achieves the best of both worlds: The high compression rate of codebook-based quantization, but dynamically like hashing-based methods, so it can be used during training. Theoretically, we prove that CCE is guaranteed to converge to the optimal codebook and give a tight bound for the number of iterations required.

## Example code

```python
import torch, cce

class GMF(torch.nn.Module):
    """ A simple Generalized Matrix Factorization model """
    def __init__(self, n_users, n_items, dim, num_params):
        super().__init__()
        self.user_embedding = cce.make_embedding(n_users, num_params, dim, 'cce', n_chunks=4)
        self.item_embedding = cce.make_embedding(n_items, num_params, dim, 'cce', n_chunks=4)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        return torch.sigmoid((user_emb * item_emb).sum(-1))

    def epoch_end(self):
        self.user_embedding.cluster()
        self.item_embedding.cluster()
```

Instead of the Clustered Compositional Embedding, the library also contain many other compressed embedding methods, such as `cce.RobeEmbedding`, `cce.CompositionalEmbedding`, `cce.TensorTrainEmbedding` and `cce.DeepHashEmbedding`.
See [cce/\_\_init\_\_.py](https://github.com/thomasahle/cce/blob/main/cce/__init__.py) for examples on how to initialize them.

## Key Takeaways
- **Context:** Modern Recommendation Systems require large embedding tables, challenging to fit in memory during training.

- **Solution:** CCE combines hashing/sketching methods with clustering during training, to learn an efficent sparse, data dependent hash function.

- **Contributions:**
  - CCE fills the gap between post-training compression (like Product Quantization) and during-training random mixing techniques (like Compositional Embeddings).
  - CCE provably finds the optimal codebook with bounded iterations, at least for linear models.
  - CCE experimentally outperforms all other methods for training large recommendation systems.
  - We provide a large, standardized library of related methods available on GitHub.

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
@inproceedings{tsang2023clustering,
  title={Clustering Embedding Tables, Without First Learning Them},
  author={Tsang, Henry Ling-Hei and Ahle, Thomas Dybdahl},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
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

### Criteo and DLRM
We adapted the [Deep Learning Recommendation Model](https://github.com/facebookresearch/dlrm) (DLRM) model to use CCE.
Even reducing the number of parameters by a factor 8,500, we were able to get the same test loss (Binary cross entropy) as the full DLRM model.

<table>
  <tr>
    <th width="50%">Criteo Kaggle</th>
    <th width="50%">Criteo Terabyte</th>
  </tr>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/criteo2.png" alt="Criteo Kaggle" width="100%"/>
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/terabyte.png" alt="Criteo Terabyte" width="100%"/>
    </td>
  </tr>
  <tr>
      <td>
                  <p>We trained DLRM on the <a href="https://www.kaggle.com/competitions/criteo-display-ad-challenge">Criteo Kaggle dataset</a> with
different compression algorithms. Each of 27 categorical features was given its own embedding table, where we
limited the number of parameters in the largest table as shown in the x-axis.
        See more details in the paper.</p>
      </td>
      <td>
                  <p>The <a href="https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/">Download Criteo 1TB Click Logs dataset</a> is the largest publicly available recommendation dataset.
        We trained for 10 epochs using the CCE method, as well as hashing, but only one repetition as the dataset is massive.</p>
      </td>
  </tr>
</table>

### Movielens and GMF
In examples/movielens.py we implemented a Generalized Matrix Factorization model, as in the example code above.
We then ran each method in the library on the different MovieLens datasets, ML-1m, ML-20m and ML-25m.
We also created a synthetic dataset, using random embeddings and the GMF model to generate interaction data.

| Size | MovieLens | Synthetic |
| --- | --- | --- |
| 1M | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/ml-1m-auc.png" width="100%"/> |  <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/syn-1m.png" width="100%"/> |
| 10M | . | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/syn-10m.png" width="100%"/> |
| 20M | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/ml-20-auc.png" width="100%"/> | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/syn-20-auc.png" width="100%"/> |
| 25M | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/ml-25-auc.png" width="100%"/> | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/syn-25-auc.png" width="100%"/> |
| 100M |  | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/syn-100-auc.png" width="100%"/> |

All models use 32 dimensional embedidng tables of the given method.

## Parameters

Many of the methods which utilize hashing allow a `n_chunks` parameter, which defines how many sub-vectors are combined to get the final embedding.
Increasing `n_chunks` nearly always give better results (as evidenced in the plot below), but it also increases time usage.
In our results above we always used a default of `n_chunks=4`.
| CE | CCE |
| --- | --- |
| <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/ml1-ce-splits.png" width="100%" /> | <img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/ml25-cce-splits.png" width="100%" /> |
