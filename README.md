# Clustered Compositional Embeddings
This repository contains the code for the paper ["Clustering the Sketch: Dynamic Compression for Embedding Tables"](https://arxiv.org/abs/2210.05974).

## Abstract
Embedding tables are used by machine learning systems to work with categorical features. In modern Recommendation Systems, these tables can be very large, necessitating the development of new methods for fitting them in memory, even during training. We suggest Clustered Compositional Embeddings (CCE) which combines clustering-based compression like quantization to codebooks with dynamic methods like The Hashing Trick and Compositional Embeddings [Shi et al., 2020]. Experimentally CCE achieves the best of both worlds: The high compression rate of codebook-based quantization, but dynamically like hashing-based methods, so it can be used during training. Theoretically, we prove that CCE is guaranteed to converge to the optimal codebook and give a tight bound for the number of iterations required.

## Install

Install with
```
pip install -e .
```

Then run
```
python examples/movielens.py --method cce
```
To test run CCE on a simple movielens 100K dataset.
You can also use `--method robe` to use the Robe method, or `ce` for compositional embeddings.

## Example runs

[Compositional Embeddings](https://arxiv.org/abs/1909.02107):
<pre>
$ py examples/movielens.py --method ce
Unique users: 943, Unique items: 1682, #params: 6400
Epoch: 0, Train Loss: 0.482, Validation Loss: 0.461
Epoch: 1, Train Loss: 0.433, Validation Loss: 0.436
Epoch: 2, Train Loss: 0.398, Validation Loss: <b>0.441</b>
Epoch: 3, Train Loss: 0.383, Validation Loss: 0.447
Epoch: 4, Train Loss: 0.374, Validation Loss: 0.453
Epoch: 5, Train Loss: 0.368, Validation Loss: 0.458
Epoch: 6, Train Loss: 0.363, Validation Loss: 0.464
Epoch: 7, Train Loss: 0.36, Validation Loss: 0.464
Epoch: 8, Train Loss: 0.357, Validation Loss: 0.469
Epoch: 9, Train Loss: 0.354, Validation Loss: 0.47
</pre>

[Robe Embeddings](https://proceedings.mlsys.org/paper_files/paper/2022/file/1eb34d662b67a14e3511d0dfd78669be-Paper.pdf):
<pre>
$ py examples/movielens.py --method robe
Unique users: 943, Unique items: 1682, #params: 6400
Epoch: 0, Train Loss: 0.482, Validation Loss: 0.465
Epoch: 1, Train Loss: 0.451, Validation Loss: 0.444
Epoch: 2, Train Loss: 0.395, Validation Loss: 0.435
Epoch: 3, Train Loss: 0.37, Validation Loss: <b>0.434</b>
Epoch: 4, Train Loss: 0.357, Validation Loss: 0.436
Epoch: 5, Train Loss: 0.349, Validation Loss: 0.438
Epoch: 6, Train Loss: 0.342, Validation Loss: 0.442
Epoch: 7, Train Loss: 0.338, Validation Loss: 0.444
Epoch: 8, Train Loss: 0.334, Validation Loss: 0.447
Epoch: 9, Train Loss: 0.33, Validation Loss: 0.449
</pre>

Clustered Compositional Embeddings (this repo):
<pre>
$ py examples/movielens.py --method cce
Unique users: 943, Unique items: 1682, #params: 6400
Epoch: 0, Train Loss: 0.467, Validation Loss: 0.428
Clustering...
Epoch: 1, Train Loss: 0.402, Validation Loss: 0.409
Clustering...
Epoch: 2, Train Loss: 0.384, Validation Loss: <b>0.409</b>
Clustering...
Epoch: 3, Train Loss: 0.372, Validation Loss: 0.413
Clustering...
Epoch: 4, Train Loss: 0.365, Validation Loss: 0.417
Clustering...
Epoch: 5, Train Loss: 0.359, Validation Loss: 0.421
Clustering...
Epoch: 6, Train Loss: 0.354, Validation Loss: 0.426
Clustering...
Epoch: 7, Train Loss: 0.351, Validation Loss: 0.428
Epoch: 8, Train Loss: 0.335, Validation Loss: 0.436
Epoch: 9, Train Loss: 0.324, Validation Loss: 0.446
</pre>

Hybrid CCE and Robe:
<pre>
$ py examples/movielens.py --method cce_robe
Unique users: 943, Unique items: 1682, #params: 6400
Epoch: 0, Train Loss: 0.484, Validation Loss: 0.464
Clustering...
Epoch: 1, Train Loss: 0.421, Validation Loss: 0.406
Clustering...
Epoch: 2, Train Loss: 0.37, Validation Loss: <b>0.397</b>
Clustering...
Epoch: 3, Train Loss: 0.356, Validation Loss: 0.4
Clustering...
Epoch: 4, Train Loss: 0.348, Validation Loss: 0.401
Clustering...
Epoch: 5, Train Loss: 0.343, Validation Loss: 0.405
Clustering...
Epoch: 6, Train Loss: 0.339, Validation Loss: 0.406
Clustering...
Epoch: 7, Train Loss: 0.335, Validation Loss: 0.411
Epoch: 8, Train Loss: 0.323, Validation Loss: 0.416
Epoch: 9, Train Loss: 0.313, Validation Loss: 0.425
</pre>

Hashing method:
<pre>
$ py examples/movielens.py --method simple --epochs 10
Unique users: 943, Unique items: 1682, #params: 6400
Epoch: 0, Train Loss: 0.474, Validation Loss: 0.467
Epoch: 1, Train Loss: 0.459, Validation Loss: 0.463
Epoch: 2, Train Loss: 0.454, Validation Loss: 0.461
Epoch: 3, Train Loss: 0.449, Validation Loss: <b>0.458</b>
Epoch: 4, Train Loss: 0.444, Validation Loss: 0.458
Epoch: 5, Train Loss: 0.439, Validation Loss: 0.457
Epoch: 6, Train Loss: 0.436, Validation Loss: 0.459
Epoch: 7, Train Loss: 0.433, Validation Loss: 0.458
Epoch: 8, Train Loss: 0.43, Validation Loss: 0.458
Epoch: 9, Train Loss: 0.428, Validation Loss: 0.461
</pre>

Full size table:
<pre>
$ py examples/movielens.py --method full --epochs 10
Unique users: 943, Unique items: 1682, #params: 53824
Epoch: 0, Train Loss: 0.49, Validation Loss: 0.465
Epoch: 1, Train Loss: 0.416, Validation Loss: <b>0.415</b>
Epoch: 2, Train Loss: 0.272, Validation Loss: 0.458
Epoch: 3, Train Loss: 0.176, Validation Loss: 0.56
Epoch: 4, Train Loss: 0.118, Validation Loss: 0.682
Epoch: 5, Train Loss: 0.0782, Validation Loss: 0.824
Epoch: 6, Train Loss: 0.0512, Validation Loss: 0.986
Epoch: 7, Train Loss: 0.0328, Validation Loss: 1.18
Epoch: 8, Train Loss: 0.0206, Validation Loss: 1.42
Epoch: 9, Train Loss: 0.0126, Validation Loss: 1.84
</pre>

Clustered Compositional Embeddings without clustering:
<pre>
$ py examples/movielens.py --method cce --epochs 10 --last-cluster 0
Unique users: 943, Unique items: 1682, #params: 6400
Epoch: 0, Train Loss: 0.467, Validation Loss: 0.428
Epoch: 1, Train Loss: 0.401, Validation Loss: <b>0.42</b>
Epoch: 2, Train Loss: 0.376, Validation Loss: 0.427
Epoch: 3, Train Loss: 0.357, Validation Loss: 0.438
Epoch: 4, Train Loss: 0.344, Validation Loss: 0.446
Epoch: 5, Train Loss: 0.333, Validation Loss: 0.455
Epoch: 6, Train Loss: 0.325, Validation Loss: 0.463
Epoch: 7, Train Loss: 0.319, Validation Loss: 0.468
Epoch: 8, Train Loss: 0.313, Validation Loss: 0.476
Epoch: 9, Train Loss: 0.309, Validation Loss: 0.484
</pre>

[Tensor Train (TT-Rec)](https://arxiv.org/abs/2101.11714):
Clustered Compositional Embeddings without clustering:
<pre>
$ py examples/movielens.py --method tt
Unique users: 943, Unique items: 1682, #params: 6400
Device: cpu
Notice: Using 5616 params, rather than 6400. rank=13, hash.range=6
Notice: Using 5544 params, rather than 6400. rank=12, hash.range=7
Epoch: 0, Time: 7.32s, Train Loss: 0.486, Validation Loss: 0.444
Epoch: 1, Time: 4.68s, Train Loss: 0.415, Validation Loss: 0.406
Epoch: 2, Time: 4.33s, Train Loss: 0.392, Validation Loss: 0.393
Epoch: 3, Time: 4.32s, Train Loss: 0.38, Validation Loss: 0.391
Epoch: 4, Time: 4.09s, Train Loss: 0.372, Validation Loss: <b>0.388</b>
Epoch: 5, Time: 5.01s, Train Loss: 0.367, Validation Loss: 0.388
Epoch: 6, Time: 4.57s, Train Loss: 0.361, Validation Loss: 0.388
Epoch: 7, Time: 4.4s, Train Loss: 0.357, Validation Loss: 0.389
Epoch: 8, Time: 4.21s, Train Loss: 0.352, Validation Loss: 0.39
Epoch: 9, Time: 4.22s, Train Loss: 0.348, Validation Loss: 0.391
</pre>

## Citation

```
@article{tsang2022clustering,
  title={Clustering Embedding Tables, Without First Learning Them},
  author={Tsang, Henry Ling-Hei and Ahle, Thomas Dybdahl},
  journal={arXiv preprint arXiv:2210.05974},
  year={2022}
}
```

## TODO

Some algorithms we'd like to add to this repository in the future:
- [Hash Embeddings](https://arxiv.org/abs/1709.03933)
- [DHE](https://arxiv.org/abs/2010.10784)
- [HashedNet](https://arxiv.org/abs/1504.04788)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clustering-embedding-tables-without-first/click-through-rate-prediction-on-criteo)](https://paperswithcode.com/sota/click-through-rate-prediction-on-criteo?p=clustering-embedding-tables-without-first)
