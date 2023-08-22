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

## Example results

For a simple 32 dim model, varying the number of parameters:
<img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/ml-1m.png" alt="ml-1m" width="70%"/>

Similarly a 16 dimensional model on Criteo Kaggle:
<img src="https://raw.githubusercontent.com/thomasahle/cce/main/results/criteo.png" alt="Criteo Kaggle" width="70%"/>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clustering-embedding-tables-without-first/click-through-rate-prediction-on-criteo)](https://paperswithcode.com/sota/click-through-rate-prediction-on-criteo?p=clustering-embedding-tables-without-first)
