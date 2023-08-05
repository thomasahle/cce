# Clustered Compositional Embeddings

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
Epoch: 0, Train Loss: 0.477, Validation Loss: 0.442
Epoch: 1, Train Loss: 0.428, Validation Loss: <b>0.438</b>
Epoch: 2, Train Loss: 0.416, Validation Loss: 0.441
Epoch: 3, Train Loss: 0.407, Validation Loss: 0.448
Epoch: 4, Train Loss: 0.4, Validation Loss: 0.454
Epoch: 5, Train Loss: 0.395, Validation Loss: 0.458
Epoch: 6, Train Loss: 0.392, Validation Loss: 0.462
Epoch: 7, Train Loss: 0.389, Validation Loss: 0.464
Epoch: 8, Train Loss: 0.387, Validation Loss: 0.469
Epoch: 9, Train Loss: 0.384, Validation Loss: 0.472
</pre>

[Robe Embeddings](https://proceedings.mlsys.org/paper_files/paper/2022/file/1eb34d662b67a14e3511d0dfd78669be-Paper.pdf):
<pre>
$ py examples/movielens.py --method robe
Unique users: 943, Unique items: 1682, #params: 6400
Epoch: 0, Train Loss: 0.488, Validation Loss: 0.465
Epoch: 1, Train Loss: 0.439, Validation Loss: 0.43
Epoch: 2, Train Loss: 0.384, Validation Loss: <b>0.427</b>
Epoch: 3, Train Loss: 0.363, Validation Loss: 0.429
Epoch: 4, Train Loss: 0.352, Validation Loss: 0.431
Epoch: 5, Train Loss: 0.345, Validation Loss: 0.435
Epoch: 6, Train Loss: 0.339, Validation Loss: 0.437
Epoch: 7, Train Loss: 0.334, Validation Loss: 0.441
Epoch: 8, Train Loss: 0.331, Validation Loss: 0.44
Epoch: 9, Train Loss: 0.327, Validation Loss: 0.445
</pre>

[Clustered Compositional Embeddings](https://arxiv.org/abs/2210.05974) (this paper):
<pre>
$ py examples/movielens.py --method cce
Unique users: 943, Unique items: 1682, #params: 6400
Epoch: 0, Train Loss: 0.462, Validation Loss: 0.431
Clustering...
Epoch: 1, Train Loss: 0.412, Validation Loss: 0.419
Clustering...
Epoch: 2, Train Loss: 0.4, Validation Loss: <b>0.417</b>
Clustering...
Epoch: 3, Train Loss: 0.394, Validation Loss: 0.423
Clustering...
Epoch: 4, Train Loss: 0.388, Validation Loss: 0.422
Clustering...
Epoch: 5, Train Loss: 0.384, Validation Loss: 0.424
Clustering...
Epoch: 6, Train Loss: 0.382, Validation Loss: 0.427
Clustering...
Epoch: 7, Train Loss: 0.379, Validation Loss: 0.428
Epoch: 8, Train Loss: 0.365, Validation Loss: 0.434
Epoch: 9, Train Loss: 0.356, Validation Loss: 0.44
</pre>
