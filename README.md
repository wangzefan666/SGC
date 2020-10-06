## Simplifying Graph Convolutional Networks

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Fork from [Tiiiger's repo](https://github.com/Tiiiger/SGC)

### Modification
- replace Adam loss function in citation.py with L-BFGS which improve performance on validation set and test set.
- replace `adj = adj + adj.T + sp.eye(adj.shape[0])` in utils.py with `adj = adj + adj.T` beacause eye of adj would be added the second time when calling adj_normalizer.  

### Experiment
on a GTX 1050 Ti (Cora, Citeseer, Pubmed) and a Intel G4600(Reddit), the results are as follows (They are different from the original results which are all run on a GTX 1050Ti).

Dataset | Metric | Training Time | lr | epoch
:------:|:------:|:-----------:|:------:|:------:
Cora    | Acc: 80.60 %     | 5.71s | 0.055 | 10
Citeseer| Acc: 72.50 %     | 1.49s | 0.05 | 4
Pubmed  | Acc: 79.20 %     | 2.10s | 0.01 | 5
Reddit  | F1:  94.97     | 52.43s  | 1 | 2


### Dependencies
Our implementation works with PyTorch>=1.0.0 Install other dependencies: `$ pip install -r requirement.txt`

### Data
citation network datasets  is provided under `data/`, which corresponds to [the public data splits](https://github.com/tkipf/gcn/tree/master/gcn/data).
Due to space limit, please download reddit dataset from [FastGCN](https://github.com/matenure/FastGCN/issues/9) and put `reddit_adj.npz`, `reddit.npz` under `data/`.

### Usage
Citation Networks: We tune the only hyperparameter, weight decay, with hyperopt and put the resulting hyperparameter under `SGC-tuning`. 
See `tuning.py` for more details on hyperparameter optimization.
```
$ python citation.py --dataset cora --epochs {} --lr {}
$ python citation.py --dataset citeseer --epochs {} --lr {}
$ python citation.py --dataset pubmed --epochs {} --lr {}
```

Reddit:
```
$ python reddit.py --test
```

### citation
If you find this repo useful, please cite:
```
@InProceedings{pmlr-v97-wu19e,
  title = 	 {Simplifying Graph Convolutional Networks},
  author = 	 {Wu, Felix and Souza, Amauri and Zhang, Tianyi and Fifty, Christopher and Yu, Tao and Weinberger, Kilian},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {6861--6871},
  year = 	 {2019},
  publisher = 	 {PMLR},
}
```