# VRKG4Rec: Virtual Relational Knowledge Graph for Recommendation

This is the PyTorch Implementation for the paper VRKG4Rec (WSDM'23):

> Lingyun Lu, Bang Wang, Zizhuo Zhang, Shenghao Liu and Han Xu. VRKG4Rec: Virtual Relational Knowledge Graph for Recommendation.  [Paper in arXiv](https://arxiv.org/). In WSDM'2023.

## Introduction

Virtual Relational Knowledge Graph for Recommendation (VRKG4Rec) is a knowledge-aware recommendation framework, which explicitly distinguishes the influence of different relations for item representation learning and design a
local weighted smoothing (LWS) mechanism for user and item encoding.

## Citation 

If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{
}
```

## Environment Requirement

The code has been tested running under Python 3.8.0. The required packages are as follows:

- pytorch == 1.10.1
- networkx == 2.5.1
- numpy == 1.22.4
- pandas == 1.4.3
- scikit-learn == 1.1.1
- scipy == 1.7.0
- torch == 1.9.0
- torch-cluster == 1.5.9
- torch-scatter == 2.0.9
- torch-sparse == 0.6.12

## Usage

The instruction of commands has been clearly stated in the codes (see the parser function in utils/parser.py). 

- Last-fm dataset

```
python main.py --dataset last-fm --lr 0.0001 --n_virtual 3 --context_hops 2 --n_iter 3
```

- MovieLens dataset

```
python main.py --dataset MovieLens --lr 0.0001 --n_virtual 3 --context_hops 2 --n_iter 3
```


## Dataset

We provide three processed datasets: Last-FM and MovieLens.

- You can find the full version of recommendation datasets via [Last-FM](https://grouplens.org/datasets/hetrec-2011/) and [MovieLens](https://grouplens.org/datasets/movielens/1m/).
- We follow the previous study to preprocess the datasets.

|                       |               |     Last-FM | MovieLens |
| :-------------------: | :------------ | ----------: | --------: |
| User-Item Interaction | #Users        |       1,872 |     6,036 |
|                       | #Items        |       3,915 |     2,347 |
|                       | #Interactions |      42,346 |   753,772 |
|    Knowledge Graph    | #Entities     |       9,366 |     6,729 |
|                       | #Relations    |          60 |         7 |
|                       | #Triplets     |      15,518 |    20,195 |


