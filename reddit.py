from time import perf_counter
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_reddit_data, sgc_precompute, set_seed
from metrics import f1
from models import SGC

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--test', action='store_true', default=False,
                    help='use test set or val set.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2,
                    help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                    choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN',
                             'AugNormAdj', 'NormAdj', 'RWalk', 'AugRWalk', 'NoNorm'],
                    help='Normalization method for the adjacency matrix.')
parser.add_argument('--model', type=str, default="SGC",
                    help='model to use.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def train_regression(model, train_features, train_labels, epochs):
    optimizer = optim.LBFGS(model.parameters(), lr=1)  # 全批量效果好
    # optimizer = optim.Adam(model.parameters(), lr=1)  # 小批量效果好
    model.train()

    def closure():
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        return loss_train

    t = perf_counter()
    for epoch in range(epochs):
        optimizer.step(closure)  # LBFGS专用
    train_time = perf_counter() - t
    return model, train_time


def test_regression(model, test_features, test_labels):
    with torch.no_grad():
        model.eval()
        return f1(model(test_features), test_labels)


# 随机种子固定结果
set_seed(args.seed, args.cuda)

# 邻接矩阵(全), 特征, 标签, 训练集，验证集，测试集
adj, features, labels, idx_train, idx_val, idx_test = \
    load_reddit_data(normalization=args.normalization, cuda=args.cuda)
print("Finished data loading.")


if args.model == 'SGC':
    model = SGC(features.size(1), labels.max().item() + 1)
    if args.cuda:
        model.cuda()
    # precompute
    processed_features, precompute_time = sgc_precompute(features, adj, args.degree)
    # train
    train_features = processed_features[idx_train]
    train_labels = labels[idx_train]
    model, train_time = train_regression(model, train_features, train_labels, args.epochs)
    # test
    test_features = processed_features[idx_test if args.test else idx_val]
    test_labels = labels[idx_test if args.test else idx_val]
    test_f1, _ = test_regression(model, test_features, test_labels)

    print("pre-compute time: {:.4f}s, Total train Time: {:.4f}s, {} F1: {:.4f}"
          .format(precompute_time, train_time + precompute_time, "Test" if args.test else "Val", test_f1))
else:
    raise NotImplemented




