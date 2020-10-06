import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter
from models import GCN

# 加载参数
args = get_citation_args()

# 模型微调
if args.tuned:
    if args.model == "SGC":
        # 读取微调超参数 - 权重衰减
        with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented

# 设置随机种子，固定结果
set_seed(args.seed, args.cuda)

# 邻接矩阵(归一化)，特征，标签，训练集，验证集，测试集
adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda)

# 模型 SGC or GCN
model = get_model(args.model, features.size(1), labels.max().item() + 1, args.hidden, args.dropout, args.cuda)

# 预计算 (S^K * X)
if args.model == "SGC":
    features, precompute_time = sgc_precompute(features, adj, args.degree)
    print("pre-compute time: {:.4f}s".format(precompute_time))


def train_regression(model,
                     adj, features, labels,
                     idx_train, idx_val,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr):
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.LBFGS(model.parameters(), lr=lr)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()

        def closure():
            optimizer.zero_grad()
            if isinstance(model, GCN):
                output = model(features, adj)
                output = output[idx_train]
            else:
                output = model(features[idx_train])
            loss_train = F.cross_entropy(output, labels[idx_train])
            loss_train.backward()
            return loss_train

        optimizer.step(closure)  # LBFGS专用

        with torch.no_grad():
            model.eval()
            if isinstance(model, GCN):
                output = model(features, adj)
                output = output[idx_val]
            else:
                output = model(features[idx_val])
            acc_val = accuracy(output, labels[idx_val])
            print("training val acc:{:.4f}".format(acc_val))

    train_time = perf_counter() - t
    return model, acc_val, train_time


def test_regression(model, adj, features, labels, idx_test):
    with torch.no_grad():
        model.eval()
        if isinstance(model, GCN):
            return accuracy(model(features, adj)[idx_test], labels[idx_test])
        else:
            return accuracy(model(features[idx_test]), labels[idx_test])


# if args.model == "SGC":
model, acc_val, train_time = train_regression(model, adj, features, labels,
                                              idx_train, idx_val,
                                              args.epochs, args.weight_decay, args.lr)
acc_test = test_regression(model, adj, features, labels, idx_test)

print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("train time: {:.4f}s".format(train_time))
