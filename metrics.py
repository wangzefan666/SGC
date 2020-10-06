from sklearn.metrics import f1_score


def accuracy(output, labels):
    preds = output.max(1)[1]
    correct = preds.eq(labels).double()  # transform bool to double
    correct = correct.sum().item()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().numpy()  # move to CPU then transform to np
    labels = labels.cpu().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro  # 整体F1, 类别平均F1
