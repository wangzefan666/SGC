import numpy as np
import scipy.sparse as sp


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])  # A + I
    adj = sp.coo_matrix(adj)  # tranform sparse to dense. = adj.tocoo()
    row_sum = np.array(adj.sum(1))  # D + I
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()  # (D + I)^-1/2
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # prevent division by zero
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # transform vec to diags
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)


def fetch_normalization(type):
    switcher = {
        'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    }
    # dict.get(key, default)
    # default -- 如果指定键的值不存在时，返回该默认值。
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
