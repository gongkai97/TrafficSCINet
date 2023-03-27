import os
import numpy as np
import pandas as pd
import torch
from utils.dataloader import normalized, de_normalized, ForecastDataset, ForecastTestDataset
from torch.utils.data import DataLoader

def get_data(args):
    data_file = os.path.join('dataset//PEMS//' + args.dataset + '.npz')
    print('data file:', data_file)
    data = np.load(data_file, allow_pickle=True)
    data = data['data'][:, :, 0]  # 一共有三个指标，只选取第一个
    train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
    valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
    test_ratio = 1 - train_ratio - valid_ratio
    train_data = data[:int(train_ratio * len(data))]
    valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
    test_data = data[int((train_ratio + valid_ratio) * len(data)):]
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')
    if len(test_data) == 0:
        raise Exception('Cannot organize enough test data')

    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0)
    train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    val_mean = np.mean(valid_data, axis=0)
    val_std = np.std(valid_data, axis=0)
    val_normalize_statistic = {"mean": val_mean.tolist(), "std": val_std.tolist()}
    test_mean = np.mean(test_data, axis=0)
    test_std = np.std(test_data, axis=0)
    test_normalize_statistic = {"mean": test_mean.tolist(), "std": test_std.tolist()}

    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                norm_statistic=train_normalize_statistic)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                norm_statistic=val_normalize_statistic)
    test_set = ForecastTestDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                                   norm_statistic=test_normalize_statistic)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    node_cnt = train_data.shape[1]
    return test_loader, train_loader, valid_loader, node_cnt, test_normalize_statistic, val_normalize_statistic


def calculate_laplacian_with_self_loop(args):
    dis = pd.read_csv('dataset/PEMS/' + args.dataset + '.csv')
    d = [[0 for _ in range(170)] for _ in range(170)]
    for i in range(len(dis)):
        f = int(dis.iloc[i]['from'])
        t = int(dis.iloc[i]['to'])
        c = (dis.iloc[i]['cost'])
        d[f][t] = d[t][f] = c
    matrix = torch.FloatTensor(d)

    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian


