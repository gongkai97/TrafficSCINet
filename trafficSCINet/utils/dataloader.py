import torch.utils.data as torch_data
import numpy as np
import torch
import pandas as pd

def normalized(data, norm_statistic=None):
    if not norm_statistic:
        norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
    mean = norm_statistic['mean']
    std = norm_statistic['std']
    std = [1 if i == 0 else i for i in std]
    data = (data - mean) / std
    norm_statistic['std'] = std
    return data, norm_statistic

def de_normalized(data, norm_statistic):
    if not norm_statistic:
        norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
    mean = norm_statistic['mean']
    std = norm_statistic['std']
    std = [1 if i == 0 else i for i in std]
    data = data * std + mean
    return data

class ForecastDataset(torch_data.Dataset):
    def __init__(self, df, window_size, horizon, norm_statistic=None, interval=1):
        self.window_size = window_size
        self.interval = interval
        self.horizon = horizon
        self.norm_statistic = norm_statistic
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        self.data, _ = normalized(self.data, norm_statistic)

    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - self.window_size
        train_data = self.data[lo: hi]
        target_data = self.data[hi:hi + self.horizon]
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx

class ForecastTestDataset(torch_data.Dataset):
    def __init__(self, df, window_size, horizon, norm_statistic=None, interval=1):
        self.window_size = window_size # 12
        self.interval = interval  #1
        self.horizon = horizon
        self.norm_statistic = norm_statistic
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        self.data, _ = normalized(self.data, norm_statistic)

    def __getitem__(self, index):
        hi = self.x_end_idx[index] #12
        lo = hi - self.window_size #0
        train_data = self.data[lo: hi] #0:12
        target_data = self.data[hi:hi + self.horizon] #12:24
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * 12] for j in range((len(x_index_set)) // 12)]
        return x_end_idx