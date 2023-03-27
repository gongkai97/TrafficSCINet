import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from data_preprocess.data_preprocess import get_data, calculate_laplacian_with_self_loop
from utils.dataloader import de_normalized
from utils.math import evaluate
from model.TrafficSCINet import TrafficSCINet

import os
import time
import warnings
from datetime import datetime
import argparse
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='SCINet on pems datasets')

### -------  dataset settings --------------
parser.add_argument('--dataset', type=str, default='PEMS08', choices=['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'])

### -------  device settings --------------
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--device', type=str, default='cuda:0')

### -------  input/output length settings --------------
parser.add_argument('--window_size', type=int, default=12)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--concat_len', type=int, default=0)

parser.add_argument('--train_length', type=float, default=6)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=2)

### -------  training settings --------------
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--validate_freq', type=int, default=1)

parser.add_argument('--epoch', type=int, default=80)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)

parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model_name', type=str, default='TrafficSCINet')

### -------  model settings --------------
parser.add_argument('--hidden-size', default=0.1, type=float, help='hidden channel scale of module')
parser.add_argument('--INN', default=0, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=5, type=int, help='kernel size for the first layer')
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--levels', type=int, default=2)
parser.add_argument('--stacks', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--RIN', type=bool, default=False)
parser.add_argument('--decompose', type=bool,default=False)

args = parser.parse_args(args=[])

def adjust_learning_rate(optimizer, epoch):

    lr_adjust = {epoch: 0.001 * (0.95 ** (epoch // 1))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    else:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
    return lr

class Exp_pems():
    def __init__(self, args):
        super(Exp_pems, self).__init__()
        self.model = self._build_model().cuda()

    def _build_model(self):
        if args.dataset == 'PEMS03':
            self.input_dim = 358
        elif args.dataset == 'PEMS04':
            self.input_dim = 307
        elif args.dataset == 'PEMS07':
            self.input_dim = 883
        elif args.dataset == 'PEMS08':
            self.input_dim = 170
        adj = calculate_laplacian_with_self_loop(args)
        model = TrafficSCINet(adj, args)

        print(model)
        return model

    def _select_optimizer(self):
        my_optim = torch.optim.Adam(params=self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        return my_optim

    def inference(self, model, dataloader, node_cnt, window_size, horizon):
        forecast_set = []
        target_set = []
        # input_set = []
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, target) in enumerate(dataloader):
                inputs = inputs.cuda()
                target = target.cuda()
                # input_set.append(inputs.detach().cpu().numpy())
                step = 0
                forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
                while step < horizon:
                    forecast_result = self.model(inputs)

                    len_model_output = forecast_result.size()[1]
                    forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                        forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()

                    step += min(horizon - step, len_model_output)

                forecast_set.append(forecast_steps)
                target_set.append(target.detach().cpu().numpy())
        # 返回整个valid或test数据集的预测和标签结果
        return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)

    def validate(self, model, epoch, forecast_loss, dataloader, statistic,
                 node_cnt, window_size, horizon, test=False):
        print("===================Validate Normal=========================")
        forecast_norm, target_norm = self.inference(model, dataloader, node_cnt, window_size, horizon)
        forecast = de_normalized(forecast_norm, statistic)
        target = de_normalized(target_norm, statistic)

        forecast_norm = torch.from_numpy(forecast_norm).float()
        target_norm = torch.from_numpy(target_norm).float()
        # loss = forecast_loss(forecast_norm, target_norm)
        score = evaluate(target, forecast)
        score_final_detail = evaluate(target, forecast, by_step=True)
        print('by each step: MAPE & MAE & RMSE', score_final_detail)

        if test:
            print(f'TEST: RAW : MAE {score[1]:7.2f};MAPE {score[0]:7.2f}; RMSE {score[2]:7.2f}.')
        else:
            print(f'VAL: RAW : MAE {score[1]:7.2f}; RMSE {score[2]:7.2f}.')

        return dict(mae=score[1], mape=score[0], rmse=score[2])

    def train(self):
        my_optim = self._select_optimizer()
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
        test_loader, train_loader, valid_loader, node_cnt, test_normalize_statistic, val_normalize_statistic = get_data(args)
        forecast_loss = nn.L1Loss().cuda()
        best_validate_mae = np.inf
        best_test_mae = np.inf
        validate_score_non_decrease_count = 0
        writer = None

        performance_metrics = {}
        epoch_start = 0

        for epoch in range(epoch_start, args.epoch):
            lr = adjust_learning_rate(my_optim, epoch)
            epoch_start_time = time.time()
            self.model.train()
            loss_total = 0
            cnt = 0

            for i, (inputs, target) in enumerate(train_loader):
                inputs = inputs.cuda()
                target = target.cuda()
                self.model.zero_grad()

                forecast = self.model(inputs)
                loss = forecast_loss(forecast, target)

                cnt += 1
                loss.backward()
                my_optim.step()
                loss_total += float(loss)

            print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} '.format(epoch, (
                        time.time() - epoch_start_time), loss_total / cnt))

            if (epoch + 1) % args.exponential_decay_step == 0:
                my_lr_scheduler.step()
            if (epoch + 1) % args.validate_freq == 0:
                is_best_for_now = False
                print('------ validate on data: VALIDATE ------')
                performance_metrics = self.validate(self.model, epoch, forecast_loss, valid_loader,val_normalize_statistic,
                                                    node_cnt, args.window_size, args.horizon, test=False)
                test_metrics = self.validate(self.model, epoch, forecast_loss, test_loader,test_normalize_statistic,
                                             node_cnt, args.window_size, args.horizon, test=True)
                if best_validate_mae > performance_metrics['mae']:
                    best_validate_mae = performance_metrics['mae']
                    is_best_for_now = True
                    validate_score_non_decrease_count = 0
                    print('got best validation result:', performance_metrics, test_metrics)
                else:
                    validate_score_non_decrease_count += 1
                if best_test_mae > test_metrics['mae']:
                    best_test_mae = test_metrics['mae']
                    print('got best test result:', test_metrics)

        return performance_metrics, test_normalize_statistic


if __name__ == '__main__':

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    Exp=Exp_pems
    exp=Exp(args)

    before_train = datetime.now().timestamp()
    print("===================Normal-Start=========================")
    _, normalize_statistic = exp.train()
    after_train = datetime.now().timestamp()
    print(f'Training took {(after_train - before_train) / 60} minutes')
    print("===================Normal-End=========================")