import math
import torch.nn.functional as F
from torch import nn
import torch
import argparse
import numpy as np

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))


class Interactor(nn.Module):
    def __init__(self, in_planes, splitting=True,
                 kernel=5, dropout=0.5, groups=1, hidden_size=1, INN=True):
        super(Interactor, self).__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1  # by default: stride==1
            pad_r = self.dilation * (self.kernel_size) // 2 + 1  # by default: stride==1

        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1  # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
        self.splitting = splitting
        self.split = Splitting()

        modules_psi = []
        modules_phi = []
        prev_size = 1
        size_hidden = self.hidden_size

        modules_phi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups=self.groups),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups=self.groups),
            nn.Tanh()
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)

    def forward(self, x):

        (x_even, x_odd) = self.split(x)
        x_even = x_even.permute(0, 2, 1)
        x_odd = x_odd.permute(0, 2, 1)

        a = x_even
        b = x_odd

        x_even_update = a + self.phi(x_odd)
        x_odd_update = b + self.psi(x_even)

        return (x_even_update, x_odd_update)


class InteractorLevel(nn.Module):
    def __init__(self, in_planes, kernel, dropout, groups, hidden_size, INN):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(in_planes=in_planes, splitting=True,
                                kernel=kernel, dropout=dropout, groups=groups, hidden_size=hidden_size, INN=INN)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.level(x)
        return (x_even_update, x_odd_update)


class LevelSCINet(nn.Module):
    def __init__(self, in_planes, kernel_size, dropout, groups, hidden_size, INN):
        super(LevelSCINet, self).__init__()
        self.interact = InteractorLevel(in_planes=in_planes, kernel=kernel_size, dropout=dropout, groups=groups,
                                        hidden_size=hidden_size, INN=INN)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1)  # even: B, T, D odd: B, T, D


class SCINet_Tree(nn.Module):
    def __init__(self, in_planes, current_level, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.current_level = current_level

        self.workingblock = LevelSCINet(
            in_planes=in_planes,
            kernel_size=kernel_size,
            dropout=dropout,
            groups=groups,
            hidden_size=hidden_size,
            INN=INN)

        if current_level != 0:
            self.SCINet_Tree_odd = SCINet_Tree(in_planes, current_level - 1, kernel_size, dropout, groups, hidden_size,
                                               INN)
            self.SCINet_Tree_even = SCINet_Tree(in_planes, current_level - 1, kernel_size, dropout, groups, hidden_size,
                                                INN)

    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)  # L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len:
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_, 0).permute(1, 0, 2)  # B, L, D

    def forward(self, x):
        x_even_update, x_odd_update = self.workingblock(x)
        if self.current_level == 0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))

class EncoderTree(nn.Module):
    def __init__(self, in_planes, num_levels, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.levels = num_levels
        self.SCINet_Tree = SCINet_Tree(
            in_planes=in_planes,
            current_level=num_levels - 1,
            kernel_size=kernel_size,
            dropout=dropout,
            groups=groups,
            hidden_size=hidden_size,
            INN=INN)

    def forward(self, x):
        x = self.SCINet_Tree(x)
        return x

class SCINet(nn.Module):
    def __init__(self, output_len, input_len, input_dim=170, hid_size=1, num_stacks=1,
                 num_levels=2, num_decoder_layer=1, concat_len=0, groups=1, kernel=5, dropout=0.5,
                 input_len_seg=0, modified=True):
        super(SCINet, self).__init__()

        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.groups = groups

        self.modified = modified
        self.kernel_size = kernel
        self.dropout = dropout
        # self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout

        self.concat_len = concat_len
        self.num_decoder_layer = num_decoder_layer

        self.blocks1 = EncoderTree(
            in_planes=self.input_dim,
            num_levels=self.num_levels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            groups=self.groups,
            hidden_size=self.hidden_size,
            INN=modified)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)
        self.overlap_len = self.input_len // 4
        self.div_len = self.input_len // 6

    def forward(self, x):

        assert self.input_len % (np.power(2,self.num_levels)) == 0  # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)

        res1 = x
        x = self.blocks1(x)
        x = self.projection1(x + res1)

        return x


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, adj):
        out = [x]

        x1 = self.nconv(x, adj)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, adj)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class AGCN(nn.Module):
    def __init__(self, aptinit, args, supports=None):
        super(AGCN, self).__init__()
        if supports is None:
            self.supports = []
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        self.supports_len += 1
        residual_channels = 24
        dilation_channels = 12
        dropout = 0.3
        self.gconv = nn.ModuleList()
        self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=(1, 1), bias=True)
        self.aptinit = aptinit.to(args.device)
        self.adp = nn.Parameter(aptinit, requires_grad=True)

        self.bn = nn.ModuleList()
        self.bn.append(nn.BatchNorm2d(residual_channels))

    def forward(self, x):
        x = torch.unsqueeze(x, -1)
        x = self.gconv[0](x, F.relu(self.adp))
        x = self.bn[0](x)
        x = self.end_conv(x)
        return x[:, :, :, 0]


class TrafficSCINet(nn.Module):
    def __init__(self, adj, args):
        super(TrafficSCINet, self).__init__()
        self.gcn = AGCN(adj, args)
        if args.dataset == 'PEMS03':
            self.input_dim = 358
        elif args.dataset == 'PEMS04':
            self.input_dim = 307
        elif args.dataset == 'PEMS07':
            self.input_dim = 883
        elif args.dataset == 'PEMS08':
            self.input_dim = 170
        self.scinet = SCINet(output_len=args.horizon,
                             input_len=args.window_size,
                             input_dim=self.input_dim,
                             hid_size=args.hidden_size,
                             num_stacks=args.stacks,
                             num_levels=args.levels,
                             concat_len=args.concat_len,
                             groups=args.groups,
                             kernel=args.kernel,
                             dropout=args.dropout,
                             modified=True)

    def forward(self, x):

        gcn_output = self.gcn(x)
        outputs = self.scinet(x + gcn_output)
        return outputs
