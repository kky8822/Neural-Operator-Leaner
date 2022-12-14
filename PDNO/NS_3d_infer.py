import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from timeit import default_timer

import matplotlib.pyplot as plt
from utils import MatReader, rel_error

from tqdm import tqdm
import sys
import argparse
import os
import shutil

import pickle

torch.manual_seed(0)
np.random.seed(0)


def save_pickle(var, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(var, f)


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class Net(nn.Module):
    def __init__(self, d_in, d_out, act, num_layer=2, num_hidden=64):
        super().__init__()
        self.linear_in = nn.Linear(d_in, num_hidden)
        self.hidden = nn.ModuleList([nn.Linear(num_hidden, num_hidden) for i in range(num_layer)])
        self.linear_out = nn.Linear(num_hidden, d_out)
        act = act.lower()
        if act == "tanh":
            self.activation = torch.tanh
        if act == "gelu":
            self.activation = F.gelu

    def forward(self, x):
        out = self.linear_in(x)
        out = F.gelu(out)
        for layer in self.hidden:
            out = layer(out)
            out = self.activation(out)
        return self.linear_out(out)


class SpectralConv3d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer, num_hidden, symbol_act, net_out):
        super(SpectralConv3d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net_out = net_out

        self.scale = 1 / (in_channels * out_channels)
        self.net_real = Net(
            d_in=3,
            d_out=in_channels * out_channels,
            num_layer=num_layer,
            num_hidden=num_hidden,
            act=symbol_act,
        )
        self.net_imag = Net(
            d_in=3,
            d_out=in_channels * out_channels,
            num_layer=num_layer,
            num_hidden=num_hidden,
            act=symbol_act,
        )
        if net_out:
            self.net_out = Net(
                d_in=3,
                d_out=in_channels * out_channels,
                num_layer=num_layer,
                num_hidden=num_hidden,
                act=symbol_act,
            )

    def _weights(self, shape):
        grid = self.get_grid_freq(shape)
        out_real = self.net_real(grid).permute(3, 0, 1, 2).contiguous()
        out_imag = self.net_imag(grid).permute(3, 0, 1, 2).contiguous()
        out_real = out_real.reshape(self.out_channels, self.in_channels, *(grid.shape[:3]))
        out_imag = out_imag.reshape(self.out_channels, self.in_channels, *(grid.shape[:3]))
        return torch.complex(out_real, out_imag)

    def _weights_out(self, shape):
        grid = self.get_grid(shape)
        out = self.net_out(grid).permute(3, 0, 1, 2).contiguous()
        out = out.reshape(self.out_channels, self.in_channels, shape[0], shape[1], shape[2])
        return out

    def cal_weights(self, shape):
        self.set_shape(shape)
        self.weights = self._weights(shape)
        if self.net_out:
            self.weights_out = self._weights_out(shape)

    def set_shape(self, shape):
        self.shape = shape

    def forward(self, x):
        batchsize = x.shape[0]
        shape = x.shape[-3:]
        self.cal_weights(shape)

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])  # (B, 1, 20, 64, 64)

        out_ft = (x_ft.unsqueeze(dim=1)) * self.weights
        if self.net_out:
            x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
            x = x * self.weights_out
            x = x.sum(dim=2)
        else:
            out_ft = out_ft.sum(dim=2)  # (B, 20, 64, 64)
            x = torch.fft.irfft2(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

    def get_grid(self, shape):
        mx, my, mt = shape[0], shape[1], shape[2]
        # mx, my = torch.meshgrid(torch.linspace(0, 1, mx), torch.linspace(0, 1, my), indexing="ij")
        mx, my, mt = torch.meshgrid(torch.linspace(0, 1, mx), torch.linspace(0, 1, my), torch.linspace(0, 1, mt))
        mx, my, mt = mx.to(device), my.to(device), mt.to(device)
        return torch.stack([mx, my, mt], dim=-1)

    def get_grid_freq(self, shape):
        mx, my, mt = shape[0], shape[1], shape[2]
        # mx, my = torch.meshgrid(torch.fft.fftfreq(mx, d=1), torch.fft.rfftfreq(my, d=1), indexing="ij")
        mx, my, mt = torch.meshgrid(torch.fft.fftfreq(mx, d=1), torch.fft.fftfreq(my, d=1), torch.fft.rfftfreq(mt, d=1))
        mx, my, mt = mx.to(device), my.to(device), mt.to(device)
        return torch.stack([mx, my, mt], dim=-1)


class NS(nn.Module):
    def __init__(self, width, symbol_act, use_last=False, num_layer=2, num_hidden=32, net_out=True):
        super(NS, self).__init__()
        self.width = width
        self.use_last = use_last
        self.padding = 6  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(13, self.width)
        if use_last:
            self.fc0 = nn.Linear(3, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv3d_fast(
            self.width,
            self.width,
            num_layer=num_layer,
            num_hidden=num_hidden,
            symbol_act=symbol_act,
            net_out=net_out,
        )
        self.conv1 = SpectralConv3d_fast(
            self.width,
            self.width,
            num_layer=num_layer,
            num_hidden=num_hidden,
            symbol_act=symbol_act,
            net_out=net_out,
        )
        self.conv2 = SpectralConv3d_fast(
            self.width,
            self.width,
            num_layer=num_layer,
            num_hidden=num_hidden,
            symbol_act=symbol_act,
            net_out=net_out,
        )
        self.conv3 = SpectralConv3d_fast(
            self.width,
            self.width,
            num_layer=num_layer,
            num_hidden=num_hidden,
            symbol_act=symbol_act,
            net_out=net_out,
        )
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def cal_weights(self, shape):
        for mod in list(self.children()):
            if isinstance(mod, SpectralConv3d_fast):
                mod.cal_weights(shape)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        if self.use_last:
            x = x[..., -1:]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic, N, F, X, Y, T (T%2 == 0)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., : -self.padding]  # N, F, X, Y, T
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_t = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_t, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_t, 1])
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = gridt.reshape(1, 1, 1, size_t, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridt), dim=-1).to(device)


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Put your hyperparameters")
    parser.add_argument("name", type=str, help="experiments name")
    parser.add_argument("--batch", default=20, type=int, help="batch size")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of Epochs")
    parser.add_argument("--lr", default=5e-3, type=float, help="learning rate")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--step_size", default=200, type=int, help="scheduler step size")
    parser.add_argument("--gamma", default=0.5, type=float, help="scheduler factor")
    parser.add_argument("--multgpu", action="store_true", help="whether multiple gpu or not")
    parser.add_argument("--nu", default=1e-5, type=float, help="vis in NS equation")
    parser.add_argument("--width", default=20, type=int, help="number of channel")
    parser.add_argument("--num_layer", default=2, type=int, help="number of hidden layer of implicit network")
    parser.add_argument("--num_hidden", default=32, type=int, help="dimension of hidden layer of implicit network")
    parser.add_argument("--load_path", default=None, type=str, help="path of directory to resume the training")
    parser.add_argument("--act", default="gelu", type=str, help="activation")
    parser.add_argument("--num_data", default=1000, type=int, help="number of data to use, only for which nu=1e-4")
    parser.add_argument("--net_out", action="store_false", help="use symbol network with a(x) or not")
    parser.add_argument("--device_name", default="cuda:0", help="name of gpu")
    parser.add_argument("--model_path", default="test", help="name of model")

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = get_args()
    print(args)
    NAME = args.name
    if args.load_path is None:
        PATH = "results/{}/".format(sys.argv[0][:-3])
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        PATH = os.path.join(PATH, NAME)
        if not os.path.exists(PATH):
            os.mkdir(PATH)
    else:
        PATH = args.load_path
        args = torch.load(os.path.join(args.load_path, "args.bin"))
        args.load_path = PATH
        args.name = NAME
        PATH = os.path.join(PATH, NAME)
        os.mkdir(PATH)

    shutil.copy(sys.argv[0], os.path.join(PATH, "code.py"))

    if args.multgpu:
        num_gpu = torch.cuda.device_count()
    else:
        num_gpu = 1

    device = torch.device(args.device_name if torch.cuda.is_available() else "cpu")

    lr = args.lr
    wd = args.wd
    batch_size = args.batch
    EPOCHS = args.epochs
    step_size = args.step_size
    gamma = args.gamma
    nu = args.nu
    width = args.width
    num_layer = args.num_layer
    num_hidden = args.num_hidden
    symbol_act = args.act
    net_out = args.net_out

    torch.save(args, os.path.join(PATH, "args.bin"))

    if nu == 1e-5:
        T_in = 10
        T = 10
        ntrain = 1000
        ntest = 200
        u = torch.load("data/ns_V1e-5_N1200_T20.bin")

    elif nu == 1e-3:
        T_in = 10
        T = 40
        ntrain = 1000
        ntest = 200
        TRAIN_PATH = "data/ns_V1000_N5000_T50.mat"
        TEST_PATH = "data/ns_V1000_N5000_T50.mat"
        sub = 1
        sub_t = None
        S = 64

    elif nu == 1e-4:
        T_in = 10
        T = 40
        ntrain = 1000
        ntest = 200
        TRAIN_PATH = "data/ns_V10000_N5000_T50.mat"
        TEST_PATH = "data/ns_V10000_N5000_T50.mat"
        sub = 1
        sub_t = None
        S = 64

    elif nu == 10:
        T_in = 10
        T = 40
        ntrain = 20
        ntest = 20
        TRAIN_PATH = "data/ns_data_V1e-4_N20_128_0_T50.mat"
        TEST_PATH = "data/ns_data_V1e-4_N20_128_0_T50.mat"
        sub = 1
        S = 128

    # x_train = u[:ntrain, :, :, :T_in]
    # y_train = u[:ntrain, :, :, T_in : T + T_in]
    # print(x_train.shape, y_train.shape)
    # trainset = torch.utils.data.TensorDataset(x_train, y_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_gpu)
    # x_test = u[-ntest:, :, :, :T_in]
    # y_test = u[-ntest:, :, :, T_in : T + T_in]
    # testset = torch.utils.data.TensorDataset(x_test, y_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_gpu)
    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field("u")[:ntrain, ::sub, ::sub, :T_in]
    train_u = reader.read_field("u")[:ntrain, ::sub, ::sub, T_in : T + T_in]

    reader = MatReader(TEST_PATH)
    test_a = reader.read_field("u")[-ntest:, ::sub, ::sub, :T_in]
    test_u = reader.read_field("u")[-ntest:, ::sub, ::sub, T_in : T + T_in]

    # a_normalizer = UnitGaussianNormalizer(train_a, device=device)
    # train_a = a_normalizer.encode(train_a)
    # test_a = a_normalizer.encode(test_a)

    # y_normalizer = UnitGaussianNormalizer(train_u, device=device)
    # train_u = y_normalizer.encode(train_u)

    train_a = train_a.reshape(ntrain, S, S, 1, T_in).repeat([1, 1, 1, T, 1])
    test_a = test_a.reshape(ntest, S, S, 1, T_in).repeat([1, 1, 1, T, 1])

    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = NS(
        width=width,
        symbol_act=symbol_act,
        use_last=False,
        num_hidden=num_hidden,
        num_layer=num_layer,
        net_out=net_out,
    ).to(device)

    if num_gpu > 1:
        print("Let's use", num_gpu, "GPUs!")
        model = nn.DataParallel(model).cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    start_epoch = 0

    if args.load_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.load_path, "weight.bin")))
        checkpoint = torch.load(os.path.join(args.load_path, "checkpoint.bin"))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        print("Load previous checkpoints from {}".format(args.load_path))
        print("Resume from %d epoch (reamining %d epochs)" % (start_epoch, EPOCHS - start_epoch))

    # train_mse_list = torch.zeros(EPOCHS + 1).to(device)
    # train_l2_list = torch.zeros(EPOCHS + 1).to(device)
    # var_l2_list = torch.zeros(EPOCHS + 1).to(device)
    myloss = LpLoss(size_average=False)
    # for ep in range(start_epoch, EPOCHS):
    #     model.train()
    #     t1 = default_timer()
    #     train_mse = 0
    #     train_l2 = 0
    #     for x, y in trainloader:
    #         x, y = x.to(device), y.to(device)
    #         # print("Train in: ", x.shape)
    #         optimizer.zero_grad()
    #         out = model(x).squeeze()

    #         mse = F.mse_loss(out, y, reduction="mean")
    #         # mse.backward()

    #         n, h, w, t = out.shape
    #         l2 = myloss(out.view(n, -1), y.view(n, -1))
    #         l2.backward()

    #         optimizer.step()
    #         train_mse += mse.item()
    #         train_l2 += l2.item()

    #     scheduler.step()
    #     torch.save(
    #         {"epoch": ep, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
    #         os.path.join(PATH, "checkpoint.bin"),
    #     )

    #     model.eval()
    #     test_l2 = 0.0
    #     with torch.no_grad():
    #         for x, y in testloader:
    #             x, y = x.to(device), y.to(device)

    #             out = model(x).squeeze()
    #             n, h, w, t = out.shape
    #             test_l2 += myloss(out.view(n, -1), y.view(n, -1)).item()

    #     train_mse /= len(trainloader)
    #     train_l2 /= ntrain
    #     test_l2 /= ntest

    #     t2 = default_timer()
    #     print(ep, t2 - t1, train_mse, train_l2, test_l2)
    #     train_mse_list[ep] = train_mse
    #     train_l2_list[ep] = train_l2
    #     var_l2_list[ep] = test_l2

    # torch.save(model.state_dict(), os.path.join(PATH, "weight.bin"))

    if args.model_path != "test":
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

    n, h, w, t = test_u.shape
    pred = torch.zeros(n, h, 2 * w, t).to(device)
    test_l2_list = torch.zeros(T).to(device)
    with torch.no_grad():
        count = 0
        # train_mse = 0
        # train_l2 = 0
        # for idx, (x, y) in enumerate(trainloader):
        #     x, y = x.to(device), y.to(device)
        #     out = model(x).squeeze()
        #     mse = F.mse_loss(out, y, reduction="mean")
        #     n, h, w, t = out.shape
        #     l2 = myloss(out.view(n, -1), y.view(n, -1))
        #     train_mse += mse.item()
        #     train_l2 += l2.item()

        test_l2 = 0.0
        for idx, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze(4)

            n, h, w, t = out.shape

            for t in range(T):
                test_l2_list[t] += myloss(
                    out[..., t].view(n, -1),
                    y[..., t].view(n, -1),
                ).item()
            test_l2 += myloss(out.view(n, -1), y.view(n, -1)).item()

            pred[count : count + n] = torch.cat((y, out), dim=2)
            count = count + n

        # test_l2_list /= ntest
        # train_mse /= len(trainloader)
        # train_l2 /= ntrain
        # test_l2 /= ntest
        # train_mse_list[ep + 1] = train_mse
        # train_l2_list[ep + 1] = train_l2
        # var_l2_list[ep + 1] = test_l2

    scipy.io.savemat(os.path.join(PATH, "pred.mat"), mdict={"pred": pred.cpu().numpy()})

    # result_dict = {
    #     "train_mse": train_mse_list.to("cpu"),
    #     "train_l2": train_l2_list.to("cpu"),
    #     "var_l2": var_l2_list.to("cpu"),
    #     "test_l2_time": test_l2_list.to("cpu"),
    # }
    # save_pickle(result_dict, os.path.join(PATH, "result.pt"))
