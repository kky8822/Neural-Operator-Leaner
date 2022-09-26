"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""
import os, sys

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from Adam import Adam

import pandas as pd
import pickle

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

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

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


def save_pickle(var, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(var, f)


################################################################
# configs
################################################################

# TRAIN_PATH = 'data/ns_data_V100_N1000_T50_1.mat'
# TEST_PATH = 'data/ns_data_V100_N1000_T50_2.mat'
DATA_PATH = "/kky/Neural-Operator-Leaner/galerkin-transformer/data"
RESULT_PATH = "result"
MODEL_PATH = "model"
GIF_PATH = "inference"
TRAIN_PATH = os.path.join(DATA_PATH, sys.argv[1])
TEST_PATH = os.path.join(DATA_PATH, sys.argv[1])
T_inp = int(sys.argv[2])
T_out = int(sys.argv[3])
model_name = sys.argv[4]
result_name = sys.argv[5]
pred_name = sys.argv[6]
device_name = sys.argv[7]

device = torch.device(device_name)
model_path = os.path.join(MODEL_PATH, model_name)
result_path = os.path.join(RESULT_PATH, result_name)
pred_path = os.path.join(GIF_PATH, pred_name)

ntrain = 1000
ntest = 200

modes = 12
width = 20

batch_size = 64

epochs = 100
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

runtime = np.zeros(
    2,
)
t1 = default_timer()

sub = 1
S = 64
T_in = T_inp
T = T_out
step = 1

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field("u")[:ntrain, ::sub, ::sub, :T_in]
train_u = reader.read_field("u")[:ntrain, ::sub, ::sub, T_in : T + T_in]

reader = MatReader(TEST_PATH)
test_a = reader.read_field("u")[-ntest:, ::sub, ::sub, :T_in]
test_u = reader.read_field("u")[-ntest:, ::sub, ::sub, T_in : T + T_in]

assert S == train_u.shape[-2]
assert T == train_u.shape[-1]

# a_normalizer = UnitGaussianNormalizer(train_a, device=device)
# train_a = a_normalizer.encode(train_a)
# test_a = a_normalizer.encode(test_a)

# y_normalizer = UnitGaussianNormalizer(train_u, device=device)
# train_u = y_normalizer.encode(train_u)

train_a = train_a.reshape(ntrain, S, S, T_in)
test_a = test_a.reshape(ntest, S, S, T_in)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_a, train_u),
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a, test_u),
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
)

t2 = default_timer()

print("preprocessing finished, time used:", t2 - t1)


################################################################
# training and evaluation
################################################################

model = FNO2d(modes, modes, width).to(device)
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

print(count_params(model))
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

train_l2_step_list = torch.zeros(epochs).to(device)
train_l2_full_list = torch.zeros(epochs).to(device)
var_l2_step_list = torch.zeros(epochs).to(device)
var_l2_full_list = torch.zeros(epochs).to(device)
myloss = LpLoss(size_average=False)
# y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        N = xx.shape[0]

        for t in range(0, T, step):
            y = yy[..., t : t + step]
            im = model(xx)

            loss += myloss(im.reshape(N, -1), y.reshape(N, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(N, -1), yy.reshape(N, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    model.eval()
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            N = xx.shape[0]

            for t in range(0, T, step):
                y = yy[..., t : t + step]
                im = model(xx)
                # im = y_normalizer.decode(im)[..., 0].unsqueeze(-1)
                loss += myloss(im.reshape(N, -1), y.reshape(N, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(N, -1), yy.reshape(N, -1)).item()

    train_l2_step /= ntrain / (T / step)
    train_l2_full /= ntrain
    test_l2_step /= ntest / (T / step)
    test_l2_full /= ntest

    t2 = default_timer()

    print(ep, t2 - t1, train_l2_step, train_l2_full, test_l2_step, test_l2_full)

    train_l2_step_list[ep] = train_l2_step
    train_l2_full_list[ep] = train_l2_full
    var_l2_step_list[ep] = test_l2_step
    var_l2_full_list[ep] = test_l2_full

torch.save(model, model_path)

n, h, w, t = test_u.shape
preds = torch.zeros(n, h, 2 * w, t).to(device)
index = 0
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False
)
test_l2_list = torch.zeros(T).to(device)
with torch.no_grad():
    count = 0
    for idx, (xx, yy) in enumerate(test_loader):
        xx = xx.to(device)
        yy = yy.to(device)

        N = xx.shape[0]

        for t in range(0, T, step):
            y = yy[..., t : t + step]
            im = model(xx)
            # im = y_normalizer.decode(im)
            test_l2_list[t] += myloss(im.reshape(N, -1), y.reshape(N, -1)).item()

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        n, h, w, t = yy.shape
        preds[count : count + n] = torch.cat((yy, pred), dim=2)
        count = count + n
    test_l2_list /= ntest

scipy.io.savemat(pred_path, mdict={"pred": preds.cpu().numpy()})

result_dict = {
    "train_l2": train_l2_full_list.to("cpu"),
    "var_l2": var_l2_full_list.to("cpu"),
    "train_l2_step": train_l2_step_list.to("cpu"),
    "var_l2_step": var_l2_step_list.to("cpu"),
    "test_l2_time": test_l2_list.to("cpu"),
}
save_pickle(result_dict, result_path)
