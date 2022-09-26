"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which takes the 2D spatial + 1D temporal equation directly as a 3D problem
"""

import os, sys
from tkinter import N
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
# 3d fourier layers
################################################################
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        # print("x, x_ft: ", x.shape, x_ft.shape)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        # print(x.shape, x_ft.shape, self.weights1.shape)
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1)
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2)
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3)
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class AFNONET(nn.Module):
    def __init__(self, width):
        super(AFNONET, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.width = width
        self.padding = 6  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(13, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = AFNO3D(hidden_size=self.width, num_blocks=4)
        self.conv1 = AFNO3D(hidden_size=self.width, num_blocks=4)
        self.conv2 = AFNO3D(hidden_size=self.width, num_blocks=4)
        self.conv3 = AFNO3D(hidden_size=self.width, num_blocks=4)
        # self.conv4 = AFNO3D(hidden_size=self.width, num_blocks=4)
        # self.conv5 = AFNO3D(hidden_size=self.width, num_blocks=4)
        # self.conv6 = AFNO3D(hidden_size=self.width, num_blocks=4)
        # self.conv7 = AFNO3D(hidden_size=self.width, num_blocks=4)
        # self.conv8 = AFNO3D(hidden_size=self.width, num_blocks=4)
        # self.conv9 = AFNO3D(hidden_size=self.width, num_blocks=4)
        # self.conv10 = AFNO3D(hidden_size=self.width, num_blocks=4)
        # self.conv11 = AFNO3D(hidden_size=self.width, num_blocks=4)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        # Feature extract
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)  # N, X, Y, T, T_in+3 --> N, X, Y, T, F
        x = x.permute(0, 4, 1, 2, 3)  # N, F(T_in), X, Y, T
        x = F.pad(x, [0, self.padding])
        x = x.permute(0, 2, 3, 4, 1)  # N, X, Y, T, F padded

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        # x = self.conv8(x)
        # x = self.conv9(x)
        # x = self.conv10(x)
        # x = self.conv11(x)
        x = self.norm(x)

        x = x[..., : -self.padding, :]  # N, X, Y, T, F
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # N, X, Y, T, 1
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


class AFNO3D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """

    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
    ):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 1 / (self.hidden_size * self.hidden_size * self.hidden_size_factor)

        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        # global conv
        # self.w1 = nn.Parameter(
        #     self.scale
        #     * torch.randn(
        #         2,
        #         64,
        #         64,
        #         24,
        #         self.num_blocks,
        #         self.block_size,
        #         self.block_size * self.hidden_size_factor,
        #     )
        # )
        # self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        # self.w2 = nn.Parameter(
        #     self.scale
        #     * torch.randn(
        #         2,
        #         64,
        #         64,
        #         24,
        #         self.num_blocks,
        #         self.block_size * self.hidden_size_factor,
        #         self.block_size,
        #     )
        # )
        # self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, T, C = x.shape

        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        x = x.reshape(
            B,
            x.shape[1],
            x.shape[2],
            x.shape[3],
            self.num_blocks,
            self.block_size,
        )

        o1_real = torch.zeros(
            [
                B,
                x.shape[1],
                x.shape[2],
                x.shape[3],
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [
                B,
                x.shape[1],
                x.shape[2],
                x.shape[3],
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = T // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, :, :kept_modes] = F.relu(torch.einsum("...bi,bio->...bo", x[:, :, :, :kept_modes].real, self.w1[0]) - torch.einsum("...bi,bio->...bo", x[:, :, :, :kept_modes].imag, self.w1[1]) + self.b1[0])

        o1_imag[:, :, :, :kept_modes] = F.relu(torch.einsum("...bi,bio->...bo", x[:, :, :, :kept_modes].imag, self.w1[0]) + torch.einsum("...bi,bio->...bo", x[:, :, :, :kept_modes].real, self.w1[1]) + self.b1[1])

        o2_real[:, :, :, :kept_modes] = torch.einsum("...bi,bio->...bo", o1_real[:, :, :, :kept_modes], self.w2[0]) - torch.einsum("...bi,bio->...bo", o1_imag[:, :, :, :kept_modes], self.w2[1]) + self.b2[0]

        o2_imag[:, :, :, :kept_modes] = torch.einsum("...bi,bio->...bo", o1_imag[:, :, :, :kept_modes], self.w2[0]) + torch.einsum("...bi,bio->...bo", o1_real[:, :, :, :kept_modes], self.w2[1]) + self.b2[1]

        # global conv
        # o1_real[:, :, :, :kept_modes] = F.relu(
        #     torch.einsum("nxyzbi,xyzbio->nxyzbo", x[:, :, :, :kept_modes].real, self.w1[0])
        #     - torch.einsum("nxyzbi,xyzbio->nxyzbo", x[:, :, :, :kept_modes].imag, self.w1[1])
        #     + self.b1[0]
        # )

        # o1_imag[:, :, :, :kept_modes] = F.relu(
        #     torch.einsum("nxyzbi,xyzbio->nxyzbo", x[:, :, :, :kept_modes].imag, self.w1[0])
        #     + torch.einsum("nxyzbi,xyzbio->nxyzbo", x[:, :, :, :kept_modes].real, self.w1[1])
        #     + self.b1[1]
        # )

        # o2_real[:, :, :, :kept_modes] = (
        #     torch.einsum("nxyzbi,xyzbio->nxyzbo", o1_real[:, :, :, :kept_modes], self.w2[0])
        #     - torch.einsum("nxyzbi,xyzbio->nxyzbo", o1_imag[:, :, :, :kept_modes], self.w2[1])
        #     + self.b2[0]
        # )

        # o2_imag[:, :, :, :kept_modes] = (
        #     torch.einsum("nxyzbi,xyzbio->nxyzbo", o1_imag[:, :, :, :kept_modes], self.w2[0])
        #     + torch.einsum("nxyzbi,xyzbio->nxyzbo", o1_real[:, :, :, :kept_modes], self.w2[1])
        #     + self.b2[1]
        # )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], C)
        x = torch.fft.irfft2(x, s=(H, W, T), dim=(1, 2, 3), norm="ortho")
        x = x.type(dtype)
        return x + bias


def save_pickle(var, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(var, f)


################################################################
# configs
################################################################

# TRAIN_PATH = 'data/ns_data_V100_N1000_T50_1.mat'
# TEST_PATH = 'data/ns_data_V100_N1000_T50_2.mat'
DATA_PATH = "/kky/Neural-Operator-Leaner/galerkin-transformer/data"
MODEL_PATH = "model"
GIF_PATH = "inference"
TRAIN_PATH = os.path.join(DATA_PATH, sys.argv[1])
TEST_PATH = os.path.join(DATA_PATH, sys.argv[1])
T_inp = int(sys.argv[2])
T_out = int(sys.argv[3])
model_name = sys.argv[4]
pred_name = sys.argv[5]
device_name = sys.argv[6]

device = torch.device(device_name)
model_path = os.path.join(MODEL_PATH, model_name)
pred_path = os.path.join(GIF_PATH, pred_name)

ntrain = 20
ntest = 20

# width = 64
# num_blocks = 1
sparsity_threshold = 0.01
hard_thresholding_fraction = 0.25
hidden_size_factor = 2

batch_size = 2


epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma, flush=True)


runtime = np.zeros(
    2,
)
t1 = default_timer()


sub = 1
sub_t = 4
S = 256 // sub
T_in = T_inp
T = T_out

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field("u")[:ntrain, ::sub, ::sub, 3 : T_in * sub_t : 4]
train_u = reader.read_field("u")[:ntrain, ::sub, ::sub, T_in * sub_t : (T + T_in) * sub_t : 4]

reader = MatReader(TEST_PATH)
test_a = reader.read_field("u")[-ntest:, ::sub, ::sub, 3 : T_in * sub_t : 4]
test_u = reader.read_field("u")[-ntest:, ::sub, ::sub, T_in * sub_t : (T + T_in) * sub_t : 4]

assert S == train_u.shape[-2]
assert T == train_u.shape[-1]


a_normalizer = UnitGaussianNormalizer(train_a, device=device)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

y_normalizer = UnitGaussianNormalizer(train_u, device=device)
train_u = y_normalizer.encode(train_u)

train_a = train_a.reshape(ntrain, S, S, 1, T_in).repeat([1, 1, 1, T, 1])
test_a = test_a.reshape(ntest, S, S, 1, T_in).repeat([1, 1, 1, T, 1])

t2 = default_timer()

print("preprocessing finished, time used:", t2 - t1, flush=True)


################################################################
# training and evaluation
################################################################
# model = AFNONET(width=64).to(device)
model = torch.load(model_path).to(device)

print(count_params(model), flush=True)

n, h, w, t = test_u.shape
pred = torch.zeros(n, h, w, t).to(device)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
y_normalizer.cuda()
with torch.no_grad():
    count = 0
    for idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)

        out = model(x).squeeze()
        out = y_normalizer.decode(out)
        n, h, w, t = out.shape
        pred[count : count + n] = out
        count = count + n


scipy.io.savemat(pred_path, mdict={"pred": pred.cpu().numpy()})
