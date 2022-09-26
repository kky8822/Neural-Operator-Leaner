# Codes for section: Results on Navier Stocks Equation (3D)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from data_load_navier_stocks import *
import matplotlib.pyplot as plt
from navier_stokes_uno3d import Uno3D_T40, Uno3D_T20, Uno3D_T10, Uno3D_T9
import operator
import random
from functools import reduce
from functools import partial
from ns_train_3d import train_model_3d
from timeit import default_timer
from utilities3 import *
from Adam import Adam
from torchsummary import summary

# for conda: from torchinfo import summary
import gc
import math

plt.rcParams["figure.figsize"] = [6, 30]
plt.rcParams["image.interpolation"] = "nearest"


S = 64  # resolution SxS
T_in = 10  # input time interval (0 - T_in)
# T_f = 10 # output time interval (T_in -  = T_in+T_f)
T_f = 40  # output time interval (T_in -  = T_in+T_f)
ntrain = 4300  # number of training instances
ntest = 1200  # number of test instances
nval = 500  # number of validation instances
batch_size = 16
width = 20  # Uplifting dimesion
inwidth = 13  # dimension of UNO input ( a(x,y,t), x,y,t )
epochs = 500
# Following code load data from two separate files containing Navier-Stokes equation simulation

# data_path = '/jwi/project/galerkin/data/ns_V1e-3_N5000_T50.mat'
# train_a_1, train_u_1, test_a_1, test_u_1 = load_NS_("path to navier stokes simulation with viscosity 1e-5 with 1000 instances"\
#                                                     ,800,200, Sample_num = 1000,T_in=T_in, T = T_f, size = S)
# train_a_2, train_u_2, test_a_2, test_u_2 = load_NS_("path to navier stokes simulation with viscosity 1e-5 with 5000 instances"\
#                                                     ,4000 ,1000, Sample_num = 5000,T_in=T_in, T = T_f, size = S)

# a = torch.cat([train_a_1,train_a_2,test_a_1,test_a_2], dim = 0)
# u = torch.cat([train_u_1,train_u_2,test_u_1,test_u_2],dim = 0)


# FNO/fourier_3d.py dataload part


# ================================================= TODO 1. Dataset Path for different viscocity
# TRAIN_PATH = '/jwi/project/FNO/data/ns_V1e-3_N5000_T50.mat'
# TEST_PATH = '/jwi/project/FNO/data/ns_V1e-3_N5000_T50.mat'

# TRAIN_PATH = '/jwi/project/FNO/data/NavierStokes_V1e-5_N1200_T20.mat'
# TEST_PATH = '/jwi/project/FNO/data/NavierStokes_V1e-5_N1200_T20.mat'

# TRAIN_PATH = '/jwi/project/FNO/data/ns_datagen_V1e-4_N1200_T30.mat'
# TEST_PATH = '/jwi/project/FNO/data/ns_datagen_V1e-4_N1200_T30.mat'

# ================================================= TODO 1. Dataset to compare common setup
TRAIN_PATH = "/kky/Neural-Operator-Leaner/galerkin-transformer/data/ns_V10000_N5000_T50.mat"
TEST_PATH = "/kky/Neural-Operator-Leaner/galerkin-transformer/data/ns_V10000_N5000_T50.mat"
pred_path = "/kky/Neural-Operator-Leaner/UNO/pred/pred_ns_V10000_T50.pt"
result_path = "/kky/Neural-Operator-Leaner/UNO/result/res_ns_V10000_T50.pt"
weight_path = "/kky/Neural-Operator-Leaner/UNO/model/UNO3D_V10000_T40_commonsetup.pt"
# ================================================= TODO 2. Index
# ntotal = 5000
# ntotal = 1200
# ntrain = int(ntotal*0.7)
# ntest = int(ntotal*0.2)
# nval = int(ntotal*0.1)


################################################################
# Convert The way to load data as FNO
################################################################
ntrain = 1000
ntest = 200
nval = 200

sub = 1
S = 64 // sub
T_in = 10

T = 40
# T = 10
# T = 20


T_f = 40
# T_f = 10
# T_f = 20


################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field("u")[:ntrain, ::sub, ::sub, :T_in]  # [N, H, W, 0~10]
train_u = reader.read_field("u")[:ntrain, ::sub, ::sub, T_in : T + T_in]  # [N, H, W, 10~50]

reader = MatReader(TEST_PATH)
test_a = reader.read_field("u")[-ntest:, ::sub, ::sub, :T_in]
test_u = reader.read_field("u")[-ntest:, ::sub, ::sub, T_in : T + T_in]

# print(train_u.shape)  # torch.Size([1000, 64, 64, 40])
# print(test_u.shape)  # torch.Size([200, 64, 64, 40])
assert S == train_u.shape[-2]
assert T == train_u.shape[-1]

# Set Validation = Test
val_a = test_a
val_u = test_u


################################################################
# # Original code
################################################################

# reader = MatReader(TRAIN_PATH)
# a = reader.read_field('u')[:,::sub,::sub,:T_in] # [N, H, W, 0~10]
# u = reader.read_field('u')[:,::sub,::sub,T_in:T+T_in] # [N, H, W, 10~50]
# print("a.shape : ", a.shape) # torch.Size([4000, 64, 64, 10])
# print("u.shape : ", u.shape) # torch.Size([4000, 64, 64, 40])


# indexs = [i for i in range(a.shape[0])]
# random.shuffle(indexs)
# train_a,val_a,test_a = a[indexs[:ntrain]], a[indexs[ntrain:ntrain+nval]],a[indexs[ntrain+nval:]]
# train_u,val_u,test_u = u[indexs[:ntrain]],u[indexs[ntrain:ntrain+nval]],u[indexs[ntrain+nval:]]

################################################################
# # Original code Ends
################################################################


train_a = train_a.reshape(ntrain, S, S, T_in)
val_a = val_a.reshape(nval, S, S, T_in)
test_a = test_a.reshape(ntest, S, S, T_in)

train_a = train_a.reshape(ntrain, S, S, T_in, 1).repeat(1, 1, 1, 1, T_in)
val_a = val_a.reshape(nval, S, S, T_in, 1).repeat(1, 1, 1, 1, T_in)
test_a = test_a.reshape(ntest, S, S, T_in, 1).repeat(1, 1, 1, 1, T_in)
gc.collect()


train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_a, train_u),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(val_a, val_u),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a, test_u),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
)
# any 3d models can be trained vai train_model_3d function

import sys

device = torch.device(sys.argv[1])
model = Uno3D_T40(inwidth, width, pad=1, factor=1).to(device)
# model = Uno3D_T10(inwidth,width,pad = 1,factor = 1).cuda()
# model = Uno3D_T20(inwidth,width,pad = 1,factor = 1).cuda()
# summary(model, (64, 64, 10,1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(count_parameters(model), flush=True)


train_model_3d(
    test_u,
    pred_path,
    result_path,
    model,
    train_loader,
    val_loader,
    test_loader,
    ntrain,
    nval,
    ntest,
    weight_path=weight_path,
    T_f=T_f,
    batch_size=batch_size,
    epochs=epochs,
    learning_rate=0.0008,
    scheduler_step=100,
    scheduler_gamma=0.7,
    weight_decay=1e-3,
    device=device,
)
