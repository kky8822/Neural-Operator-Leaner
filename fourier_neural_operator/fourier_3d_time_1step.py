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
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

################################################################
# fourier layers
################################################################

class SpectralConv3d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_fast, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        # print("input, weight: ", input.shape, weights.shape)
        # return torch.einsum("bixyz,ioxyz->boxyz", input, torch.view_as_complex(weights))
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        # print(x.shape)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])
        # print(x_ft.shape)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        # print(out_ft.shape)
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        # print(x.shape)
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(4, self.width)

        self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        # print(x.shape)
        x = self.fc0(x)
        # print(x.shape)
        x = x.permute(0, 4, 1, 2, 3)
        # print(x.shape)

        x1 = self.conv0(x)
        # print(x1.shape)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn3(x1 + x2)

        x = self.fc3(x)
        x = F.relu(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        


        return x

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()
        self.conv1 = SimpleBlock2d(modes, modes, 4, width)

    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

################################################################
# configs
################################################################

TRAIN_PATH = './PDE_datasets/ns_data_V10000_N1200_T50.mat'
TEST_PATH = './PDE_datasets/ns_data_V10000_N1200_T50.mat'

ntrain = 1000
ntest = 200

modes = 12
width = 20

batch_size = 20
batch_size2 = batch_size


epochs = 500
learning_rate = 0.0025
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

path = 'ns_fourier_3d_rnn_1step_interp_V10000_T50_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path


runtime = np.zeros(2, )
t1 = default_timer()


sub = 1
S = 64
T_in = 10
T_start = 0
step = T_in - T_start
T = 40

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain,::sub,::sub,T_start:T_in]
train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:,::sub,::sub,T_start:T_in]
test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

print("train dataset shape: ", train_a.shape, train_u.shape)
print("test dataset shape: ", test_a.shape, test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])



train_a = train_a.reshape(ntrain,S,S,step,1)
test_a = test_a.reshape(ntest,S,S,step,1)

# cat the location information (x,y,t)
gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, step, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, step, 1])
gridt = torch.tensor(np.linspace(0, 1, step+1)[1:], dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, step, 1).repeat([1, S, S, 1, 1])

# print("grid shape: ", gridx.shape, gridy.shape, gridt.shape)

train_a = torch.cat((gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
                       gridt.repeat([ntrain,1,1,1,1]), train_a), dim=-1)
test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                       gridt.repeat([ntest,1,1,1,1]), test_a), dim=-1)

print("train dataset shape: ", train_a.shape, train_u.shape)
print("test dataset shape: ", test_a.shape, test_u.shape)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

################################################################
# training and evaluation
################################################################
model = Net2d(modes, width).cuda()
# model = torch.load('./model/ns_fourier_3d_rnn_V10000_T50_N1000_ep500_m12_w20')

print(model.count_params())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)

gridx = gridx.to(device)
gridy = gridy.to(device)
gridt = gridt.to(device)
for ep in range(epochs):
   model.train()
   t1 = default_timer()
   train_l2_step = 0
   train_l2_full = 0
   for xx, yy in train_loader:
       loss = 0
       xx = xx.to(device)
       yy = yy.to(device)

       for t in range(0, T):
           y = yy[..., t]
           im = model(xx)
           loss += myloss(im.reshape(batch_size,-1), y.reshape(batch_size,-1))

           if t == 0:
               pred = im.squeeze()
           else:
               pred = torch.cat((pred, im.squeeze()), -1)

           im = im.reshape(batch_size,S,S,1,1)
           print(gridx.repeat([batch_size, 1, 1, 1, 1]).shape)
           im = torch.cat((gridx.repeat([batch_size, 1, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1, 1]),
                                gridt.repeat([batch_size, 1, 1, 1, 1]), im), dim=-1)
           # print("im shape: ", im.shape)
           xx = torch.cat([xx[..., 1:, :], im], -2)

       train_l2_step += loss.item()
       l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
       train_l2_full += l2_full.item()

       optimizer.zero_grad()
       loss.backward()
       # l2_full.backward()
       optimizer.step()

   test_l2_step = 0
   test_l2_full = 0
   with torch.no_grad():
       for xx, yy in test_loader:
           loss = 0
           xx = xx.to(device)
           yy = yy.to(device)

           for t in range(0, T, step):
               y = yy[..., t:t + step]
               im = model(xx)
               loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

               if t == 0:
                   pred = im.squeeze()
               else:
                   pred = torch.cat((pred, im.squeeze()), -1)

               im = im.reshape(batch_size,S,S,step,1)
               im = torch.cat((gridx.repeat([batch_size, 1, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1, 1]),
                               gridt.repeat([batch_size, 1, 1, 1, 1]), im), dim=-1)
               xx = torch.cat([xx[..., step:, :], im], -2)

           test_l2_step += loss.item()
           test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

   t2 = default_timer()
   scheduler.step()
   print(ep, t2-t1, train_l2_step/ntrain/(T/step), train_l2_full/ntrain, test_l2_step/ntest/(T/step), test_l2_full/ntest, flush=True)
torch.save(model, path_model)


pred_tot = torch.zeros([200,64,64,40])
u_tot = torch.zeros([200,64,64,40])
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
gridx = gridx.to(device)
gridy = gridy.to(device)
gridt = gridt.to(device)

with torch.no_grad():
    for i, (xx, yy) in enumerate(test_loader):
        xx = xx.to(device)
        yy = yy.to(device)
        print(i, xx.shape, yy.shape)
        
        pred = torch.zeros([64,64,40])
        u = torch.zeros([64,64,40])
        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
        
            u[:,:,t:t+step] = y
            pred[:,:,t:t+step] = im

            im = im.reshape(1,S,S,step,1)
            im = torch.cat((gridx.repeat([1, 1, 1, 1, 1]), gridy.repeat([1, 1, 1, 1, 1]),
                                gridt.repeat([1, 1, 1, 1, 1]), im), dim=-1)
            
            xx = torch.cat([xx[..., step:, :], im], -2)
        pred_tot[i] = pred
        u_tot[i] = u

scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred_tot.cpu().numpy(), 'u':u_tot.cpu().numpy()})