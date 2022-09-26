import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import sys
from utilities3 import *
import torch.nn.functional as F
from timeit import default_timer

MODEL_PATH = "model"
RESULT_PATH = "result"
INFER_PATH = "inference"
DATA_PATH = "../galerkin-transformer/data"
batch_size = 64
device_name = "cuda:1"

visc_list = ["V1000", "V10000", "Vmix"]
force_list = ["Ffix", "Fvar"]
pred_list = [10, 40]
de_solver_list = [""]


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
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
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

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

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(13, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv1 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv2 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv3 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        # print("FNO3d in: ", x.shape)
        x = torch.cat((x, grid), dim=-1)
        # print("FNO3d emb: ", x.shape)
        x = self.fc0(x)  # N, X, Y, T, T_in+3 --> N, X, Y, T, F
        # print("FNO3d FE: ", x.shape)
        x = x.permute(0, 4, 1, 2, 3)  # N, F(T_in), X, Y, T
        x = F.pad(
            x, [0, self.padding]
        )  # pad the domain if input is non-periodic, N, F, X, Y, T (T%2 == 0)

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
        x = x.permute(0, 2, 3, 4, 1)  # N, X, Y, T, F
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # N, X, Y, T, 1
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat(
            [batchsize, 1, size_y, size_z, 1]
        )
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat(
            [batchsize, size_x, 1, size_z, 1]
        )
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat(
            [batchsize, size_x, size_y, 1, 1]
        )
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
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

        self.conv0 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv1 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv2 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv3 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2
        )
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


def inference3d(batch_size, df, mf, in_s, out_s, device_name):
    t1 = default_timer()
    # TRAIN_PATH = 'data/ns_data_V100_N1000_T50_1.mat'
    # TEST_PATH = 'data/ns_data_V100_N1000_T50_2.mat'
    DATA_PATH = "/kky/Neural-Operator-Leaner/galerkin-transformer/data"
    RESULT_PATH = "result"
    MODEL_PATH = "model"
    GIF_PATH = "inference"
    TRAIN_PATH = os.path.join(DATA_PATH, df)
    TEST_PATH = os.path.join(DATA_PATH, df)
    T_inp = in_s
    T_out = out_s
    model_name = mf
    device_name = device_name

    device = torch.device(device_name)
    model_path = os.path.join(MODEL_PATH, model_name)

    ntrain = 1000
    ntest = 200

    modes = 12
    mode_t = 4
    width = 20

    sub = 1
    S = 64 // sub
    T_in = T_inp
    T = T_out

    ################################################################
    # load data
    ################################################################

    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field("u")[:ntrain, ::sub, ::sub, :T_in]
    train_u = reader.read_field("u")[:ntrain, ::sub, ::sub, T_in : T + T_in]

    reader = MatReader(TEST_PATH)
    test_a = reader.read_field("u")[-ntest:, ::sub, ::sub, :T_in]
    test_u = reader.read_field("u")[-ntest:, ::sub, ::sub, T_in : T + T_in]

    assert S == test_u.shape[-2]
    assert T == test_u.shape[-1]

    a_normalizer = UnitGaussianNormalizer(train_a, device=device)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u, device=device)
    train_u = y_normalizer.encode(train_u)
    y_normalizer.cuda()

    train_a = train_a.reshape(ntrain, S, S, 1, T_in).repeat([1, 1, 1, T, 1])
    test_a = test_a.reshape(ntest, S, S, 1, T_in).repeat([1, 1, 1, T, 1])

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u),
        batch_size=batch_size,
        shuffle=False,
    )

    t2 = default_timer()

    print("preprocessing finished, time used:", t2 - t1, flush=True)

    # model = FNO3d(modes, modes, mode_t, width).to(device)
    model = torch.load(model_path).to(device)
    model.eval()

    myloss = LpLoss(size_average=False)

    n, h, w, t = test_u.shape
    pred = torch.zeros(n, h, 2 * w, t).to(device)
    test_l2_list = torch.zeros(T).to(device)
    with torch.no_grad():
        count = 0
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            out = model(x).squeeze()
            out = y_normalizer.decode(out)
            n, h, w, t = out.shape
            for t in range(T):
                test_l2_list[t] += myloss(
                    out[..., t].view(n, -1),
                    y[..., t].view(n, -1),
                ).item()

            pred[count : count + n] = torch.cat((y, out), dim=2)
            count = count + n
        test_l2_list /= ntest

    return pred.permute(0, 3, 1, 2).cpu().numpy(), test_l2_list.cpu().numpy()


def inference2d(batch_size, df, mf, in_s, out_s, device_name):
    t1 = default_timer()
    # TRAIN_PATH = 'data/ns_data_V100_N1000_T50_1.mat'
    # TEST_PATH = 'data/ns_data_V100_N1000_T50_2.mat'
    DATA_PATH = "/kky/Neural-Operator-Leaner/galerkin-transformer/data"
    RESULT_PATH = "result"
    MODEL_PATH = "model"
    GIF_PATH = "inference"
    TRAIN_PATH = os.path.join(DATA_PATH, df)
    TEST_PATH = os.path.join(DATA_PATH, df)
    T_inp = in_s
    T_out = out_s
    model_name = mf
    device_name = device_name

    device = torch.device(device_name)
    model_path = os.path.join(MODEL_PATH, model_name)

    ntrain = 1000
    ntest = 200

    modes = 12
    width = 20

    sub = 1
    S = 64 // sub
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

    assert S == test_u.shape[-2]
    assert T == test_u.shape[-1]

    # a_normalizer = UnitGaussianNormalizer(train_a, device=device)
    # train_a = a_normalizer.encode(train_a)
    # test_a = a_normalizer.encode(test_a)

    # y_normalizer = UnitGaussianNormalizer(train_u, device=device)
    # train_u = y_normalizer.encode(train_u)
    # y_normalizer.cuda()

    train_a = train_a.reshape(ntrain, S, S, T_in)
    test_a = test_a.reshape(ntest, S, S, T_in)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
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

    t2 = default_timer()

    print("preprocessing finished, time used:", t2 - t1, flush=True)

    # model = FNO3d(modes, modes, mode_t, width).to(device)
    model = torch.load(model_path).to(device)
    model.eval()

    myloss = LpLoss(size_average=False)

    n, h, w, t = test_u.shape
    preds = torch.zeros(n, h, 2 * w, t).to(device)
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

    return preds.permute(0, 3, 1, 2).cpu().numpy(), test_l2_list.cpu().numpy()


def save_gifs(u_pred, fname):
    data = u_pred
    cm = plt.get_cmap("RdYlBu")
    colored_data = [cm(d)[:, :, :3] * 255 for d in data]
    imgs = [Image.fromarray(d.astype(np.uint8)) for d in colored_data]
    imgs[0].save(fname, save_all=True, append_images=imgs[1:], duration=100, loop=0)


fig, ax = plt.subplots(
    figsize=(11.75, 8.25 * 2), nrows=len(force_list) * 2, ncols=len(visc_list)
)
for i, force in enumerate(force_list):
    for j, visc in enumerate(visc_list):
        for pred, c in zip(pred_list, ["r", "b"]):
            for de_solver, s, de_solver_idx in zip(de_solver_list, ["-"], ["euler"]):
                # plot loss during training
                rf = f"result_FNO3d_ns_{visc}_{force}_T1024_V200_10to{pred}{de_solver}.pt"
                rf1 = f"result_FNO3d_12_12_12_ns_{visc}_{force}_T1024_V200_10to{pred}{de_solver}.pt"
                rf2 = f"result_FNO3d_12_12_24_ns_{visc}_{force}_T1024_V200_10to{pred}{de_solver}.pt"
                print(rf)
                fpath = os.path.join(RESULT_PATH, rf)
                if os.path.isfile(fpath):
                    f = open(os.path.join(RESULT_PATH, rf), "rb")
                    data = pickle.load(f)
                    # print(f"{rf} {data['best_val_epoch']} {data['best_val_metric']} {len(data['loss_val'])}")
                    ax[i, j].plot(data["var_l2"], c + s, label=rf)
                    ax[i, j].plot(data["train_l2"], c + "--", label=rf)
                    # ax[i + len(force_list), j].plot(data["test_l2_time"], c + s, label=rf)
                if os.path.isfile(os.path.join(RESULT_PATH, rf1)):
                    f = open(os.path.join(RESULT_PATH, rf1), "rb")
                    data = pickle.load(f)
                    # print(f"{rf} {data['best_val_epoch']} {data['best_val_metric']} {len(data['loss_val'])}")
                    ax[i, j].plot(data["var_l2"], "g" + s, label=rf)
                    ax[i, j].plot(data["train_l2"], "g" + "--", label=rf)
                if os.path.isfile(os.path.join(RESULT_PATH, rf2)):
                    f = open(os.path.join(RESULT_PATH, rf2), "rb")
                    data = pickle.load(f)
                    # print(f"{rf} {data['best_val_epoch']} {data['best_val_metric']} {len(data['loss_val'])}")
                    ax[i, j].plot(data["var_l2"], "m" + s, label=rf)
                    ax[i, j].plot(data["train_l2"], "m" + "--", label=rf)

                # inference
                if visc == "Vmix":
                    if force == "Ffix":
                        df = f"ns_{visc}_N3000_T50.mat"
                    elif force == "Fvar":
                        df = f"ns_{visc}_N3000_T50_var_f.mat"
                else:
                    if force == "Ffix":
                        df = f"ns_{visc}_N5000_T50.mat"
                    elif force == "Fvar":
                        df = f"ns_{visc}_N5000_T50_var_f.mat"
                mf = (
                    f"model_FNO3d_ns_{visc}_{force}_T1024_V200_10to{pred}{de_solver}.pt"
                )

                mf1 = f"model_FNO3d_12_12_12_ns_{visc}_{force}_T1024_V200_10to{pred}{de_solver}.pt"

                mf2 = f"model_FNO3d_12_12_24_ns_{visc}_{force}_T1024_V200_10to{pred}{de_solver}.pt"

                gif_dir = os.path.join(
                    INFER_PATH,
                    f"ns_FNO3d_{visc}_{force}_T1024_V200_10to{pred}{de_solver}",
                )

                gif_dir1 = os.path.join(
                    INFER_PATH,
                    f"ns_FNO3d_12_12_12_{visc}_{force}_T1024_V200_10to{pred}{de_solver}",
                )

                gif_dir2 = os.path.join(
                    INFER_PATH,
                    f"ns_FNO3d_12_12_24_{visc}_{force}_T1024_V200_10to{pred}{de_solver}",
                )

                if os.path.isfile(os.path.join(DATA_PATH, df)) and os.path.isfile(
                    os.path.join(MODEL_PATH, mf)
                ):
                    b_u_pred, metric_val = inference3d(
                        batch_size, df, mf, 10, pred, device_name
                    )
                    ax[i + len(force_list), j].plot(metric_val, c + s, label=rf)

                    if os.path.isdir(gif_dir) is False:
                        os.mkdir(gif_dir)
                    N, T, H, W = b_u_pred.shape
                    for idx, u_pred in enumerate(b_u_pred):
                        save_gifs(
                            u_pred,
                            os.path.join(gif_dir, f"{idx}.gif"),
                        )

                if os.path.isfile(os.path.join(DATA_PATH, df)) and os.path.isfile(
                    os.path.join(MODEL_PATH, mf1)
                ):
                    b_u_pred, metric_val = inference3d(
                        batch_size, df, mf1, 10, pred, device_name
                    )
                    ax[i + len(force_list), j].plot(metric_val, "g" + s, label=rf)

                    if os.path.isdir(gif_dir1) is False:
                        os.mkdir(gif_dir1)
                    N, T, H, W = b_u_pred.shape
                    for idx, u_pred in enumerate(b_u_pred):
                        save_gifs(
                            u_pred,
                            os.path.join(gif_dir1, f"{idx}.gif"),
                        )

                if os.path.isfile(os.path.join(DATA_PATH, df)) and os.path.isfile(
                    os.path.join(MODEL_PATH, mf2)
                ):
                    b_u_pred, metric_val = inference3d(
                        batch_size, df, mf2, 10, pred, device_name
                    )
                    ax[i + len(force_list), j].plot(metric_val, "m" + s, label=rf)

                    if os.path.isdir(gif_dir2) is False:
                        os.mkdir(gif_dir2)
                    N, T, H, W = b_u_pred.shape
                    for idx, u_pred in enumerate(b_u_pred):
                        save_gifs(
                            u_pred,
                            os.path.join(gif_dir2, f"{idx}.gif"),
                        )

        ax[i, j].set_ylim([1e-3, 1])
        # ax[i, j].legend()
        ax[i, j].grid()
        ax[i, j].semilogy()

        ax[i + len(force_list), j].set_ylim([1e-3, 1])
        # ax[i + len(force_list), j].legend()
        ax[i + len(force_list), j].grid()
        ax[i + len(force_list), j].semilogy()

plt.savefig(os.path.join(INFER_PATH, "val_loss_FNO3d_mode_test.png"))
