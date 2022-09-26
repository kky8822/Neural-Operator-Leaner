import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from functools import reduce
from functools import partial
import random
import gc
from timeit import default_timer
from utilities3 import *
from Adam import Adam
import pickle

# import logging
# from pathlib import Path

# ckpt_save_dir = Path('/jwi/project/UNO/ckpt')
# logging.basicConfig(
#    level=logging.INFO,
#    datefmt="%a, %d %b %Y %H:%M:%S",
#    format="%(asctime)s - %(message)s",
# #    filename=ckpt_save_dir.joinpath("train_log_10e-3_T40.log").absolute().as_posix(),
#    filename=ckpt_save_dir.joinpath("train_log_10e-5_T20.log").absolute().as_posix(),
#    filemode="a",
# )


def save_pickle(var, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(var, f)


def train_model_3d(
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
    weight_path,
    T_f=10,
    step=1,
    batch_size=20,
    epochs=150,
    learning_rate=0.0001,
    scheduler_step=50,
    scheduler_gamma=0.5,
    device="cuda:0",
    weight_decay=1e-3,
):

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    train_mse_list = torch.zeros(epochs).to(device)
    train_l2_list = torch.zeros(epochs).to(device)
    var_l2_list = torch.zeros(epochs).to(device)

    Min_error_t = 100000.000
    myloss = LpLoss(size_average=False)

    for ep in range(epochs):
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0

        train_l2_step = 0
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            S = x.shape[1]
            optimizer.zero_grad()
            out = model(x).view(batch_size, S, S, T_f)
            # print(" ================================== ")
            # print("x.shape : ", x.shape)
            # print("out.shape : ", out.shape)
            # print("y.shape : ", y.shape)

            mse = F.mse_loss(out, y, reduction="mean")

            temp_step_loss = 0
            with torch.no_grad():
                for time in range(T_f):
                    k, l = out[..., time], y[..., time]
                    temp_step_loss += myloss(k.view(batch_size, -1), l.view(batch_size, -1))
                train_l2_step += temp_step_loss.item()

            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward()

            optimizer.step()

            train_mse += mse.item()
            train_l2 += l2.item()

            del x, y, out, l2
            gc.collect()

        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()

        train_l2_step /= ntrain * T_f
        if ep % 2 == 1:
            t12 = default_timer()
            print("epochs", ep, "time", t12 - t1, "train_loss ", train_l2_step)
            continue

        model.eval()
        val_l2_step = 0.0
        test_l2 = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                batch_size = x.shape[0]
                out = model(x).view(batch_size, S, S, T_f)
                # out = y_normalizer.decode(out)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1))

                temp_step_loss = 0
                for time in range(T_f):
                    k, l = out[..., time], y[..., time]
                    temp_step_loss += myloss(k.view(batch_size, -1), l.view(batch_size, -1))

                val_l2_step += temp_step_loss.item()

                del x, y, out

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()

        print("2nd is per epoch time : ", ep, t2 - t1, train_mse, train_l2, test_l2)
        train_mse_list[ep] = train_mse
        train_l2_list[ep] = train_l2
        var_l2_list[ep] = test_l2

        gc.collect()
        val_l2_step /= nval * T_f

        t2 = default_timer()
        print(
            "epochs", ep, "time", t2 - t1, "train_loss ", train_l2_step, "Val_loss  ", val_l2_step
        )
        torch.cuda.empty_cache()
        if Min_error_t > val_l2_step:
            torch.save(model.state_dict(), weight_path)
            print("model saved", Min_error_t - val_l2_step)
            Min_error_t = val_l2_step

        # logging.info(
        # f" train_l2_step: { train_l2_step} "
        # )

    print("Traning Ended")
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    test_l2_step = 0.0
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            out = model(x).view(batch_size, S, S, T_f)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
            temp_step_loss = 0
            for time in range(T_f):
                k, l = out[..., time], y[..., time]
                temp_step_loss += myloss(k.view(batch_size, -1), l.view(batch_size, -1))

            test_l2_step += temp_step_loss.item()

            del x, y, out

    gc.collect()
    test_l2_step /= ntest * T_f
    test_l2 /= ntest

    # logging.info(ntest
    #     f"test_l2, test_l2_step: {test_l2, test_l2_step} "
    # )
    print("*** Test error: ", test_l2, test_l2_step)

    # ===============================================
    # Save the Results
    # ===============================================
    # pred = torch.zeros(test_u.shape) # [16, 64, 64, 40]

    pred = torch.zeros(ntest, test_u.shape[1], 2 * test_u.shape[2], test_u.shape[3]).to(device)

    test_l2_list = torch.zeros(T_f).to(device)
    with torch.no_grad():
        count = 0
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            out = model(x).squeeze()
            n, h, w, t = out.shape
            for t in range(T_f):
                test_l2_list[t] += myloss(
                    out[..., t].view(n, -1),
                    y[..., t].view(n, -1),
                ).item()

            # [16, 64, 128, 40]
            pred[count : count + n] = torch.cat((y, out), dim=2)
            count = count + n
        test_l2_list /= ntest

    scipy.io.savemat(pred_path, mdict={"pred": pred.cpu().numpy()})

    result_dict = {
        "train_mse": train_mse_list.to("cpu"),
        "train_l2": train_l2_list.to("cpu"),
        "var_l2": var_l2_list.to("cpu"),
        "test_l2_time": test_l2_list.to("cpu"),
    }
    save_pickle(result_dict, result_path)
