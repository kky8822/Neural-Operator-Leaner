import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from pathlib import Path
import random
from datetime import datetime

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights, VPTRFormerNAR
from model import GDL, MSELoss, L1Loss, GANLoss, BiPatchNCE
from utils import KTHDataset, BAIRDataset, MovingMNISTDataset, write_code_files
from utils import (
    VidCenterCrop,
    VidPad,
    VidResize,
    VidNormalize,
    VidReNormalize,
    VidCrop,
    VidRandomHorizontalFlip,
    VidRandomVerticalFlip,
    VidToTensor,
)
from utils import (
    visualize_batch_clips_inference,
    save_ckpt,
    load_ckpt,
    set_seed,
    AverageMeters,
    init_loss_dict,
    write_summary,
    resume_training,
)
from utils import set_seed, get_dataloader

import logging

from utils.metrics import PSNR, SSIM, LPIPS
import numpy as np
import pandas as pd

torch.backends.cudnn.benchmark = True

PSNR_metric = PSNR
SSIM_metric = SSIM()
LPIPS_metric = LPIPS


def NAR_show_samples(
    idx,
    VPTR_Enc,
    VPTR_Dec,
    VPTR_Transformer,
    sample,
    save_dir,
    renorm_transform,
    device=torch.device("cuda:0"),
    img_channels=1,
    pred_channels=1,
):
    VPTR_Transformer = VPTR_Transformer.eval()
    with torch.no_grad():
        past_frames, future_frames = sample
        past_frames = past_frames.to(device)
        future_frames = future_frames.to(device)

        past_gt_feats = VPTR_Enc(past_frames[:, :, 0:img_channels, ...])
        future_gt_feats = VPTR_Enc(future_frames[:, :, 0:img_channels, ...])

        rec_past_frames = VPTR_Dec(past_gt_feats)
        rec_future_frames = VPTR_Dec(future_gt_feats)

        pred_future_feats = VPTR_Transformer(past_gt_feats)
        pred_future_frames = VPTR_Dec(pred_future_feats)
        # print(pred_future_frames.shape)

        gt_frames = torch.cat(
            (
                past_frames[:, :, 0:pred_channels, ...],
                future_frames[:, :, 0:pred_channels, ...],
            ),
            dim=1,
        )
        ae_frames = torch.cat((rec_past_frames, rec_future_frames), dim=1)
        pred_frames = torch.cat(
            (past_frames[:, :, 0:pred_channels, ...], pred_future_frames), dim=1
        )

    N, T, C, X, Y = gt_frames.shape
    psnr = torch.zeros((T, C))
    ssim = torch.zeros((T, C))

    for c in range(C):
        for t in range(T):
            psnr[t, c] = PSNR_metric(
                gt_frames[:, t, c, ...].unsqueeze(1),
                pred_frames[:, t, c, ...].unsqueeze(1),
            )
            ssim[t, c] = SSIM_metric(
                gt_frames[:, t, c, ...].unsqueeze(1),
                pred_frames[:, t, c, ...].unsqueeze(1),
            )

    psnr = psnr.cpu()
    ssim = ssim.cpu()

    N = pred_frames.shape[0]
    # idx = min(N, 4)
    if idx == 0:
        visualize_batch_clips_inference(
            idx,
            gt_frames[0:N, ...],
            ae_frames[0:N, ...],
            pred_frames[0:N, ...],
            save_dir,
            renorm_transform,
            desc="NAR",
        )

    del gt_frames
    del ae_frames
    del pred_frames
    # torch.cuda.empty_cache()  # GPU 캐시 데이터 삭제

    # return psnr_sol_t, psnr_w_t, ssim_sol_t, ssim_w_t, lpips_sol_t, lpips_w_t
    return psnr, ssim


if __name__ == "__main__":
    set_seed(2021)

    if sys.argv[1] == "ns":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0722_NS_10_NAR_MSEGDL_ckpt")
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0722_NS_10_NAR_MSEGDL_tensorboard"
        )
        resume_AE_ckpt = Path(
            "/kky/VPTR/VPTR_ckpts/0722_NS_10_ResNetAE_MSEGDLgan_ckpt"
        ).joinpath("epoch_100.tar")
        device = torch.device("cuda:0")
        data_set_name = "Navier-Stokes"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-255/pngs"
        num_past_frames = 10
        num_future_frames = 10
        test_past_frames = 10
        test_future_frames = 10
    elif sys.argv[1] == "nsbd":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0723_NSBD_10_NAR_MSEGDL_ckpt")
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0723_NSBD_10_NAR_MSEGDL_tensorboard"
        )
        resume_AE_ckpt = Path(
            "/kky/VPTR/VPTR_ckpts/0723_NSBD_10_ResNetAE_MSEGDLgan_ckpt"
        ).joinpath("epoch_20.tar")
        device = torch.device("cuda:0")
        data_set_name = "Navier-Stokes-BD"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-255-12000/pngs"
        num_past_frames = 10
        num_future_frames = 10
        test_past_frames = 10
        test_future_frames = 10
    elif sys.argv[1] == "kth":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0721_KTH_NAR_MSEGDL_ckpt")
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0721_KTH_NAR_MSEGDL_tensorboard"
        )
        resume_AE_ckpt = Path(
            "/kky/VPTR/VPTR_ckpts/0721_KTH_ResNetAE_MSEGDLgan_ckpt"
        ).joinpath("epoch_50.tar")
        data_set_name = "KTH"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/kth_action/pngs"
        device = torch.device("cuda:1")
        num_past_frames = 10
        num_future_frames = 40
        test_past_frames = 10
        test_future_frames = 40

    elif sys.argv[1] == "nsbd-field":
        ckpt_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0804_NSBDField_RK4_40_NAR_MSEGDL_ckpt"
        )
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0804_NSBDField_RK4_40_NAR_MSEGDL_tensorboard"
        )
        resume_AE_ckpt = Path(
            "/kky/VPTR/VPTR_ckpts/0728_NSBDField_40_ResNetAE_MSEGDLgan_ckpt"
        ).joinpath("epoch_23.tar")
        device = torch.device("cuda:1")
        data_set_name = "Navier-Stokes-field-BD"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-255-field-12000"
        num_past_frames = 10
        num_future_frames = 40
        test_past_frames = 10
        test_future_frames = 40
        img_channels = 2  # 3 channels for BAIR datset
        pred_channels = 2  # 1: w, 4: w, dw/dx, dw/dy, d2w/dxdy
        window_size = 4
        resume_ckpt = os.path.join(ckpt_save_dir, "epoch_36.tar")
        # resume_ckpt = None
        rk = "rk4"
        transform_norm = "min_max"
        visc_list = "total"

    elif sys.argv[1] == "nsbd-field-test":
        ckpt_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0727_test_NSBDField_10_NAR_MSEGDL_ckpt"
        )
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0727_test_NSBDField_10_NAR_MSEGDL_tensorboard"
        )
        resume_AE_ckpt = Path(
            "/kky/VPTR/VPTR_ckpts/0727_test_NSBDField_10_ResNetAE_MSEGDLgan_ckpt"
        ).joinpath("epoch_1.tar")
        device = torch.device("cuda:1")
        data_set_name = "Navier-Stokes-field-BD-test"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-255-field-12000"
        num_past_frames = 10
        num_future_frames = 10
        test_past_frames = 10
        test_future_frames = 10
        img_channels = 2  # 3 channels for BAIR datset
        window_size = 4
        resume_ckpt = None

    elif sys.argv[1] == "nsbd-total":
        ckpt_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0812_NSBDTotalMinMax_40_NAR_MSEGDL_ckpt"
        )
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0812_NSBDTotalMinMax_40_NAR_MSEGDL_tensorboard"
        )
        resume_AE_ckpt = Path(
            "/kky/VPTR/VPTR_ckpts/0811_NSBDTotalMinMax_40_ResNetAE_MSEGDLlsgan_ckpt"
        ).joinpath("epoch_12.tar")
        device = torch.device("cuda:0")
        data_set_name = "Navier-Stokes-total-BD"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-total-12000"
        num_past_frames = 10
        num_future_frames = 40
        test_past_frames = 10
        test_future_frames = 40
        window_size = 4
        resume_ckpt = os.path.join(ckpt_save_dir, "epoch_21.tar")
        # resume_ckpt = None
        rk = False
        img_channels = 7  # 3 channels for BAIR datset
        pred_channels = 7  # 1: w, 4: w, dw/dx, dw/dy, d2w/dxdy
        transform_norm = "min_max"
        visc_list = "total"
        # visc_list = [sys.argv[2]]

    #############Set the logger#########
    if not Path(ckpt_save_dir).exists():
        Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%a, %d %b %Y %H:%M:%S",
        format="%(asctime)s - %(message)s",
        filename=ckpt_save_dir.joinpath("train_log.log").absolute().as_posix(),
        filemode="a",
    )

    start_epoch = 0

    summary_writer = SummaryWriter(tensorboard_save_dir.absolute().as_posix())
    encH, encW, encC = 8, 8, 528
    epochs = 100
    N = 64
    # AE_lr = 2e-4
    Transformer_lr = 1e-4
    max_grad_norm = 1.0
    TSLMA_flag = False
    rpe = False
    # padding_type = 'zero'

    lam_gan = None  # 0.001
    lam_pc = 0.1

    show_example_epochs = 1
    save_ckpt_epochs = 1

    #####################Init Dataset ###########################

    train_loader, val_loader, test_loader, renorm_transform = get_dataloader(
        data_set_name,
        N,
        dataset_dir,
        num_past_frames,
        num_future_frames,
        transform_norm=transform_norm,
        visc_list=visc_list,
    )

    #####################Init model###########################
    VPTR_Enc = VPTREnc(img_channels, feat_dim=encC, n_downsampling=3).to(device)
    VPTR_Dec = VPTRDec(
        pred_channels, feat_dim=encC, n_downsampling=3, out_layer="Tanh"
    ).to(device)
    VPTR_Enc = VPTR_Enc.eval()
    VPTR_Dec = VPTR_Dec.eval()

    VPTR_Disc = None
    # VPTR_Disc = VPTRDisc(img_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(device)
    # VPTR_Disc = VPTR_Disc.eval()
    # init_weights(VPTR_Disc)
    init_weights(VPTR_Enc)
    init_weights(VPTR_Dec)

    VPTR_Transformer = VPTRFormerNAR(
        num_past_frames,
        num_future_frames,
        encH=encH,
        encW=encW,
        d_model=encC,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=8,
        dropout=0.1,
        window_size=window_size,
        Spatial_FFN_hidden_ratio=4,
        TSLMA_flag=TSLMA_flag,
        rpe=rpe,
        device=device,
        rk=rk,
    ).to(device)
    optimizer_D = None
    # optimizer_D = torch.optim.Adam(params = VPTR_Disc.parameters(), lr = Transformer_lr, betas = (0.5, 0.999))
    optimizer_T = torch.optim.AdamW(
        params=VPTR_Transformer.parameters(), lr=Transformer_lr
    )

    Transformer_parameters = sum(
        p.numel() for p in VPTR_Transformer.parameters() if p.requires_grad
    )
    print(f"NAR Transformer num_parameters: {Transformer_parameters}")

    #####################Init loss function###########################
    loss_name_list = [
        "T_MSE",
        "T_GDL",
        "T_gan",
        "T_total",
        "T_bpc",
        "Dtotal",
        "Dfake",
        "Dreal",
    ]
    # gan_loss = GANLoss('vanilla', target_real_label=1.0, target_fake_label=0.0).to(device)
    bpnce = BiPatchNCE(N, num_future_frames, 8, 8, 1.0).to(device)
    loss_dict = init_loss_dict(loss_name_list)
    mse_loss = MSELoss()
    gdl_loss = GDL(alpha=1)

    # load the trained autoencoder, we initialize the discriminator from scratch, for a balanced training
    loss_dict, start_epoch = resume_training(
        {"VPTR_Enc": VPTR_Enc, "VPTR_Dec": VPTR_Dec},
        {},
        resume_AE_ckpt,
        loss_name_list,
        map_location=device,
    )

    if resume_ckpt is not None:
        loss_dict, start_epoch = resume_training(
            {"VPTR_Transformer": VPTR_Transformer},
            {"optimizer_T": optimizer_T},
            resume_ckpt,
            loss_name_list,
            map_location=device,
        )

    #####################Train ################################
    psnr = 0
    ssim = 0
    # lpips_sol_t = 0
    # lpips_w_t = 0
    for idx, sample in enumerate(test_loader):
        (
            b_psnr,
            b_ssim,
            # b_lpips_sol_t,
            # b_lpips_w_t,
        ) = NAR_show_samples(
            idx,
            VPTR_Enc,
            VPTR_Dec,
            VPTR_Transformer,
            sample,
            ckpt_save_dir.joinpath("re_inference_gifs_" + visc_list[0]),
            renorm_transform,
            device=device,
            img_channels=img_channels,  # 3 channels for BAIR datset
            pred_channels=pred_channels,  # 1: w, 4: w, dw/dx, dw/dy, d2w/dxdy
        )
        psnr += b_psnr / len(test_loader)
        ssim += b_ssim / len(test_loader)

        print(f"{idx}/{len(test_loader)}")
        # break

    columns = (
        ["t"]
        + ["psnr_" + str(i) for i in range(psnr.shape[1])]
        + ["ssim_" + str(i) for i in range(psnr.shape[1])]
    )
    data = np.hstack((psnr.numpy(), ssim.numpy()))
    data = np.hstack((np.arange(len(psnr)).reshape(-1, 1), data))

    df = pd.DataFrame(columns=columns, data=data)
    df.to_csv(
        ckpt_save_dir.joinpath("re_NAR_test_metrics_" + visc_list[0] + ".csv"),
        index=False,
    )
