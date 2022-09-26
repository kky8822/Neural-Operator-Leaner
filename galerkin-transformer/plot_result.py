import pickle
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from libs import *
from libs.ns_lite import *
from PIL import Image


MODEL_PATH = "models"
INFER_PATH = "inference"
DATA_PATH = "data"
batch_size = 32
device_name = "cuda:1"

visc_list = ["V1000", "V10000", "Vmix"]
force_list = ["Ffix", "Fvar"]
pred_list = [10, 40]
de_solver_list = [""]


def inference2d(batch_size, df, mf, in_s, out_s, de_solver, device_name):
    data_path = os.path.join(DATA_PATH, df)
    valid_dataset = NavierStokesDatasetLite(
        data_path=data_path,
        train_data=False,
        time_steps_input=in_s,
        time_steps_output=out_s,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    config = defaultdict(
        lambda: None,
        node_feats=10 + 2,
        pos_dim=2,
        n_targets=1,
        n_hidden=48,  # attention's d_model
        num_feat_layers=0,
        num_encoder_layers=4,
        n_head=1,
        dim_feedforward=96,
        attention_type="galerkin",
        feat_extract_type=None,
        xavier_init=0.01,
        diagonal_weight=0.01,
        layer_norm=True,
        attn_norm=False,
        return_attn_weight=False,
        return_latent=False,
        decoder_type="ifft",
        freq_dim=20,  # hidden dim in the frequency domain
        num_regressor_layers=2,  # number of spectral layers
        fourier_modes=12,  # number of Fourier modes
        spacial_dim=2,
        spacial_fc=False,
        dropout=0.0,
        encoder_dropout=0.0,
        decoder_dropout=0.0,
        ffn_dropout=0.05,
        debug=False,
        de_solver=de_solver,
    )

    # config = defaultdict(
    #     lambda: None,
    #     node_feats=10 + 2 + 1,
    #     pos_dim=2,
    #     temp_in_dim=1,
    #     n_targets=1,
    #     n_hidden=48,  # attention's d_model
    #     num_feat_layers=0,
    #     num_encoder_layers=4,
    #     n_head=8,
    #     dim_feedforward=96,
    #     attention_type="galerkin",
    #     feat_extract_type=None,
    #     xavier_init=0.01,
    #     diagonal_weight=0.01,
    #     layer_norm=True,
    #     attn_norm=False,
    #     return_attn_weight=False,
    #     return_latent=False,
    #     decoder_type="ifft",
    #     freq_dim=20,  # hidden dim in the frequency domain
    #     num_regressor_layers=2,  # number of spectral layers
    #     fourier_modes=12,  # number of Fourier modes
    #     spacial_dim=2,
    #     spacial_fc=False,
    #     dropout=0.0,
    #     encoder_dropout=0.0,
    #     decoder_dropout=0.0,
    #     ffn_dropout=0.05,
    #     debug=False,
    #     de_solver=de_solver,
    #     device=device,
    # )

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()
    model = FourierTransformer2DLite(**config)
    # model = FourierTransformer3DLite(**config)
    print(get_num_params(model))
    model = model.to(device)

    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, mf)))

    metric_func = WeightedL2Loss2d(regularizer=False, h=1 / 64)

    u_true, u_pred, metric_val = validate_epoch_ns_metric_and_plot(
        model, metric_func, valid_loader, batch_size, out_s, device
    )

    u_true = u_true.permute(0, 3, 1, 2)
    u_pred = u_pred.permute(0, 3, 1, 2)

    u_true = u_true.cpu().numpy()
    u_pred = u_pred.cpu().numpy()
    metric_val = metric_val.cpu().numpy()

    return u_true, u_pred, metric_val


def inference3d(batch_size, df, mf, in_s, out_s, de_solver, device_name):
    data_path = os.path.join(DATA_PATH, df)
    valid_dataset = NavierStokesDatasetLite(
        data_path=data_path,
        train_data=False,
        time_steps_input=in_s,
        time_steps_output=out_s,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    config = defaultdict(
        lambda: None,
        node_feats=10 + 2 + 1,
        pos_dim=3,
        n_targets=1,
        n_hidden=48,  # attention's d_model
        num_feat_layers=0,
        num_encoder_layers=4,
        n_head=1,
        dim_feedforward=96,
        attention_type="galerkin",
        feat_extract_type=None,
        xavier_init=0.01,
        diagonal_weight=0.01,
        layer_norm=True,
        attn_norm=False,
        return_attn_weight=False,
        return_latent=False,
        decoder_type="ifft",
        freq_dim=20,  # hidden dim in the frequency domain
        num_regressor_layers=2,  # number of spectral layers
        fourier_modes=12,  # number of Fourier modes
        spacial_dim=2,
        spacial_fc=False,
        dropout=0.0,
        encoder_dropout=0.0,
        decoder_dropout=0.0,
        ffn_dropout=0.05,
        debug=False,
        de_solver=de_solver,
        device=device,
    )

    torch.cuda.empty_cache()
    model = FourierTransformer3DLite(**config)
    print(get_num_params(model))
    model = model.to(device)

    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, mf)))

    metric_func = WeightedL2Loss2d(regularizer=False, h=1 / 64)

    u_true, u_pred = validate_epoch_ns_3d_metric_and_plot(
        model, metric_func, valid_loader, batch_size, out_s, device
    )

    u_true = u_true.permute(0, 3, 1, 2)
    u_pred = u_pred.permute(0, 3, 1, 2)

    u_true = u_true.cpu().numpy()
    u_pred = u_pred.cpu().numpy()
    # metric_val = metric_val.cpu().numpy()

    return u_true, u_pred


def save_gifs(u_true, u_pred, fname):
    # data = np.concatenate((u_true, u_pred), axis=-1)
    # cm = plt.get_cmap("RdYlBu")
    cm = plt.get_cmap("jet")
    colored_data = [cm(d)[:, :, :3] * 255 for d in u_pred]
    imgs = [Image.fromarray(d.astype(np.uint8)) for d in colored_data]
    imgs[0].save(fname, save_all=True, append_images=imgs[1:], duration=100, loop=0)


# batch_size = 64
# df = "ns_V1000_N5000_T50.mat"
# mf = "model_GT3d_ns_V1000_Ffix_T1024_V200_10to40_500"
# in_s = 10
# out_s = 40
# de_solver = "euler"
# device_name = "cuda:0"

min_V1000 = -2.1
max_V1000 = 2.1
min_V10000 = -4.2
max_V10000 = 4.2

u_true, b_u_pred = inference3d(
    batch_size=64,
    df="ns_V10000_N5000_T50.mat",
    mf="model_GT3d_ns_V10000_Ffix_T1024_V200_10to40_500.pt",
    in_s=10,
    out_s=40,
    de_solver="euler",
    device_name="cuda:0",
)

gif_dir = "gif/V10000"
for idx, u_pred in enumerate(b_u_pred):
    min_u = min_V10000
    max_u = max_V10000
    u_transform = (u_pred - min_u) / (max_u - min_u)
    save_gifs(
        u_true,
        u_transform,
        os.path.join(gif_dir, f"{idx}.gif"),
    )
# print(metric_val)


# fig, ax = plt.subplots(
#     figsize=(11.75, 8.25 * 2), nrows=len(force_list) * 2, ncols=len(visc_list)
# )
# for i, force in enumerate(force_list):
#     for j, visc in enumerate(visc_list):
#         for pred, c in zip(pred_list, ["r", "b"]):
#             for de_solver, s, de_solver_idx in zip(de_solver_list, ["-"], ["euler"]):

#                 # plot loss during training
#                 rf = (
#                     f"result_GT2d_ns_{visc}_{force}_T1024_V200_10to{pred}{de_solver}.pt"
#                 )
#                 fpath = os.path.join(MODEL_PATH, rf)
#                 if os.path.isfile(fpath):
#                     f = open(os.path.join(MODEL_PATH, rf), "rb")
#                     data = pickle.load(f)
#                     print(
#                         f"{rf} {data['best_val_epoch']} {data['best_val_metric']} {len(data['loss_val'])}"
#                     )
#                     ax[i, j].plot(data["loss_val"], c + s, label=rf)
#                     ax[i, j].plot(data["loss_train"][:, 0], c + "--", label=rf)

#                 # inference
#                 if visc == "Vmix":
#                     if force == "Ffix":
#                         df = f"ns_{visc}_N3000_T50.mat"
#                     elif force == "Fvar":
#                         df = f"ns_{visc}_N3000_T50_var_f.mat"
#                 else:
#                     if force == "Ffix":
#                         df = f"ns_{visc}_N5000_T50.mat"
#                     elif force == "Fvar":
#                         df = f"ns_{visc}_N5000_T50_var_f.mat"
#                 mf = f"model_GT2d_ns_{visc}_{force}_T1024_V200_10to{pred}{de_solver}.pt"

#                 if os.path.isfile(os.path.join(DATA_PATH, df)) and os.path.isfile(
#                     os.path.join(MODEL_PATH, mf)
#                 ):
#                     b_u_true, b_u_pred, metric_val = inference2d(
#                         batch_size, df, mf, 10, pred, de_solver_idx, device_name
#                     )
#                     ax[i + len(force_list), j].plot(metric_val, c + s, label=rf)
#                     N, T, H, W = b_u_true.shape
#                     for idx, (u_true, u_pred) in enumerate(zip(b_u_true, b_u_pred)):
#                         gif_dir = os.path.join(
#                             INFER_PATH,
#                             f"GT2d_ns_{visc}_{force}_T1024_V200_10to{pred}{de_solver}",
#                         )
#                         if os.path.isdir(gif_dir) is False:
#                             os.mkdir(gif_dir)
#                         save_gifs(
#                             u_true,
#                             u_pred,
#                             os.path.join(gif_dir, f"{idx}.gif"),
#                         )

#         ax[i, j].set_ylim([1e-3, 1])
#         # ax[i, j].legend()
#         ax[i, j].grid()
#         ax[i, j].semilogy()

#         ax[i + len(force_list), j].set_ylim([1e-3, 1])
#         # ax[i + len(force_list), j].legend()
#         ax[i + len(force_list), j].grid()
#         ax[i + len(force_list), j].semilogy()

# plt.savefig(os.path.join(INFER_PATH, "val_loss_GT2d.png"))
