"""
(2+1)D Navier-Stokes equation + Galerkin Transformer
MIT license: Paper2394 authors, NeurIPS 2021 submission.
"""
# from libs_path import *
from libs import *
from libs.ns_lite import *

# torch.autograd.set_detect_anomaly(True)

get_seed(1127802)

dataset_name = sys.argv[1]
input_steps = int(sys.argv[2])
output_steps = int(sys.argv[3])
de_solver = sys.argv[4]
save_model_name = sys.argv[5]
save_result_name = sys.argv[6]
device_cuda = sys.argv[7]

device = torch.device(device_cuda if torch.cuda.is_available() else "cpu")

n_target = 1
train_batch = train_batch_ns_3d
validate_epoch = validate_epoch_ns_3d
attn_type = "galerkin"

data_path = os.path.join(DATA_PATH, dataset_name)
train_dataset = NavierStokesDatasetLite(
    data_path=data_path,
    train_data=True,
    time_steps_input=input_steps,
    time_steps_output=output_steps,
)
valid_dataset = NavierStokesDatasetLite(
    data_path=data_path,
    train_data=False,
    time_steps_input=input_steps,
    time_steps_output=output_steps,
)
batch_size = 8
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=False,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=4,
    pin_memory=False,
)

config = defaultdict(
    lambda: None,
    node_feats=10 + 2 + 1,
    pos_dim=3,
    n_targets=n_target,
    n_hidden=48,  # attention's d_model
    num_feat_layers=0,
    num_encoder_layers=4,
    n_head=1,
    dim_feedforward=96,
    attention_type=attn_type,
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
# model = FourierTransformer2DLite(**config)
model = FourierTransformer3DLite(**config)


print(get_num_params(model))

model = model.to(device)

epochs = 500
lr = 1e-3
max_lr = lr
h = 1 / 64

start_epoch = 0
# if resume_ckpt != "False":
#     model.load_state_dict(torch.load(os.path.join(MODEL_PATH, resume_ckpt)))
#     model.eval()
#     f = open(os.path.join(MODEL_PATH, resume_ckpt_result), "rb")
#     prev_result = pickle.load(f)
#     start_epoch = prev_result["best_val_epoch"]
#     lr = prev_result["lr_history"][start_epoch * 1024 // batch_size - 1]
#     epochs = epochs - start_epoch
#     max_lr = lr
# else:
#     start_epoch = 0


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = OneCycleLR(
    optimizer,
    max_lr=lr,
    div_factor=1e4,
    final_div_factor=1e4,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
)

loss_func = WeightedL2Loss2d(regularizer=True, h=h, gamma=0.1)

metric_func = WeightedL2Loss2d(regularizer=False, h=h)


result = run_train(
    model,
    loss_func,
    metric_func,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    train_batch=train_batch,
    validate_epoch=validate_epoch,
    epochs=epochs,
    patience=None,
    tqdm_mode="batch",
    mode="min",
    device=device,
    model_name=save_model_name,
    result_name=save_result_name,
    start_epoch=start_epoch,
    tqdm_switch=False,
)
"""
4 GT layers: 48 d_model
2 SC layers: 20 d_model for spectral conv with 12 Fourier modes
Total params: 862049

diag 0 + xavier 1e-2, encoder dp = ffn dp = 5e-2
    3.406e-03 at epoch 99

diag 1e-2 + xavier 1e-2, encoder dp 0, ffn dp = 5e-2
    3.078e-03 at epoch 100
"""
