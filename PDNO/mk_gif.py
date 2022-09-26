import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.io
import os
import torch


def save_gifs(u_pred, fname):
    data = u_pred
    # cm = plt.get_cmap("RdYlBu")
    cm = plt.get_cmap("jet")
    colored_data = [cm(d)[:, :, :3] * 255 for d in data]
    imgs = [Image.fromarray(d.astype(np.uint8)) for d in colored_data]
    imgs[0].save(fname, save_all=True, append_images=imgs[1:], duration=100, loop=0)


min_V1000 = -2.1
max_V1000 = 2.1
min_V10000 = -4.2
max_V10000 = 4.2

# gif_file = "results/NS_3d_infer/mk_gif_V10000/pred.mat"
# gif_dir = "gif/PDNO_V10000"
gif_file = "results/NS_3d_infer/V10000_superR/pred.mat"
gif_dir = "gif/PDNO_V10000_superR"
b_u_pred = torch.Tensor(scipy.io.loadmat(gif_file)["pred"]).permute(0, 3, 1, 2).numpy()

for idx, u_pred in enumerate(b_u_pred[:, :, :, :]):
    min_u = min_V10000
    max_u = max_V10000
    u_transform = (u_pred - min_u) / (max_u - min_u)
    save_gifs(
        u_transform,
        os.path.join(gif_dir, f"{idx}.gif"),
    )
