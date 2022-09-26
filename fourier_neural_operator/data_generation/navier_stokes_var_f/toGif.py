from scipy import io
import torch
import torchvision.transforms as transforms
from torch import Tensor
from pathlib import Path


class VidReNormalize(object):
    def __init__(self, mean, std):
        try:
            self.inv_std = [1.0 / s for s in std]
            self.inv_mean = [-m for m in mean]
            self.renorm = transforms.Compose(
                [
                    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=self.inv_std),
                    transforms.Normalize(mean=self.inv_mean, std=[1.0, 1.0, 1.0]),
                ]
            )
        except TypeError:
            # try normalize for grey_scale images.
            self.inv_std = 1.0 / std
            self.inv_mean = -mean
            self.renorm = transforms.Compose(
                [
                    transforms.Normalize(mean=0.0, std=self.inv_std),
                    transforms.Normalize(mean=self.inv_mean, std=1.0),
                ]
            )

    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = self.renorm(clip[i, ...])

        return clip


def save_clip(clip, file_name):
    renorm_transform = VidReNormalize(0.0, 1.0)
    imgs = []

    print(clip.min(), clip.max())
    clip = renorm_transform(clip)
    print(clip.min(), clip.max())
    clip = torch.clamp(clip, min=0.0, max=1.0)
    print(clip.min(), clip.max())

    for i in range(clip.shape[0]):
        img = transforms.ToPILImage()(clip[i, ...])
        imgs.append(img)

    imgs[0].save(str(Path(file_name).absolute()), save_all=True, append_images=imgs[1:])


mat_file = io.loadmat("ns_data_V1e-4_N1000_0_T50.mat")

a = mat_file["a"]
f = mat_file["f"]
u = Tensor(mat_file["u"]).permute(0, 3, 1, 2)
u = u[:, :, None, ...]
u = (u + 7) / 14

for i in range(10):
    save_clip(u[i], f"test_{i}.gif")
