import torch
import math

from timeit import default_timer


class GaussianRF(object):
    def __init__(
        self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None
    ):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))

        k_max = size // 2

        if dim == 1:
            k = torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            )

            self.sqrt_eig = (
                size
                * math.sqrt(2.0)
                * sigma
                * ((4 * (math.pi**2) * (k**2) + tau**2) ** (-alpha / 2.0))
            )
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            ).repeat(size, 1)

            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers

            self.sqrt_eig = (
                (size**2)
                * math.sqrt(2.0)
                * sigma
                * (
                    (4 * (math.pi**2) * (k_x**2 + k_y**2) + tau**2)
                    ** (-alpha / 2.0)
                )
            )
            self.sqrt_eig[0, 0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            ).repeat(size, size, 1)

            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)

            self.sqrt_eig = (
                (size**3)
                * math.sqrt(2.0)
                * sigma
                * (
                    (4 * (math.pi**2) * (k_x**2 + k_y**2 + k_z**2) + tau**2)
                    ** (-alpha / 2.0)
                )
            )
            self.sqrt_eig[0, 0, 0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real


class GaussianF(object):
    def __init__(self, size, device=None):

        self.size = size
        self.device = device
        self.x = torch.linspace(0, 1, self.size + 1, device=device)[0:-1]
        self.y = torch.linspace(0, 1, self.size + 1, device=device)[0:-1]
        self.X, self.Y = torch.meshgrid(self.x, self.y)  # size,size

    def sample(self, N):
        w = (
            torch.randn(N, 4, device=self.device)
            .expand(self.size, self.size, N, 4)
            .permute(2, 3, 0, 1)
        )  # N, 4, size, size

        X = self.X.expand(N, self.size, self.size)
        Y = self.Y.expand(N, self.size, self.size)  # N, size, size
        X = w[:, 0, ...] * X + w[:, 1, ...]
        Y = w[:, 2, ...] * Y + w[:, 3, ...]  # N, size, size

        f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
        # print("f: ", f.shape)

        return f
