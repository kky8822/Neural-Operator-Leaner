import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from PIL import Image
import os

ftotal_list = [
    "/kky/Neural-Operator-Leaner/fourier_neural_operator/gif/GT_V10000/0.gif",
    "/kky/Neural-Operator-Leaner/UNO/gif/V10000/0.gif",
    "/kky/Neural-Operator-Leaner/PDNO/gif/PDNO_V10000/0.gif",
    "/kky/Neural-Operator-Leaner/galerkin-transformer/gif/V10000/0.gif",
    "/kky/Neural-Operator-Leaner/fourier_neural_operator/gif/FNO_V10000/0.gif",
    "/kky/Neural-Operator-Leaner/fourier_neural_operator/gif/AFNO_V10000/0.gif",
]

dir_list = ["T", "UNO", "PDNO", "GT", "FNO", "AFNO"]


def iter_frames(gif, dir):
    try:
        # i = 0
        # while 1:
        #     im.seek(i)
        #     imframe = im.copy()
        #     if i == 0:
        #         palette = imframe.getpalette()
        #     else:
        #         imframe.putpalette(palette)
        #     yield imframe
        #     i += 1
        im = Image.open(gif)
        i = 0
        while 1:
            background = Image.new("RGB", im.size, (255, 255, 255))
            background.paste(im)
            background.save(os.path.join(dir, f"{i}.png"), "png", quality=100)

            i += 1
            im.seek(im.tell() + 1)
    except EOFError:
        pass


for f, dir in zip(ftotal_list, dir_list):
    # gif = f
    # img = Image.open(gif)
    # img.save(gif + ".png", "png", optimize=True, quality=100)
    iter_frames(f, os.path.join("png/V10000", dir))
    # for i, frame in enumerate(iter_frames(Image.open(f))):
    #     frame.save(os.path.join(f"png/V1000/{dir}", f"{i}.png"), "png", optimize=True, quality=100)
