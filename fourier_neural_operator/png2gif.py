from PIL import Image
import glob
import os, sys

import re
def natural_key(string_):
    """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

source_path = sys.argv[1]
save_path = sys.argv[2]


# Create the frames
frames = []
imgs = glob.glob(os.path.join(source_path, "*.png"))
imgs = sorted(imgs, key=natural_key)
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save(save_path, format='GIF',
              append_images=frames[1:],
              save_all=True,
              duration=300, loop=0)
