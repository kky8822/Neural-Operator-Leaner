import os, sys
from cv2 import Mat
import matplotlib.pyplot as plt
from utilities3 import *


matPath = sys.argv[1]
savePath = sys.argv[2]
types = ["u", "pred"]

reader = MatReader(matPath)

for type in types:
    data = reader.read_field(type)
    
    for i, d_t in enumerate(data):
        if i >50:
            break

        path = os.path.join(savePath, type, str(i))
        timesteps = d_t.shape[-1]
        print(timesteps)
        os.makedirs(path, exist_ok=True)
        for t in range(timesteps):
            plt.clf()
            plt.cla()
            
            plt.imshow(d_t[:,:,t])
            plt.savefig(os.path.join(path, f'{t}.png'))


