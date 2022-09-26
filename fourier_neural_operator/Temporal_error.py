from utilities3 import MatReader
import numpy as np
import torch
import matplotlib.pyplot as plt


PATHs = ['./pred/ns_fourier_3d_rnn_V10000_T50_N1000_ep500_m12_w20.mat',
'./pred/ns_fourier_3d_rnn_extrapolation_10step_V10000_T50_N1000_ep500_m12_w20.mat']

for PATH in PATHs:
    reader = MatReader(PATH)

    truths = reader.read_field('u')
    preds = reader.read_field('pred')

    print(torch.max(truths))
    print(torch.min(truths))

    batch_size = truths.shape[0]
    time_steps = truths.shape[-1]

    error = []
    for t in range(time_steps):
        e = 0
        for b in range(batch_size):
            truth = truths[b,:,:,t]
            pred = preds[b,:,:,t]
            e += torch.sum((pred - truth)**2)
        error.append(e/batch_size/64/64)
    plt.plot(error)

plt.savefig("test.png")
    
    

