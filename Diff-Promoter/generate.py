import sys
sys.path.append('..')
import torch
from model import *
from utils import *
from dataset import *
import numpy as np
import os
from itertools import islice

device = torch.device('cuda:3')

timesteps = 1000
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)


files = os.listdir(r'../params/')

for filename in files:
    model = UNetModel()
    model.load_state_dict(torch.load(r'../params/' + filename))
    model.to(device)
    model.eval()

    g_seqs = generate_gen(model, gaussian_diffusion, len(dataset_train))

    names = filename.split('_')
    with open('./res/' + names[0] + '_' + names[1] + '_gene_seqs.fasta', 'w') as f:
        for idx, seq in enumerate(g_seqs):
            f.write('>gen_' + str(idx) + '\n')
            f.write(seq + '\n')

 
