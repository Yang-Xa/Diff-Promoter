import torch
import torch.nn as nn
from tqdm import tqdm
from controlnet import *
from utils import *
import numpy as np
from adv_model import Net as advNet
import csv

device = torch.device('cuda:3')

model = ControlNet().to(device)
model.load_state_dict(torch.load('ControlNet_param.pkl', map_location=device))
model.eval()

adv_model = advNet().to(device)
adv_model.load_state_dict(torch.load('adv_model_params.pkl', map_location=device))
adv_model.eval()


def cond_fn(x, target=10, guidance_loss_scale=10000):
    loss_fn = nn.MSELoss()
    tar = torch.tensor([target] * x.size(0), dtype=torch.float, device=x.device)
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        y = adv_model(x_in)
        # loss = torch.abs(target - y).mean()
        # loss = torch.square(target - y).mean()
        loss = loss_fn(y, tar)
        return -torch.autograd.grad(loss, x_in)[0] * guidance_loss_scale

timesteps = 1000
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)


def seqs2tensor(seqs, device=device):
    one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}

    seqs_tensor = []
    for seq in seqs:
        encode_seq = []
        for element in 'NNN' + seq.upper() + 'NNN':
            encode_seq.append(one_hot[element])
        seqs_tensor.append(torch.tensor(encode_seq, dtype=torch.float, device=device).t().unsqueeze(dim=0))
    return torch.cat(seqs_tensor, 0)

hot_one = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

import pandas as pd
dict_seqs = {}
df = pd.read_excel(r'gene_change_20bp_from_135.xlsx')
for row in df.index.values:
    dict_seqs[df.iloc[row, 1]] = df.iloc[row, 2].upper()


for gene, seq in dict_seqs.items():
    x_start = torch.randn(100, 4, 176, device=device)

    for idx in range(len(seq) - 6):
        c_seqs = [seq[:idx] + 'N'*6 + seq[idx+6:]] * 100
        c = seqs2tensor(c_seqs)

        #####ori#####
        generate_gen = gaussian_diffusion.sample(model, c, 176, batch_size=100, channels=4, cond=False, x_start=x_start)
        
        np.save("res/"+ gene +  "_" + str(idx) + "_ori.npy", generate_gen[-1])
        
        ori_seqs = []
        for seq in generate_gen[-1]:
            res = ''
            for a in seq.T[3:-3]:
                index = np.argmax(a)
                res += hot_one[index]
            ori_seqs.append(res)

        with open('res/' + gene + '_' + str(idx) + '_ori_gene.txt', 'w') as f:
            for i, seq in enumerate(ori_seqs):
                f.write('>gen_' + str(i) + '\n')
                f.write(seq + '\n')

        sss = torch.tensor(generate_gen[-1], dtype=torch.float, device=device)
        tensor_v = adv_model(sss).detach().cpu().numpy().tolist()

        sss_cc = seqs2tensor(ori_seqs)
        tensor2seq_v = adv_model(sss_cc).detach().cpu().numpy().tolist()

        with open('res/' + gene + '_' + str(idx) + '_ori_pred_v.csv', 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['gene', 'tensor_v', 'tensor2seq_v'])

            for i, (tv, tsv) in enumerate(list(zip(tensor_v, tensor2seq_v))):
                csv_writer.writerow(['gen_' + str(i), tv, tsv])

        #####last#####
        generate_gen = gaussian_diffusion.sample(model, c, 176, batch_size=100, channels=4, cond=True, cond_fn=cond_fn, x_start=x_start)

        np.save("res/"+ gene + "_" + str(idx) + "_last.npy", generate_gen[-1])

        last_seqs = []
        for seq in generate_gen[-1]:
            res = ''
            for a in seq.T[3:-3]:
                index = np.argmax(a)
                res += hot_one[index]
            last_seqs.append(res)

        with open('res/'+ gene + '_' + str(idx) + '_last_gene.txt', 'w') as f:
            for i, seq in enumerate(last_seqs):
                f.write('>gen_' + str(i) + '\n')
                f.write(seq + '\n')

        sss = torch.tensor(generate_gen[-1], dtype=torch.float, device=device)
        tensor_v = adv_model(sss).detach().cpu().numpy().tolist()

        sss_cc = seqs2tensor(last_seqs)
        tensor2seq_v = adv_model(sss_cc).detach().cpu().numpy().tolist()

        with open('res/'+ gene + '_' + str(idx) + '_last_pred_v.csv', 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['gene', 'tensor_v', 'tensor2seq_v'])

            for i, (tv, tsv) in enumerate(list(zip(tensor_v, tensor2seq_v))):
                csv_writer.writerow(['gen_' + str(i), tv, tsv])
        
    np.save("res/" + gene + "_input.npy", x_start.detach().cpu().numpy())


'''
#################### same input ####################
x_start = torch.randn(1800, 4, 176, device=device)
# np.save("input.npy", x_start.detach().cpu().numpy())
# x_start = np.load(r'/home/yxs/Diffusion/maize_promoter/v6_epoch1000_valid/008_evalution/m2/res/input.npy')
# x_start = torch.tensor(x_start, device=device)

import pandas as pd
c_seqs = []
df = pd.read_excel(r'gene_change_20bp_from_135.xlsx')
for row in df.index.values:
    tmp = df.iloc[row, 2].upper()
    c_seqs += [tmp[:135] + 'N' * 20 + tmp[155:]] * 200

c = seqs2tensor(c_seqs)


#################### no guidance ####################

generate_gen = gaussian_diffusion.sample(model, c, 176, batch_size=1800, channels=4, cond=False, x_start=x_start)

# for idx, seq in enumerate(generate_gen[-1]):
#     np.savetxt("res/" + str(idx) + ".csv", seq, delimiter=',')

np.save("res/ori.npy", generate_gen[-1])

hot_one = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

ori_seqs = []
for seq in generate_gen[-1]:
    res = ''
    for a in seq.T[3:-3]:
        index = np.argmax(a)
        res += hot_one[index]
    ori_seqs.append(res)

with open('res/ori_gene.txt', 'w') as f:
    for i, seq in enumerate(ori_seqs):
        f.write('>gen_' + str(i) + '\n')
        f.write(seq + '\n')

sss = torch.tensor(generate_gen[-1], dtype=torch.float, device=device)
tensor_v = adv_model(sss).detach().cpu().numpy().tolist()

sss_cc = seqs2tensor(ori_seqs)
tensor2seq_v = adv_model(sss_cc).detach().cpu().numpy().tolist()

with open('res/ori_pred_v.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['gene', 'tensor_v', 'tensor2seq_v'])

    for i, (tv, tsv) in enumerate(list(zip(tensor_v, tensor2seq_v))):
        csv_writer.writerow(['gen_' + str(i), tv, tsv])


#################### has guidance ####################

generate_gen = gaussian_diffusion.sample(model, c, 176, batch_size=1800, channels=4, cond=True, cond_fn=cond_fn, x_start=x_start)

# for idx, seq in enumerate(generate_gen[-1]):
#     np.savetxt("res/" + str(idx) + ".csv", seq, delimiter=',')

np.save("res/last.npy", generate_gen[-1])

hot_one = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

last_seqs = []
for seq in generate_gen[-1]:
    res = ''
    for a in seq.T[3:-3]:
        index = np.argmax(a)
        res += hot_one[index]
    last_seqs.append(res)

with open('res/last_gene.txt', 'w') as f:
    for i, seq in enumerate(last_seqs):
        f.write('>gen_' + str(i) + '\n')
        f.write(seq + '\n')

sss = torch.tensor(generate_gen[-1], dtype=torch.float, device=device)
tensor_v = adv_model(sss).detach().cpu().numpy().tolist()

sss_cc = seqs2tensor(last_seqs)
tensor2seq_v = adv_model(sss_cc).detach().cpu().numpy().tolist()

with open('res/last_pred_v.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['gene', 'tensor_v', 'tensor2seq_v'])

    for i, (tv, tsv) in enumerate(list(zip(tensor_v, tensor2seq_v))):
        csv_writer.writerow(['gen_' + str(i), tv, tsv])


#################### end guidance ####################

np.save("res/input.npy", x_start.detach().cpu().numpy())
'''
