import torch
from controlnet import *
from dataset import *
from utils import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import matplotlib.animation as animation


device = torch.device('cuda:3')
model = ControlNet().to(device)
# load locked param
ori_model_param = torch.load('model.pkl', map_location=device).state_dict()
model.locked.load_state_dict(ori_model_param)
# load trainable copy param
need_copy = ['downs', 'mid_block1', 'mid_attn', 'mid_block2']
new_param = {}
for key in ori_model_param:
    if key.split('.')[0] in need_copy:
        new_param['trainable_' + key] = ori_model_param[key]
model_param = model.state_dict()
model_param.update(new_param)
model.load_state_dict(model_param)
# frozen locked param, do not train
for name, param in model.named_parameters():
    if 'locked' in name:
        param.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

timesteps = 1000
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)

epochs = 500

train_loss = []

for epoch in range(epochs):
    tmp_loss = 0.0
    for step, (batch, conds) in enumerate(val_loader):

        batch_size = batch.size(0)
        batch = batch.to(device)
        conds = conds.to(device)

        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = gaussian_diffusion.train_losses(model, batch, t, conds)

        tmp_loss += (loss.item() * batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch " + str(epoch) + " have down!", flush=True)
    train_loss.append(tmp_loss / len(dataset_val))

torch.save(model.state_dict(), 'ControlNet_param.pkl')

plt.figure(figsize=(6, 5))
plt.plot(train_loss, 'ro-', label="Train Loss")
plt.legend()
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("res.png")
