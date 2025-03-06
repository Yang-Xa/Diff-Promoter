import torch
from model import *
from dataset import *
from utils import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from scipy.stats import wasserstein_distance


device = torch.device('cuda:3')
model = UNetModel()
model.to(device)

timesteps = 1000
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

Epochs = 2000

train_loss = []
train_wd = []

for epoch in range(1, Epochs + 1):
    model.train()
    tmp_loss = 0.0
    for step, batch in enumerate(train_loader):

        batch_size = batch.size(0)
        batch = batch.to(device)

        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = gaussian_diffusion.train_losses(model, batch, t)

        tmp_loss += (loss.item() * batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch " + str(epoch) + " have down!", flush=True)
    train_loss.append(tmp_loss / len(dataset_train))

    if epoch % 50 == 0:
        model.eval()
        g_seqs = generate_gen(model, gaussian_diffusion, len(dataset_train))
        real_rate, gene_rate = getTATADistribution(list_promoter, g_seqs)
        wd = wasserstein_distance([i for i in range(164)], [i for i in range(164)], real_rate, gene_rate)
        train_wd.append(wd)
        print('epoch_'+str(epoch)+'_wd:' + str(wd), flush=True)
        torch.save(model.state_dict(), './params/epoch_'+str(epoch)+'_params.pkl')

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(train_loss, 'ro-', label="Train Loss")
plt.legend()
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_wd, 'bs-', label="Train Loss")
plt.legend()
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("wasserstein_distance")

plt.savefig("res.png")
