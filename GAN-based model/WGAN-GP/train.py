import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model import Generator, Discriminator
from dataset import train_loader


batch_size = 64
num_epochs = 2000
lambda_gp = 20
n_critic = 5
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

generator = Generator().to(device)
discriminator = Discriminator().to(device)

initial_lr_G = 1e-6
max_lr_G = 5e-4
initial_lr_D = 1e-6
max_lr_D = 5e-5
warmup_steps = 500

optimizer_G = optim.Adam(generator.parameters(), lr=initial_lr_G, betas=(0.5, 0.9))
optimizer_D = optim.Adam(discriminator.parameters(), lr=initial_lr_D, betas=(0.5, 0.9))


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1, 1)).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.shape[0], 1).to(real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


def generate_gen(num_sequences, epoch_now, device=device):
    sequences = []
    hot_one = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    for i in range(num_sequences):
        z = torch.randn(64, 128).to(device)
        gene_img = generator(z).detach().cpu()

        for gImg in gene_img:
            res = ''
            for a in gImg.T:
                index = torch.argmax(a)
                res += hot_one[index.item()]
            sequences.append(res)

    with open(f'./gen_save/gen_{epoch_now}.fasta', 'w') as file:
        for i, seq in enumerate(sequences, start=1):
            file.write(f'>gene_{i}\n')
            file.write(seq + '\n')


Loss_G = []
Loss_D = []
g_loss = torch.tensor(0.0)
d_loss = torch.tensor(0.0)
os.makedirs('./model_save/G_save', exist_ok=True)
os.makedirs('./model_save/D_save', exist_ok=True)
os.makedirs('./gen_save', exist_ok=True)
for epoch in range(num_epochs):
    if epoch < warmup_steps:
        lr_D = initial_lr_D + (max_lr_D - initial_lr_D) * epoch / warmup_steps
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = lr_D

        lr_R = initial_lr_G + (max_lr_G - initial_lr_G) * epoch / warmup_steps
        for param_group in optimizer_G.param_groups:
            param_group['lr'] = lr_R

    loss_G_for_this_epoch = []
    loss_D_for_this_epoch = []
    for i, (real_samples) in enumerate(train_loader):
        real_samples = real_samples.to(device)
        z = torch.randn((batch_size, 128)).to(device)

        optimizer_D.zero_grad()
        fake_samples = generator(z).detach()

        real_validity = discriminator(real_samples)
        fake_validity = discriminator(fake_samples)

        gradient_penalty = compute_gradient_penalty(discriminator, real_samples.data, fake_samples.data)

        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty*lambda_gp

        d_loss.backward()
        optimizer_D.step()
        loss_D_for_this_epoch.append(d_loss.item())

        if i % n_critic == 0:
            optimizer_G.zero_grad()
            gen_z = torch.randn((batch_size, 128)).to(device)
            gen_imgs = generator(gen_z)
            g_loss = -torch.mean(discriminator(gen_imgs))

            g_loss.backward()
            optimizer_G.step()
            loss_G_for_this_epoch.append(g_loss.item())

    print(f"[Epoch {epoch}/{num_epochs}]"
          f"[D loss: {np.mean(loss_D_for_this_epoch):.4f}] [G loss: {np.mean(loss_G_for_this_epoch):.4f}]")

    Loss_D.append(np.mean(loss_D_for_this_epoch))
    Loss_G.append(np.mean(loss_G_for_this_epoch))

    if (epoch + 1) % 25 == 0:
        torch.save(generator, './model_save/G_save/netG_epoch_%d.pth' % (epoch + 1))
        generator.eval()
        generate_gen(num_sequences=len(train_loader), epoch_now=epoch + 1)
        generator.train()

    if (epoch + 1) % 100 == 0:
        torch.save(discriminator, './model_save/D_save/netD_epoch_%d.pth' % (epoch + 1))

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(Loss_G, label="G Loss")
plt.plot(Loss_D, label="D Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss.png')

print(f"Length of Loss_D: {len(Loss_D)}")
print(f"Length of Loss_G: {len(Loss_G)}")
