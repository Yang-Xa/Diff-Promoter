import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim

from dataset import train_loader
from model import Generator, Discriminator


def generate_gen(num_sequences, epoch_now, model, device):
    sequences = []
    hot_one = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    for i in range(num_sequences):
        z = torch.randn(64, 128).to(device)
        gene_img = model(z).detach().cpu()

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


def train(num_epochs, dataloader, device, netG, netD, criterion, optimizerG, optimizerD, real_label, fake_label):
    G_losses = []
    D_losses = []
    D_X_ALL = []
    D_G_X_ALL = []
    Gen_ACC = []
    Real_ACC = []

    for epoch in range(num_epochs):
        D_X_for_this_epoch = []
        D_G_z1_for_this_epoch = []
        D_G_z2_for_this_epoch = []
        errD_for_this_epoch = []
        errG_for_this_epoch = []
        Gen_ACC_for_this_epoch = []
        Real_ACC_for_this_epoch = []
        for i, data in enumerate(dataloader, 0):
            if i % 5 == 0:
                optimizerD.zero_grad()
                real_gpu = data.to(device)
                b_size = real_gpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                output = netD(real_gpu).view(-1)
                threshold = 0.5
                predicted_labels = (output >= threshold).float()
                num_real_correct = predicted_labels.sum().item()
                Real_ACC_for_this_epoch.append(num_real_correct / b_size)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()
                D_X_for_this_epoch.append(D_x)

                noise = torch.randn(b_size, 128, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach()).view(-1)
                predicted_labels_fake = (output >= threshold).float()
                num_fake_correct = predicted_labels_fake.sum().item()
                Gen_ACC_for_this_epoch.append(num_fake_correct / b_size)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                D_G_z1_for_this_epoch.append(D_G_z1)
                errD = errD_real + errD_fake
                errD_for_this_epoch.append(errD.item())
                optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            noise = torch.randn(b_size, 128, device=device)
            fake = netG(noise)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG_for_this_epoch.append(errG.item())
            errG.backward()
            D_G_z2 = output.mean().item()
            D_G_z2_for_this_epoch.append(D_G_z2)
            optimizerG.step()

        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, np.mean(errD_for_this_epoch), np.mean(errG_for_this_epoch),
                 np.mean(D_X_for_this_epoch), np.mean(D_G_z1_for_this_epoch), np.mean(np.mean(D_G_z2_for_this_epoch))))

        G_losses.append(np.mean(errG_for_this_epoch))
        D_losses.append(np.mean(errD_for_this_epoch))
        D_X_ALL.append(np.mean(D_X_for_this_epoch))
        D_G_X_ALL.append((np.mean(D_G_z1_for_this_epoch)+np.mean(np.mean(D_G_z2_for_this_epoch))/2))
        Gen_ACC.append(np.mean(Gen_ACC_for_this_epoch))
        Real_ACC.append(np.mean(Real_ACC_for_this_epoch))

        if (epoch+1) % 25 == 0:
            torch.save(netG, './model_save/G_save/netG_epoch_%d.pth' % (epoch+1))
            netG.eval()
            generate_gen(num_sequences=len(train_loader), epoch_now=epoch + 1, model=netG, device=device)
            netG.train()

    return G_losses, D_losses, D_X_ALL, D_G_X_ALL, Gen_ACC, Real_ACC


def draw_pic(G_losses, D_losses, D_X_ALL, D_G_X_ALL, Gen_ACC, Real_ACC):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Losses')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(D_X_ALL, label='D(x) - Real Samples')
    plt.plot(D_G_X_ALL, label='D(G(z)) - Fake Samples')
    plt.xlabel('Iterations')
    plt.ylabel('Output Mean')
    plt.title('Discriminator Output Means')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(Gen_ACC, label='Generator Accuracy')
    plt.plot(Real_ACC, label='Discriminator Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Generator and Discriminator Accuracies')
    plt.legend()

    plt.tight_layout()

    os.makedirs('./pic_save', exist_ok=True)
    plt.savefig('./pic_save/1.png')


if __name__ == '__main__':
    os.makedirs('./model_save/G_save', exist_ok=True)
    os.makedirs('./model_save/D_save', exist_ok=True)
    os.makedirs('./gen_save', exist_ok=True)
    device = torch.device("cuda:3" if (torch.cuda.is_available()) else "cpu")
    Generator = Generator().to(device)
    Discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(Generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    optimizerD = optim.Adam(Discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    G_losses, D_losses, D_X_ALL, D_G_X_ALL, Gen_ACC, Real_ACC = train(2000, train_loader, device, Generator,
                                                                      Discriminator, criterion, optimizerG, optimizerD,
                                                                      real_label=1, fake_label=0)
    draw_pic(G_losses, D_losses, D_X_ALL, D_G_X_ALL, Gen_ACC, Real_ACC)
