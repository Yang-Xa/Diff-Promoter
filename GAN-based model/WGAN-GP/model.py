import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(128, 64 * 11))

        self.l1 = nn.Sequential(
            nn.ConvTranspose1d(64, 128, 2, stride=2, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )

        self.l2 = nn.Sequential(
            nn.ConvTranspose1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )

        self.l3 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )

        self.l4 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 2, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )

        self.last = nn.Sequential(
            nn.Conv1d(64, 4, 1, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.fc1(z)
        z = z.view(z.size(0), 64, 11)
        z = self.l1(z)
        z = self.l2(z)
        z = self.l3(z)
        z = self.l4(z)

        z = self.last(z)

        return z


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv1d(4, 64, 4, stride=2, padding=1),
            nn.LayerNorm([64, 85]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.l2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.LayerNorm([128, 43]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.l3 = nn.Sequential(
            nn.Conv1d(128, 64, 3, stride=2, padding=1),
            nn.LayerNorm([64, 22]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.l4 = nn.Sequential(
            nn.Conv1d(64, 32, 2, stride=2, padding=0),
            nn.LayerNorm([32, 11]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Linear(32 * 11, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    z = torch.randn(64, 128)
    net = Generator()
    print(net(z).shape)

    z = torch.randn(64, 4, 170)
    net = Discriminator()
    print(net(z).shape)
