import torch
from torch import nn
from torch.nn.modules.module import T


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(128, 64*11))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(64),

            nn.ConvTranspose1d(64, 128, 2, stride=2, padding=0),  # (, 128, 22)
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.ReLU(),

            nn.ConvTranspose1d(128, 256, 3, stride=2, padding=1),  # (, 256, 43)
            nn.Conv1d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm1d(256, 0.8),
            nn.ReLU(),

            nn.ConvTranspose1d(256, 128, 3, stride=2, padding=1),  # (, 128, 85)
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.ReLU(),

            nn.ConvTranspose1d(128, 64, 2, stride=2, padding=0),  # (, 64, 170)
            nn.Conv1d(64, 4, 3, stride=1, padding=1),
            nn.Tanh()
        )


    def forward(self, z):
        x = self.fc1(z)
        x = x.view(-1, 64, 11)

        x = self.conv_blocks(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(4, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.utils.parametrizations.spectral_norm(nn.Conv1d(64, 128, 4, stride=4, padding=2)),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2),

            nn.utils.parametrizations.spectral_norm(nn.Conv1d(128, 64, 3, stride=4, padding=0)),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2),

            nn.utils.parametrizations.spectral_norm(nn.Conv1d(64, 4, 1, stride=1, padding=0)),
        )

        self.fc = nn.Linear(4*11, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(-1, 4*11)
        x = self.fc(x)
        x = self.sigmod(x)
        return x


if __name__ == '__main__':
    model = Generator()
    z = torch.randn(4, 128)
    print(model(z).shape)

    model2 = Discriminator()
    x = torch.randn(4, 4, 170)
    print(model2(x).shape)