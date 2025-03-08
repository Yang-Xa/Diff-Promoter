import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class ResNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(out_dim),

            nn.Conv1d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(out_dim),
        )

        self.shortcut = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            ResNet(4, 32),
            ResNet(32, 32),
        )

        self.lation_Layer = nn.TransformerEncoderLayer(d_model=32, nhead=8, dim_feedforward=64, batch_first=True)
        self.lation_Encoder = nn.TransformerEncoder(self.lation_Layer, num_layers=4) 

        self.conv2 = nn.Sequential(
            ResNet(32, 64),
            ResNet(64, 64),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),

            ResNet(64, 128),
            ResNet(128, 128),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),

            ResNet(128, 64),
            ResNet(64, 64),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 23, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=256, out_features=64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=64, out_features=1),
        )

        # self.common = nn.Sequential(
        #     self.conv1,
        #     Rearrange('b d n -> b n d'),
        #     self.lation_Encoder,
        #     Rearrange('b n d -> b d n'),
        #     self.conv2,
        #     Rearrange('b d n -> b (d n)'),
        #     self.fc,
        # )

    def forward(self, x):
        # res = self.common(x)
        x = x[:,:,3:-3]
        x = self.conv1(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.lation_Encoder(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        res = self.fc(x)
        return res[:, 0]

if __name__ == '__main__':
    
    x = torch.rand(64, 4, 170).cuda(4)

    net = Net().cuda(4)
    b = net(x)
    print(b.shape)

