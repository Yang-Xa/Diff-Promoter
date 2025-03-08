import torch
import torch.nn as nn
from model import *


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.locked = UNetModel()
        
        in_out_dims = [(16, 32), (32, 64), (64, 128), (128, 128)]
        time_dim = 2 * 128

        self.zero_conv1 = zero_module(nn.Conv1d(4, 16, 1))

        self.trainable_downs = nn.ModuleList([])
        for ind, (in_dim, out_dim) in enumerate(in_out_dims):
            is_last = ind >= (len(in_out_dims) - 1)
            self.trainable_downs.append(
                nn.ModuleList([
                    ResidualBlock(in_dim, in_dim, time_dim),
                    ResidualBlock(in_dim, in_dim, time_dim),
                    Residual(PreNorm(in_dim, LinearAttention(in_dim))),
                    Downsample(in_dim, out_dim) if not is_last else nn.Conv1d(in_dim, out_dim, 3, padding=1)
                ])
            )

        mid_dim = in_out_dims[-1][-1]
        self.trainable_mid_block1 = ResidualBlock(mid_dim, mid_dim, time_dim)
        self.trainable_mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.trainable_mid_block2 = ResidualBlock(mid_dim, mid_dim, time_dim)

        self.trainable_mid_zero = zero_module(nn.Conv1d(mid_dim, mid_dim, 1))

        self.trainable_ups = nn.ModuleList([])
        for ind, (in_dim, out_dim) in enumerate(reversed(in_out_dims)):
            is_last = ind == (len(in_out_dims) - 1)
            self.trainable_ups.append(
                nn.ModuleList([
                    zero_module(nn.Conv1d(in_dim + out_dim, out_dim, 1)),
                    zero_module(nn.Conv1d(in_dim + out_dim, out_dim, 1)),
                    zero_module(Upsample(out_dim, in_dim)) if not is_last else zero_module(nn.Conv1d(out_dim, in_dim, 1))
                ])
            )

    def forward(self, x, t, c):
        x = self.locked.init_conv(x)
        r = x.clone()

        t = self.locked.time_mlp(t)

        # c = self.process_cond(c) 
        # c = self.locked.init_conv(c)
        c = self.zero_conv1(c)
        sample = x + c

        h = []
        for block1, block2, attn, downsample in self.trainable_downs:
            sample = block1(sample, t)
            h.append(sample)

            sample = block2(sample, t)
            sample = attn(sample)
            h.append(sample)

            sample = downsample(sample)
        
        sample = self.trainable_mid_block1(sample, t)
        sample = self.trainable_mid_attn(sample)
        sample = self.trainable_mid_block2(sample, t)
        sample = self.trainable_mid_zero(sample)
        sample_mid = sample

        trainable_h = []
        for block1, block2, upsample in self.trainable_ups:
            sample = torch.cat((sample, h.pop()), dim=1)
            sample = block1(sample)
            trainable_h.insert(0, sample)

            sample = torch.cat((sample, h.pop()), dim=1)
            sample = block2(sample)
            trainable_h.insert(0, sample)

            sample = upsample(sample)
        
        
        locked_h = []
        for block1, block2, attn, downsample in self.locked.downs:
            x = block1(x, t)
            locked_h.append(x)

            x = block2(x, t)
            x = attn(x)
            locked_h.append(x)
                
            x = downsample(x)
            
        x = x + sample_mid
        x = self.locked.mid_block1(x, t)
        x = self.locked.mid_attn(x)
        x = self.locked.mid_block2(x, t)

        for block1, block2, attn, upsample in self.locked.ups:
            x = x + trainable_h.pop()
            x = torch.cat((x, locked_h.pop()), dim=1)
            x = block1(x, t)

            x = x + trainable_h.pop()
            x = torch.cat((x, locked_h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)
        
        x = torch.cat((x, r), dim=1)
        x = self.locked.final_res_conv(x, t)
        return self.locked.final_conv(x)


if __name__ == '__main__':
    device = torch.device('cuda:3')
    model = ControlNet().to(device)
    model.locked.load_state_dict(torch.load('model.pkl', map_location=device).state_dict())

    # x = torch.rand(64, 4, 176)
    # c = torch.rand(64, 4, 176)
    # t = torch.randint(0, 100, (64,))

    # res = model(x.to(device), t.to(device), c.to(device))
    # # print(res)
    # print(res.shape)

    # for name, param in model.named_parameters():
    #     print(name, param)

    # for name in model.state_dict():
    #     print(name, model.state_dict()[name].shape)
    print(model.state_dict()['trainanle_downs'])


# newModel = zero_module(
#     nn.Conv1d(4, 32, kernel_size=8)
# )


# minput = torch.rand(128, 4, 170)
# print(minput)
# mout = newModel(minput)
# print(mout)