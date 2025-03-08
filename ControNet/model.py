import torch
import math
import torch.nn as nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# Time Embedding
# (batch_size, 1) -> (batch_size, dim)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.SiLU()
        )

        self.time_emb = nn.Sequential(
            nn.Linear(time_dim, 2 *out_dim),
            nn.SiLU(),
            nn.Linear(2 * out_dim, 2 * out_dim)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, 3, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.SiLU()
        )

        self.shortcut = nn.Conv1d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x, t):
        t_emb = self.time_emb(t)[:, :, None]
        scale, shift = torch.chunk(t_emb, 2, dim=1)

        h = self.conv1[0](x)
        h = self.conv1[1](h)
        h = h * (scale + 1) + shift
        h = self.conv1[2](h)

        h = self.conv2(h)
        return h + self.shortcut(x)


def Upsample(in_dim, out_dim):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(in_dim, out_dim, 3, padding=1)
    )


def Downsample(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv1d(in_dim, out_dim, kernel_size=2, stride=2)
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)


class UNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        time_dim = 2 * 128
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.init_conv = nn.Conv1d(4, 16, kernel_size=1, padding=0)

        in_out_dims = [(16, 32), (32, 64), (64, 128), (128, 128)]

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (in_dim, out_dim) in enumerate(in_out_dims):
            is_last = ind >=  (len(in_out_dims) - 1)
            self.downs.append(
                nn.ModuleList([
                    ResidualBlock(in_dim, in_dim, time_dim),
                    ResidualBlock(in_dim, in_dim, time_dim),
                    Residual(PreNorm(in_dim, LinearAttention(in_dim))),
                    Downsample(in_dim, out_dim) if not is_last else nn.Conv1d(in_dim, out_dim, 3, padding=1)
                ])
            )
        
        mid_dim = in_out_dims[-1][-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_dim)

        for ind, (in_dim, out_dim) in enumerate(reversed(in_out_dims)):
            is_last = ind == (len(in_out_dims) - 1)
            self.ups.append(
                nn.ModuleList([
                    ResidualBlock(in_dim + out_dim, out_dim, time_dim),
                    ResidualBlock(in_dim + out_dim, out_dim, time_dim),
                    Residual(PreNorm(out_dim, LinearAttention(out_dim))),
                    Upsample(out_dim, in_dim) if not is_last else nn.Conv1d(out_dim, in_dim, 3, padding=1)
                ])
            )
        
        self.final_res_conv = ResidualBlock(16 + 16, 16, 2 * 128)
        self.final_conv = nn.Conv1d(16, 4, 1)

    def forward(self, x, t):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(t)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)
        
        x = torch.cat((x, r), dim=1)
        x = self.final_res_conv(x, t)
        return self.final_conv(x)



if __name__ == '__main__':
    device = torch.device('cuda:6')
    unet = UNetModel()
    unet.to(device)
    x = torch.rand(128, 4, 176)
    t = torch.randint(0, 100, (128,))

    res = unet(x.to(device), t.to(device))
    print(res.shape)

