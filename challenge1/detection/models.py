"""Segmentation models: one CNN reference + three transformers spanning the
locality-bias axis (most local -> least local):

* ``unet``      — plain CNN U-Net. Not a transformer: the reference every
                  transformer has to beat to justify itself.
* ``transunet`` — hybrid: CNN encoder to stride 8, global self-attention on the
                  19x19 token grid (long-range context: the lattice of interdots),
                  CNN decoder with full-resolution skips.
* ``segformer`` — hierarchical transformer (Mix Transformer): overlapping conv patch
                  embeddings, spatial-reduction attention, Mix-FFN, **no positional
                  embedding** (resolution-agnostic), MLP decoder + full-res head.
* ``vit``       — plain ViT (patch 4, global attention, no hierarchy), SETR-style
                  progressive upsampling + a light full-res head.

Sticks are 1-2 px wide and 4-8 px long, so every model ends with a **full-resolution**
head fed with the raw image: a stride-4 prediction alone cannot represent them.
Inputs are padded (replicate) to the model's stride and the output is cropped back.
All attention uses ``F.scaled_dot_product_attention`` (flash / mem-efficient kernels).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- utils
def conv_bn_act(cin: int, cout: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.GELU()
    )


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(conv_bn_act(cin, cout), conv_bn_act(cout, cout))


def sincos_2d(h: int, w: int, dim: int, device) -> torch.Tensor:
    """Fixed 2-D sin-cos position embedding (h*w, dim): any grid size works."""
    assert dim % 4 == 0
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = 1.0 / (10000 ** (torch.arange(dim // 4, device=device, dtype=torch.float32) / (dim // 4)))
    out = []
    for pos in (y.flatten(), x.flatten()):
        a = pos[:, None] * omega[None]
        out += [a.sin(), a.cos()]
    return torch.cat(out, dim=1)


class DropPath(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = torch.rand(x.shape[0], *([1] * (x.ndim - 1)), device=x.device) >= self.p
        return x * keep / (1 - self.p)


class Attention(nn.Module):
    """Multi-head self-attention; optional spatial reduction of K/V (SegFormer)."""

    def __init__(self, dim: int, heads: int, sr: int = 1) -> None:
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.sr = sr
        if sr > 1:
            self.sr_conv = nn.Conv2d(dim, dim, sr, sr)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x, h: int, w: int):
        b, n, c = x.shape
        q = self.q(x).view(b, n, self.heads, c // self.heads).transpose(1, 2)
        if self.sr > 1:
            xs = x.transpose(1, 2).reshape(b, c, h, w)
            xs = self.sr_conv(xs).flatten(2).transpose(1, 2)
            xs = self.sr_norm(xs)
        else:
            xs = x
        kv = self.kv(xs).view(b, -1, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, kv[0], kv[1])
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class Block(nn.Module):
    """Pre-norm transformer block. ``mix_ffn`` adds a depthwise 3x3 conv (SegFormer)."""

    def __init__(self, dim, heads, mlp_ratio=4.0, sr=1, drop_path=0.0, mix_ffn=False):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.n1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, sr)
        self.n2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hid)
        self.dw = nn.Conv2d(hid, hid, 3, 1, 1, groups=hid) if mix_ffn else None
        self.fc2 = nn.Linear(hid, dim)
        self.dp = DropPath(drop_path)

    def forward(self, x, h, w):
        x = x + self.dp(self.attn(self.n1(x), h, w))
        z = self.fc1(self.n2(x))
        if self.dw is not None:
            b, n, c = z.shape
            z = self.dw(z.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)
        z = self.fc2(F.gelu(z))
        return x + self.dp(z)


class FullResHead(nn.Module):
    """Fuses upsampled decoder features with full-res features of the raw image."""

    def __init__(self, cin: int, cmid: int = 32) -> None:
        super().__init__()
        self.pix = nn.Sequential(conv_bn_act(1, cmid), conv_bn_act(cmid, cmid))
        self.fuse = nn.Sequential(conv_bn_act(cin + cmid, cmid), conv_bn_act(cmid, cmid))
        self.out = nn.Conv2d(cmid, 1, 1)

    def forward(self, feat, img):
        feat = F.interpolate(feat, size=img.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.fuse(torch.cat([feat, self.pix(img)], 1)))


class Padded(nn.Module):
    """Pads (replicate) to a multiple of ``stride``, crops logits back."""

    def __init__(self, net: nn.Module, stride: int) -> None:
        super().__init__()
        self.net, self.stride = net, stride

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.stride, (-w) % self.stride
        if ph or pw:
            x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode="replicate")
        y = self.net(x)
        return y[..., ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]


# --------------------------------------------------------------------------- U-Net
class UNet(nn.Module):
    def __init__(self, base: int = 32, depth: int = 3) -> None:
        super().__init__()
        ch = [base * 2**i for i in range(depth + 1)]
        self.inc = DoubleConv(1, ch[0])
        self.downs = nn.ModuleList(DoubleConv(ch[i], ch[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in reversed(range(depth))
        )
        self.dec = nn.ModuleList(DoubleConv(2 * ch[i], ch[i]) for i in reversed(range(depth)))
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        y = skips.pop()
        for up, dec in zip(self.ups, self.dec):
            y = dec(torch.cat([up(y), skips.pop()], 1))
        return self.out(y)


# --------------------------------------------------------------------------- TransUNet
class TransUNet(nn.Module):
    def __init__(self, base=32, dim=256, depth=6, heads=8, drop_path=0.1) -> None:
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.e1 = DoubleConv(1, c1)  # /1
        self.e2 = nn.Sequential(conv_bn_act(c1, c2, s=2), conv_bn_act(c2, c2))  # /2
        self.e3 = nn.Sequential(conv_bn_act(c2, c3, s=2), conv_bn_act(c3, c3))  # /4
        self.embed = nn.Conv2d(c3, dim, 2, 2)  # /8 tokens
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(Block(dim, heads, drop_path=p) for p in dpr)
        self.norm = nn.LayerNorm(dim)
        self.up3 = nn.ConvTranspose2d(dim, c3, 2, 2)
        self.d3 = DoubleConv(2 * c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.d2 = DoubleConv(2 * c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.d1 = DoubleConv(2 * c1, c1)
        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        t = self.embed(s3)
        b, c, h, w = t.shape
        z = t.flatten(2).transpose(1, 2) + sincos_2d(h, w, c, x.device)
        for blk in self.blocks:
            z = blk(z, h, w)
        t = self.norm(z).transpose(1, 2).reshape(b, c, h, w)
        y = self.d3(torch.cat([self.up3(t), s3], 1))
        y = self.d2(torch.cat([self.up2(y), s2], 1))
        y = self.d1(torch.cat([self.up1(y), s1], 1))
        return self.out(y)


# --------------------------------------------------------------------------- SegFormer
class OverlapPatchEmbed(nn.Module):
    def __init__(self, cin, cout, k, s):
        super().__init__()
        self.proj = nn.Conv2d(cin, cout, k, s, k // 2)
        self.norm = nn.LayerNorm(cout)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), h, w


class SegFormer(nn.Module):
    def __init__(
        self,
        dims=(32, 64, 160, 256),
        depths=(2, 2, 2, 2),
        heads=(1, 2, 5, 8),
        srs=(8, 4, 2, 1),
        dec_dim=128,
        drop_path=0.1,
    ) -> None:
        super().__init__()
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.embeds, self.stages, self.norms = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        cin, k = 1, 0
        for i, (d, n, hd, sr) in enumerate(zip(dims, depths, heads, srs)):
            self.embeds.append(OverlapPatchEmbed(cin, d, 7 if i == 0 else 3, 4 if i == 0 else 2))
            self.stages.append(
                nn.ModuleList(Block(d, hd, 4.0, sr, dpr[k + j], mix_ffn=True) for j in range(n))
            )
            self.norms.append(nn.LayerNorm(d))
            cin, k = d, k + n
        self.lin = nn.ModuleList(nn.Conv2d(d, dec_dim, 1) for d in dims)
        self.fuse = conv_bn_act(4 * dec_dim, dec_dim, k=1)
        self.head = FullResHead(dec_dim)

    def forward(self, x):
        img, feats, z = x, [], x
        for emb, blocks, norm in zip(self.embeds, self.stages, self.norms):
            t, h, w = emb(z)
            for blk in blocks:
                t = blk(t, h, w)
            z = norm(t).transpose(1, 2).reshape(t.shape[0], -1, h, w)
            feats.append(z)
        size = feats[0].shape[-2:]
        f = [
            F.interpolate(l(f), size=size, mode="bilinear", align_corners=False)
            for l, f in zip(self.lin, feats)
        ]
        return self.head(self.fuse(torch.cat(f, 1)), img)


# --------------------------------------------------------------------------- ViT (SETR-PUP)
class ViTSeg(nn.Module):
    def __init__(self, patch=4, dim=192, depth=8, heads=3, drop_path=0.1) -> None:
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv2d(1, dim, patch, patch)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(Block(dim, heads, drop_path=p) for p in dpr)
        self.norm = nn.LayerNorm(dim)
        ups, c = [], dim
        for _ in range(int(math.log2(patch))):
            ups += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), conv_bn_act(c, 64)]
            c = 64
        self.pup = nn.Sequential(*ups)
        self.head = FullResHead(64)

    def forward(self, x):
        t = self.embed(x)
        b, c, h, w = t.shape
        z = t.flatten(2).transpose(1, 2) + sincos_2d(h, w, c, x.device)
        for blk in self.blocks:
            z = blk(z, h, w)
        t = self.norm(z).transpose(1, 2).reshape(b, c, h, w)
        return self.head(self.pup(t), x)


# --------------------------------------------------------------------------- factory
ARCHS = {
    "unet": (lambda: UNet(base=32, depth=3), 8),
    "transunet": (lambda: TransUNet(base=32, dim=256, depth=6, heads=8), 8),
    "segformer": (lambda: SegFormer(), 32),
    "vit": (lambda: ViTSeg(patch=4, dim=192, depth=8, heads=3), 4),
}


class PhysInput(nn.Module):
    """Physical input conditioning, part of the model (so it is applied at inference too).

    On top of the global median/MAD normalisation done by the caller:
    * row-median subtraction: removes per-line offsets (stripes, drift, charge jumps), the
      same prior as the classical detectors (sticks are sparse, so the median is background);
    * re-scaling by the MAD, so the input stays in units of noise sigma;
    * de-spiking: a pixel above 6 sigma whose 8 neighbours all stay below min(30 % of it,
      3.5 sigma) is an isolated glitch, not a stick (a stick is >= 4 px long, its neighbours
      are bright too): it is replaced by its 3x3 median. Weak sticks (< 6 sigma) are never
      touched. Measured: removes 90 % of +-10 spikes, alters 0.4 % of stick pixels (those
      are single bright pixels, indistinguishable from a glitch).
    No learned parameter; the sign is left to the network (polarity is augmented).
    """

    def __init__(self, spike_z: float = 6.0, spike_ratio: float = 0.3, spike_cap: float = 3.5) -> None:
        super().__init__()
        self.spike_z, self.spike_ratio, self.spike_cap = spike_z, spike_ratio, spike_cap

    def forward(self, x):
        b, _, h, w = x.shape
        x = x - x.median(dim=-1, keepdim=True).values
        mad = x.flatten(1).abs().median(dim=1).values.view(b, 1, 1, 1)
        x = x / (1.4826 * mad + 1e-6)
        p = F.unfold(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3).view(b, 9, h, w)
        neigh = torch.cat([p[:, :4], p[:, 5:]], 1).abs().amax(1, keepdim=True)
        med = p.median(dim=1, keepdim=True).values
        spike = (x.abs() > self.spike_z) & (neigh < (self.spike_ratio * x.abs()).clamp_max(self.spike_cap))
        return torch.where(spike, med, x)


def build_model(name: str) -> nn.Module:
    if name == "unet_robust":  # U-Net behind the physical input conditioning
        return nn.Sequential(PhysInput(), Padded(UNet(), ARCHS["unet"][1]))
    if name == "unet16_robust":  # same, 4x fewer parameters (base 16: ~0.5 M), capacity axis
        return nn.Sequential(PhysInput(), Padded(UNet(base=16), ARCHS["unet"][1]))
    make, stride = ARCHS[name]
    return Padded(make(), stride)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
