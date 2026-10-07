"""The anisotropic 3D U-Net (tournament winner ``a03-aniso`` of cmb-valdo-chain) and its checkpoint loader.

Level 0 convolves (1, 3, 3) and pools (1, 2, 2) in-plane only, because on the 0.5 x 0.5 x 1 mm grid the
through-plane direction is mostly interpolated (slices of 0.8-5 mm); from level 1 on the grid is 1 mm
isotropic and the network is an ordinary 3D U-Net. 32 base channels, four poolings, InstanceNorm,
LeakyReLU, transposed convolutions, one logit per voxel. 22.5 M parameters.

Checkpoints: ``load_unet(path)`` accepts the product files (``{"state_dict": ..., "meta": ...}``) and the raw
``modell.pt`` of the research code (German attribute names), which ``convert_legacy_keys`` renames.
"""
from __future__ import annotations

import re
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_MULTIPLE = (8, 16, 16)          # (z, y, x): four poolings, the first one in-plane only


class ConvBlock(nn.Sequential):
    """Two ``Conv3d -> InstanceNorm -> LeakyReLU`` layers; ``kernel`` may be anisotropic."""

    def __init__(self, in_channels: int, out_channels: int, kernel: Sequence[int] = (3, 3, 3)) -> None:
        padding = tuple(k // 2 for k in kernel)

        def layer(c_in: int):
            return [nn.Conv3d(c_in, out_channels, tuple(kernel), padding=padding, bias=False),
                    nn.InstanceNorm3d(out_channels, affine=True), nn.LeakyReLU(0.01, inplace=True)]

        super().__init__(*layer(in_channels), *layer(out_channels))


class AnisotropicUNet3D(nn.Module):
    """``(B, in_channels, D, H, W)`` -> logits ``(B, 1, D, H, W)``; D divisible by 8, H and W by 16."""

    IN_PLANE_KERNEL = (1, 3, 3)
    IN_PLANE_POOL = (1, 2, 2)

    def __init__(self, in_channels: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc0 = ConvBlock(in_channels, c, kernel=self.IN_PLANE_KERNEL)
        self.enc1 = ConvBlock(c, 2 * c)
        self.enc2 = ConvBlock(2 * c, 4 * c)
        self.enc3 = ConvBlock(4 * c, 8 * c)
        self.bottleneck = ConvBlock(8 * c, 16 * c)
        self.pool_in_plane = nn.MaxPool3d(self.IN_PLANE_POOL)
        self.pool = nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(16 * c, 8 * c, 2, stride=2)
        self.up2 = nn.ConvTranspose3d(8 * c, 4 * c, 2, stride=2)
        self.up1 = nn.ConvTranspose3d(4 * c, 2 * c, 2, stride=2)
        self.up0 = nn.ConvTranspose3d(2 * c, c, self.IN_PLANE_POOL, stride=self.IN_PLANE_POOL)
        self.dec3 = ConvBlock(16 * c, 8 * c)
        self.dec2 = ConvBlock(8 * c, 4 * c)
        self.dec1 = ConvBlock(4 * c, 2 * c)
        self.dec0 = ConvBlock(2 * c, c, kernel=self.IN_PLANE_KERNEL)
        self.head = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip0 = self.enc0(x)
        skip1 = self.enc1(self.pool_in_plane(skip0))
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        h = self.bottleneck(self.pool(skip3))
        for up, dec, skip in ((self.up3, self.dec3, skip3), (self.up2, self.dec2, skip2),
                              (self.up1, self.dec1, skip1), (self.up0, self.dec0, skip0)):
            h = dec(torch.cat([up(h), skip], dim=1))
        return self.head(h)


class PadToMultiple(nn.Module):
    """Pads the volume to multiples of ``PAD_MULTIPLE`` before the network and crops the output back."""

    def __init__(self, net: nn.Module, multiple: Sequence[int] = PAD_MULTIPLE) -> None:
        super().__init__()
        self.net, self.multiple = net, tuple(multiple)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x.shape[-3:]
        target = [-(-g[i] // self.multiple[i]) * self.multiple[i] for i in range(3)]
        pad = [target[i] - g[i] for i in range(3)]
        if any(pad):
            x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
        y = self.net(x)
        return y[..., :g[0], :g[1], :g[2]]


_LEGACY_RULES = [
    (r"^netz\.", "net."),
    (r"\.mitte\.net\.", ".bottleneck."),
    (r"\.e(\d)\.net\.", r".enc\1."),
    (r"\.d(\d)\.net\.", r".dec\1."),
    (r"\.u(\d)\.", r".up\1."),
    (r"\.aus\.", ".head."),
]


def convert_legacy_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename the keys of a research checkpoint (``netz.e0.net.0.weight`` ...) to this module's names."""
    out = {}
    for key, tensor in state.items():
        new = key
        for pattern, repl in _LEGACY_RULES:
            new = re.sub(pattern, repl, new)
        out[new] = tensor
    return out


def build_unet(in_channels: int = 2, base_channels: int = 32) -> PadToMultiple:
    return PadToMultiple(AnisotropicUNet3D(in_channels, base_channels))


def load_unet(path: str, device: str = "cpu") -> PadToMultiple:
    """Load a product weight file or a research ``modell.pt``; returns the padded model in eval mode."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    if any(k.startswith("netz.") for k in state):
        state = convert_legacy_keys(state)
    model = build_unet()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
