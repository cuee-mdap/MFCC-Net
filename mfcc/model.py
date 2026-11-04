"""Adapted from srn20210829/model/rc20r2rg1nup.py and common.py.

Keep module names and weight sharing compatible with the original checkpoints.
Upstream MIT notices are preserved in LICENSE.
"""
import torch
from torch import nn


def conv(in_channels, out_channels, kernel_size=3):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     stride=1, padding=kernel_size // 2, bias=True)


class MeanShift(nn.Conv2d):
    def __init__(self, sign):
        super().__init__(3, 3, kernel_size=1)
        with torch.no_grad():
            self.weight.copy_(torch.eye(3).view(3, 3, 1, 1))
            self.bias.copy_(sign * torch.tensor([0.4488, 0.4371, 0.4040]))
        self.requires_grad_(False)


class CALayer(nn.Module):
    def __init__(self, channels=64, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1), nn.ReLU(True),
            nn.Conv2d(channels // reduction, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.conv_du(self.avg_pool(x))


class RCAB(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(conv(64, 64), nn.ReLU(True),
                                  conv(64, 64), CALayer())

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(*[RCAB() for _ in range(20)], conv(64, 64))

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, scale):
        layers = []
        factors = {2: [2], 3: [3], 4: [2, 2], 8: [2, 2, 2]}[scale]
        for factor in factors:
            layers.extend([conv(64, 64 * factor * factor), nn.PixelShuffle(factor)])
        super().__init__(*layers)


class MFCC(nn.Module):
    """Journal architecture: RGB [0,1], 64 channels, 20 RCAB, reduction 16."""
    def __init__(self, scale):
        super().__init__()
        if scale not in (2, 3, 4, 8):
            raise ValueError('scale must be 2, 3, 4, or 8')
        self.scale = scale
        self.sub_mean = MeanShift(-1)
        self.head = nn.Sequential(conv(3, 64))
        # These are the SAME objects in all three paths, including the final conv.
        shared = [ResidualGroup(), conv(64, 64)]
        self.body = nn.Sequential(*shared)
        self.body2 = nn.Sequential(*shared)
        self.body3 = nn.Sequential(*shared)
        self.tail = nn.Sequential(Upsampler(scale))
        self.add_mean = MeanShift(1)
        self.conv1 = Upsampler(scale)
        self.conf = conv(64, 3)

    def forward(self, x):
        x = self.head(self.sub_mean(x))
        y = self.conv1(x)
        f11 = self.body(x)
        f12 = self.body2(x)
        f11 += x
        f12 += x
        f21 = self.body3(f11)
        x = self.tail(f21 + f12)
        x += y
        return self.add_mean(self.conf(x))


def load_weights(model, path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('Expected a state dict or a training checkpoint')
    state = payload.get('model', payload)
    # Refuse ambiguous alias values rather than silently overwriting tied weights.
    for key, value in state.items():
        if key.startswith('body.'):
            suffix = key[len('body.'):]
            for prefix in ('body2.', 'body3.'):
                other = prefix + suffix
                if other in state and not torch.equal(value, state[other]):
                    raise ValueError(f'Inconsistent shared weights: {key} / {other}')
    model.load_state_dict(state, strict=True)
    return payload
