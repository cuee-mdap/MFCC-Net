"""Legacy Y-PSNR; explicit MATLAB-style Gaussian Y-SSIM implementation."""
import math
import torch
from torch.nn import functional as F


def quantize(sr):
    return sr.mul(255).clamp(0, 255).round().div(255)


def y_plane(rgb):
    weights = rgb.new_tensor([65.738, 129.057, 25.064]).view(1, 3, 1, 1) / 256
    return (rgb * weights).sum(dim=1, keepdim=True)


def psnr(sr, hr, scale):
    if sr.shape != hr.shape or min(sr.shape[-2:]) <= 2 * scale:
        raise ValueError('PSNR needs matching images larger than twice the scale')
    # Exactly the original utility.calc_psnr benchmark convention, RGB range 1.
    diff = y_plane(sr - hr)[..., scale:-scale, scale:-scale]
    mse = diff.square().mean().item()
    return float('inf') if mse == 0 else -10 * math.log10(mse)


def ssim(sr, hr, scale):
    # 11x11 Gaussian sigma=1.5, valid window, population covariance, Y in [16,235].
    x = y_plane(sr.double())[..., scale:-scale, scale:-scale] * 255 + 16
    y = y_plane(hr.double())[..., scale:-scale, scale:-scale] * 255 + 16
    if min(x.shape[-2:]) < 11:
        raise ValueError('SSIM needs at least 11 pixels after border shaving')
    t = torch.arange(-5, 6, device=x.device, dtype=x.dtype)
    g = torch.exp(-t.square() / (2 * 1.5**2)); g /= g.sum()
    kernel = (g[:, None] * g[None, :]).view(1, 1, 11, 11)
    filt = lambda a: F.conv2d(a, kernel)
    mx, my = filt(x), filt(y)
    vx, vy, cov = filt(x*x)-mx*mx, filt(y*y)-my*my, filt(x*y)-mx*my
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    return (((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2))).mean().item()
