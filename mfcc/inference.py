"""Single-device inference; legacy chop preserved for Journal comparisons."""
import torch
from .metrics import quantize, psnr, ssim


def chop(model, x, scale, min_size=160000, shave=10):
    if min_size < 1 or shave < 0:
        raise ValueError('Invalid chop size/overlap')
    h, w = x.shape[-2:]
    hh, wh = h // 2, w // 2
    hs, ws = hh + shave, wh + shave
    hs += scale - hs % scale
    ws += scale - ws % scale
    # Tiny inputs cannot safely use the original split/merge rule.
    if hs >= h or ws >= w:
        return model(x)
    patches = [x[..., :hs, :ws], x[..., :hs, w-ws:],
               x[..., h-hs:, :ws], x[..., h-hs:, w-ws:]]
    fn = (lambda p: model(p)) if hs * ws < min_size else (
        lambda p: chop(model, p, scale, min_size, shave))
    results = [fn(p) for p in patches]
    h, w, hh, wh, hs, ws = [v*scale for v in (h, w, hh, wh, hs, ws)]
    out = x.new_empty(x.shape[0], 3, h, w)
    out[..., :hh, :wh] = results[0][..., :hh, :wh]
    out[..., :hh, wh:] = results[1][..., :hh, ws-w+wh:]
    out[..., hh:, :wh] = results[2][..., hs-h+hh:, :wh]
    out[..., hh:, wh:] = results[3][..., hs-h+hh:, ws-w+wh:]
    return out


def predict(model, lr, mode='journal', min_size=160000):
    if mode == 'journal':
        mode = 'direct' if model.scale == 8 else 'chop'
    if mode == 'direct':
        return model(lr)
    if mode == 'chop':
        return chop(model, lr, model.scale, min_size=min_size)
    raise ValueError(f'Unknown inference mode: {mode}')


@torch.inference_mode()
def evaluate(model, pairs, device, mode='journal', min_size=160000, with_ssim=False):
    was_training = model.training
    model.eval()
    rows = []
    for lr, hr, name in pairs:
        sr = quantize(predict(model, lr.unsqueeze(0).to(device), mode, min_size)).cpu()
        hr = hr.unsqueeze(0)
        row = {'image': name, 'y_psnr': psnr(sr, hr, model.scale)}
        if with_ssim: row['y_ssim'] = ssim(sr, hr, model.scale)
        rows.append(row)
    model.train(was_training)
    return rows
