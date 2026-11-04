"""Parameters and multiply-accumulates (MACs) of the Journal model.

Parameters are unique tensors (body, body2 and body3 share one residual group, so state_dict() lists more values).
MACs are counted from every nn.Conv2d with forward hooks during a real forward pass, for an HR output of
--hr-height x --hr-width (default 1280x720, the usual convention for Multi-Adds).
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json

import torch
from torch import nn
from mfcc.model import MFCC, load_weights
from mfcc.runtime import checkpoint_for

# Table 2 of the paper
PAPER = {2: (1861, 364.5), 3: (2230, 374.1), 4: (2157, 395.2), 8: (2453, 518.2)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hr-height', type=int, default=720)
    p.add_argument('--hr-width', type=int, default=1280)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    for scale in (2, 3, 4, 8):
        model = MFCC(scale)
        load_weights(model, checkpoint_for(scale))
        model.eval().to(args.device)
        params = sum(t.numel() for t in {id(t): t for t in model.parameters()}.values())
        macs = [0]

        def hook(m, inp, out):
            macs[0] += out.numel() * m.in_channels * m.kernel_size[0] * m.kernel_size[1] // m.groups
        handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
        h, w = args.hr_height // scale, args.hr_width // scale
        with torch.inference_mode():
            model(torch.zeros(1, 3, h, w, device=args.device))
        for handle in handles:
            handle.remove()
        print(json.dumps({'scale': scale, 'lr': [h, w], 'params': params, 'params_paper_K': PAPER[scale][0],
                          'macs_G': round(macs[0] / 1e9, 1), 'multi_adds_paper_G': PAPER[scale][1]}), flush=True)


if __name__ == '__main__':
    main()
