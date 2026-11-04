"""Check all Journal weights, aliases, gradients, metrics, optional legacy parity."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from mfcc.model import MFCC, load_weights
from mfcc.metrics import psnr, ssim
from mfcc.runtime import (ROOT, checkpoint_for, environment, new_run, sha256,
                          write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-source', help='Optional original srn20210829 directory')
    p.add_argument('--original-root', help='Optional parent project root for source SHA-256 checks')
    p.add_argument('--run-dir')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1)
    results = []
    expected = {2: 1861547, 3: 2230827, 4: 2156971, 8: 2452395}
    if args.legacy_source:
        sys.path.insert(0, str(Path(args.legacy_source).resolve()))
        from model.rc20r2rg1nup import make_model
        from model import Model as LegacyWrapper
    for scale, count in expected.items():
        m = MFCC(scale)
        load_weights(m, checkpoint_for(scale))
        assert sum(p.numel() for p in m.parameters()) == count
        assert m.body[0] is m.body2[0] is m.body3[0]
        assert m.body[1] is m.body2[1] is m.body3[1]
        x = torch.rand(1, 3, 48, 48)
        m.eval()
        with torch.no_grad(): y = m(x)
        assert y.shape == (1, 3, 48*scale, 48*scale) and torch.isfinite(y).all()
        row = {'scale': scale, 'params': count, 'strict_load': True, 'shared_weights': True,
               'forward_finite': True}
        if args.legacy_source:
            cfg = SimpleNamespace(n_resgroups=10, n_resblocks=20, n_feats=64,
                                  reduction=16, scale=[scale], rgb_range=1, n_colors=3, res_scale=1)
            old = make_model(cfg).eval()
            for layer in old.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    layer.stride = tuple(int(v) for v in layer.stride)
            torch.nn.Module.load_state_dict(old, m.state_dict(), strict=True)
            with torch.no_grad():
                old_y = old(x)
                torch.testing.assert_close(y, old_y, rtol=0, atol=0)
                wrapper = object.__new__(LegacyWrapper)
                torch.nn.Module.__init__(wrapper)
                wrapper.model, wrapper.scale, wrapper.idx_scale, wrapper.n_GPUs = old, [scale], 0, 1
                from mfcc.inference import chop
                old_chop = wrapper.forward_chop(x)
                torch.testing.assert_close(chop(m, x, scale), old_chop, rtol=0, atol=0)
            row['legacy_direct_max_abs_error'] = (y-old_y).abs().max().item()
            row['legacy_chop_exact'] = True
        m.train()
        opt = torch.optim.Adam((p for p in m.parameters() if p.requires_grad), lr=1e-4)
        old_weight = m.head[0].weight.detach().clone()
        opt.zero_grad(); pred = m(x)
        loss = (pred-torch.rand_like(pred)).abs().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step()
        assert not torch.equal(old_weight, m.head[0].weight)
        row['adam_update_passed'] = True
        results.append(row)
        print(row, flush=True)
    a = torch.rand(1, 3, 32, 32)
    assert math.isinf(psnr(a, a, 2))
    assert abs(ssim(a, a, 2)-1) < 1e-12
    b = a*0.9
    weights = a.new_tensor([65.738, 129.057, 25.064]).view(1,3,1,1)/256
    legacy_diff = ((a-b)*weights).sum(1)[..., 2:-2, 2:-2]
    legacy_psnr = -10*math.log10(legacy_diff.square().mean().item())
    assert psnr(a, b, 2) == legacy_psnr
    # Check that conflicting state-dict aliases cannot silently alter shared weights.
    m = MFCC(2)
    state = m.state_dict()
    key = 'body2.0.body.0.body.0.weight'
    state[key] = state[key].clone()+1
    run = new_run('verify', args.run_dir)
    temp = run/'conflicting-alias-test.pt'
    torch.save(state, temp)
    try:
        load_weights(m, temp)
    except ValueError as exc:
        assert 'Inconsistent shared weights' in str(exc)
    else:
        raise AssertionError('Conflicting aliases accepted')
    finally:
        temp.unlink()
    checked_sources = 0
    if args.original_root:
        manifest = json.loads((ROOT/'weights'/'manifest.json').read_text())
        for rel, digest in manifest['source_files'].items():
            assert sha256(Path(args.original_root)/rel) == digest, f'Original changed: {rel}'
            checked_sources += 1
    report = {'environment': environment(torch.device('cpu')), 'models': results,
              'metric_checks_passed': True, 'alias_rejection_passed': True,
              'original_file_hashes_verified': checked_sources}
    write_json(run/'verification.json', report)
    print(f'Report: {run / "verification.json"}')


if __name__ == '__main__': main()
