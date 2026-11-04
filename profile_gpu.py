"""Bounded synthetic inference/training check with measured CUDA peaks."""
import sys
sys.dont_write_bytecode = True
import argparse
import time
import torch
from mfcc.model import MFCC, load_weights
from mfcc.runtime import checkpoint_for, device_for, environment, new_run, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scale', type=int, choices=[2, 3, 4, 8], default=4)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr-patch', type=int, default=48)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--run-dir')
    a = p.parse_args()
    if min(a.batch_size, a.lr_patch, a.steps) < 1: raise ValueError('Sizes and steps must be positive')
    device = device_for(a.device)
    if a.amp and device.type != 'cuda': raise ValueError('--amp requires CUDA')
    torch.set_num_threads(4)
    model = MFCC(a.scale)
    load_weights(model, checkpoint_for(a.scale))
    model.to(device).train()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=a.amp)
    x = torch.rand(a.batch_size, 3, a.lr_patch, a.lr_patch, device=device)
    target = torch.rand(a.batch_size, 3, a.lr_patch*a.scale, a.lr_patch*a.scale, device=device)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=a.amp):
            loss = torch.nn.functional.l1_loss(model(x), target)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        return loss.item()
    step()  # warm-up includes optimizer-state allocation
    if device.type == 'cuda':
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(a.steps): loss = step()
    if device.type == 'cuda': torch.cuda.synchronize()
    report = {'environment': environment(device), 'arguments': vars(a),
              'seconds_per_optimizer_step': (time.perf_counter()-start)/a.steps,
              'last_synthetic_loss': loss, 'weights_saved': False}
    if device.type == 'cuda':
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        report['peak_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
        report['total_vram_mib'] = torch.cuda.get_device_properties(device).total_memory/2**20
    run = new_run('profile', a.run_dir)
    write_json(run/'profile.json', report)
    print(report); print(f'Report: {run / "profile.json"}')


if __name__ == '__main__': main()
