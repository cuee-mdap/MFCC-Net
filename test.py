"""Evaluate paired images, or upscale an LR folder with the Journal weights."""
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import statistics
import time

import torch
from mfcc.data import Pairs, image_files, read_image, tensor, save_image
from mfcc.inference import predict
from mfcc.metrics import quantize, psnr, ssim
from mfcc.model import MFCC, load_weights
from mfcc.runtime import checkpoint_for, device_for, environment, new_run, sha256, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scale', type=int, choices=[2, 3, 4, 8], required=True)
    p.add_argument('--lr-dir', required=True)
    p.add_argument('--hr-dir', help='Omit for inference only; metrics require true paired HR')
    p.add_argument('--checkpoint', help='Defaults to SHA-256 verified Journal weight')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--mode', choices=['journal', 'direct', 'chop'], default='journal')
    p.add_argument('--chop-min-size', type=int, default=160000)
    p.add_argument('--ssim', action='store_true', help='Also compute documented Gaussian Y-SSIM')
    p.add_argument('--limit', type=int)
    p.add_argument('--save-images', action='store_true')
    p.add_argument('--run-dir', help='New output directory; default is ./runs/<timestamp>')
    return p


def main():
    args = parser().parse_args()
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    checkpoint = checkpoint_for(args.scale, args.checkpoint)
    model = MFCC(args.scale)
    load_weights(model, checkpoint)
    model.to(device).eval()
    if args.hr_dir:
        samples = Pairs(args.lr_dir, args.hr_dir, args.scale, limit=args.limit)
    else:
        paths = image_files(args.lr_dir)
        if args.limit is not None:
            if args.limit < 1: raise ValueError('limit must be positive')
            paths = paths[:args.limit]
        samples = ((tensor(read_image(p)), None, p.stem) for p in paths)
    run = new_run('test', args.run_dir)
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    rows = []
    with torch.inference_mode():
        for lr, hr, name in samples:
            x = lr.unsqueeze(0).to(device)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            start = time.perf_counter()
            sr = quantize(predict(model, x, args.mode, args.chop_min_size)).cpu()
            elapsed = time.perf_counter() - start
            if not torch.isfinite(sr).all(): raise RuntimeError(f'Non-finite output: {name}')
            row = {'image': name, 'lr_shape': list(lr.shape), 'sr_shape': list(sr.shape[1:]),
                   'inference_seconds': elapsed}
            if hr is not None:
                row['y_psnr'] = psnr(sr, hr.unsqueeze(0), args.scale)
                if args.ssim: row['y_ssim'] = ssim(sr, hr.unsqueeze(0), args.scale)
            if args.save_images: save_image(sr, run/f'{name}_x{args.scale}_SR.png')
            rows.append(row)
            print(row, flush=True)
    report = {'environment': environment(device), 'arguments': vars(args),
              'checkpoint_sha256': sha256(checkpoint), 'count': len(rows), 'images': rows,
              'parameters': sum(p.numel() for p in model.parameters())}
    if args.hr_dir:
        report['mean_y_psnr'] = statistics.mean(r['y_psnr'] for r in rows)
        if args.ssim: report['mean_y_ssim'] = statistics.mean(r['y_ssim'] for r in rows)
    if device.type == 'cuda':
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device)/2**20
        report['peak_reserved_mib'] = torch.cuda.max_memory_reserved(device)/2**20
    write_json(run/'metrics.json', report)
    print(f'Report: {run / "metrics.json"}', flush=True)
    if args.hr_dir: print(f'Mean Y-PSNR: {report["mean_y_psnr"]:.6f} dB', flush=True)


if __name__ == '__main__':
    main()
