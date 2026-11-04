"""Train/fine-tune MFCC with paired images, accumulation, AMP and epoch resume."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path
import random
import statistics
import time

import torch
from torch.utils.data import DataLoader
from mfcc.data import Pairs, EpochSamples, seed_worker
from mfcc.inference import evaluate
from mfcc.model import MFCC, load_weights
from mfcc.runtime import (atomic_save, device_for, environment, new_run,
                          seed_all, write_json)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scale', type=int, choices=[2, 3, 4, 8], required=True)
    p.add_argument('--lr-dir', required=True)
    p.add_argument('--hr-dir', required=True)
    p.add_argument('--val-lr-dir')
    p.add_argument('--val-hr-dir')
    p.add_argument('--val-limit', type=int)
    p.add_argument('--val-mode', choices=['journal', 'direct', 'chop'], default='journal')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--init', help='Weights for fine-tuning, with fresh optimizer')
    group.add_argument('--resume', help='Modern training checkpoint, including optimizer/RNG')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=4, help='Physical microbatch')
    p.add_argument('--accumulate', type=int, default=4, help='Effective batch = batch-size * accumulate')
    p.add_argument('--lr-patch', type=int, default=48, help='LR dimension; HR patch = this * scale')
    p.add_argument('--epochs', type=int, default=600)
    p.add_argument('--steps-per-epoch', type=int, default=1000, help='Optimizer updates, after accumulation')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--decay-every', type=int, default=200, help='Halve LR every N completed epochs')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--amp', action='store_true', help='CUDA mixed precision; opt-in, not Journal FP32 baseline')
    p.add_argument('--print-every', type=int, default=100)
    p.add_argument('--run-dir', help='New output directory; defaults to ./runs/<timestamp>')
    return p


def main():
    args = parser().parse_args()
    for field in ('batch_size', 'accumulate', 'lr_patch', 'epochs', 'steps_per_epoch',
                  'decay_every', 'print_every', 'threads'):
        if getattr(args, field) < 1: raise ValueError(f'{field} must be positive')
    if args.workers < 0 or args.lr <= 0: raise ValueError('Invalid workers or learning rate')
    if bool(args.val_lr_dir) != bool(args.val_hr_dir):
        raise ValueError('Provide both validation LR and HR directories')
    device = device_for(args.device)
    if args.amp and device.type != 'cuda': raise ValueError('--amp requires CUDA')
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    train_data = Pairs(args.lr_dir, args.hr_dir, args.scale, args.lr_patch, augment=True)
    val = Pairs(args.val_lr_dir, args.val_hr_dir, args.scale, limit=args.val_limit) if args.val_lr_dir else None
    model = MFCC(args.scale)
    payload = None
    if args.init: load_weights(model, args.init)
    if args.resume:
        payload = load_weights(model, args.resume)
        if payload.get('format') != 'mfcc-training-v1':
            raise ValueError('Use --init for legacy model-only weights; --resume requires modern training state')
        for key in ('scale', 'batch_size', 'accumulate', 'lr_patch', 'steps_per_epoch',
                    'decay_every', 'lr', 'seed', 'amp', 'workers'):
            if payload['arguments'][key] != getattr(args, key):
                raise ValueError(f'Resume configuration differs: {key}; use --init to start a new experiment')
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_every, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    generator = torch.Generator().manual_seed(args.seed)
    first_epoch, best = 0, float('-inf')
    if payload:
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        scaler.load_state_dict(payload['scaler'])
        first_epoch, best = payload['completed_epochs'], payload['best_y_psnr']
        random.setstate(payload['python_rng'])
        torch.set_rng_state(payload['torch_rng'])
        generator.set_state(payload['loader_rng'])
        if device.type == 'cuda' and payload['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
    if args.epochs <= first_epoch:
        raise ValueError(f'Checkpoint already completed {first_epoch} epochs; increase --epochs')
    sample_count = args.steps_per_epoch * args.batch_size * args.accumulate
    loader = DataLoader(EpochSamples(train_data, sample_count), batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.workers, worker_init_fn=seed_worker,
                        generator=generator, pin_memory=device.type == 'cuda', drop_last=True)
    run = new_run('train', args.run_dir)
    write_json(run/'config.json', {'arguments': vars(args), 'environment': environment(device),
                                  'effective_batch': args.batch_size*args.accumulate,
                                  'train_pairs': len(train_data), 'validation_pairs': len(val) if val else 0})
    print(f'Run: {run}', flush=True)
    for epoch in range(first_epoch, args.epochs):
        model.train()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss_sum, step = 0.0, 0
        lr_used = optimizer.param_groups[0]['lr']
        optimizer.zero_grad(set_to_none=True)
        for microbatch, (lr, hr, _) in enumerate(loader):
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                sr = model(lr)
                loss = torch.nn.functional.l1_loss(sr, hr)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            loss_sum += loss.item()
            scaler.scale(loss / args.accumulate).backward()
            if (microbatch+1) % args.accumulate == 0:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.print_every == 0 or step == args.steps_per_epoch:
                    print(f'Epoch {epoch+1}, step {step}/{args.steps_per_epoch}, L1 {loss_sum/(microbatch+1):.6f}', flush=True)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - start
        scheduler.step()
        metrics = {'epoch': epoch+1, 'learning_rate': lr_used, 'mean_l1': loss_sum/len(loader),
                   'train_seconds': train_seconds, 'optimizer_steps': step}
        if device.type == 'cuda':
            metrics['train_peak_allocated_mib'] = torch.cuda.max_memory_allocated(device)/2**20
            metrics['train_peak_reserved_mib'] = torch.cuda.max_memory_reserved(device)/2**20
        improved = False
        if val:
            rows = evaluate(model, val, device, args.val_mode)
            score = statistics.mean(r['y_psnr'] for r in rows)
            metrics.update({'val_y_psnr': score, 'validation': rows})
            improved = score > best
            if improved: best = score
        state = {'format': 'mfcc-training-v1', 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'scaler': scaler.state_dict(), 'arguments': vars(args),
                 'completed_epochs': epoch+1, 'best_y_psnr': best,
                 'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
                 'loader_rng': generator.get_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        atomic_save(state, run/'last.pt')
        if improved: atomic_save(model.state_dict(), run/'best.pt')
        with (run/'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics)+'\n')
        print(metrics, flush=True)


if __name__ == '__main__':
    main()
