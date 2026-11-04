# MFCC: Multi-FusNet of Cross Channel Network for Image Super-Resolution

PyTorch train/test code and the published weights (×2/×3/×4/×8) for

W. Ruangsang, S. Aramvith, T. Onoye, "Multi-FusNet of Cross Channel Network for Image Super-Resolution,"
*IEEE Access*, vol. 11, pp. 56287–56299, 2023, doi: [10.1109/ACCESS.2023.3282571](https://doi.org/10.1109/ACCESS.2023.3282571).

ภาษาไทย: [README.th.md](README.th.md)

The model (64 channels, 20 RCAB, reduction 16, RGB range 1; the three bodies share one residual group and convolution)
keeps the exact structure and state-dict format of the journal model, so the released weights load strictly and
reproduce Table 2 of the paper.

## Reproduced results (Table 2)

Measured with these weights on an RTX 3060 (PyTorch 2.2.2+cu121), on the same LR/HR benchmark files used for the paper,
all 20 cells (Set5, Set14, B100, Urban100, Manga109 × ×2/×3/×4/×8): max |ΔPSNR| vs the paper **0.0091 dB** (mean 0.0033),
max |ΔSSIM| **0.0001**. 16/20 cells round to the published two-decimal value; the other four differ by <0.01 dB
(×2 Set14 33.8565 vs 33.85, ×2 Urban100 32.6591 vs 32.65, ×4 Urban100 26.4746 vs 26.48, ×8 Set14 25.0048 vs 25.01).
Per-cell numbers: [results/table2_reproduction.md](results/table2_reproduction.md) · details: [VALIDATION.md](VALIDATION.md) (Thai).

| Scale | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| ×2 | 38.16 / 0.9611 | 33.86 / 0.9195 | 32.28 / 0.9009 | 32.66 / 0.9331 | 39.11 / 0.9780 |
| ×3 | 34.67 / 0.9294 | 30.51 / 0.8456 | 29.22 / 0.8079 | 28.64 / 0.8616 | 34.15 / 0.9478 |
| ×4 | 32.42 / 0.8973 | 28.73 / 0.7849 | 27.67 / 0.7398 | 26.47 / 0.7977 | 30.98 / 0.9131 |
| ×8 | 27.07 / 0.7762 | 25.00 / 0.6412 | 24.84 / 0.5980 | 22.54 / 0.6196 | 24.63 / 0.7791 |

Values measured with this repository (PSNR / SSIM on Y, border = scale).

## Complexity

| Scale | Params (this code) | Params (paper) | MACs, 1280×720 output (`complexity.py`) | Multi-Adds (paper) |
|---|---|---|---|---|
| ×2 | 1,861,547 | 1,861K | 1,140.1 G | 364.5 G |
| ×3 | 2,230,827 | 2,230K | 544.5 G | 374.1 G |
| ×4 | 2,156,971 | 2,157K | 354.2 G | 395.2 G |
| ×8 | 2,452,395 | 2,453K | 157.7 G | 518.2 G |

The parameter counts match Table 2. The Multi-Adds printed for MFCC in Table 2 rise with the scale, unlike the other
rows of that table (counted for a 1280×720 output); counted for a 1280×720 output, MFCC needs the MACs above.
Use these values when comparing with other models on that basis.

## Files

- `mfcc/model.py` — the journal model; `mfcc/metrics.py` — Y-PSNR / Y-SSIM; `mfcc/inference.py` — chop / direct inference
- `test.py` — evaluate LR/HR pairs, or upscale LR images only
- `reproduce.py` — run all benchmark cells and compare with the paper
- `train.py` — train from scratch / fine-tune / resume (gradient accumulation, optional CUDA AMP)
- `profile_gpu.py` — time and peak VRAM of a training step
- `verify.py` — check the four weights (SHA-256), shared weights, gradients and metrics
- `complexity.py` — parameters and Conv2d multiply-accumulates (real forward pass with hooks)
- `weights/mfcc_x{2,3,4,8}.pt` (~35 MB total) + `weights/manifest.json` (SHA-256, provenance, original config)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or the CPU wheel
pip install -r requirements.txt
python verify.py
```

Tested with PyTorch 2.2.2+cu121 (Linux, RTX 3060) and 2.12.1+cpu (Windows). `--device cuda` stops with an error if
CUDA is unavailable (no silent CPU fallback).

## Test

Benchmarks in the RCAN/SAN layout: `<set>/HR` and `<set>/LR_bicubic/X<scale>` (lower-case `x<scale>` also accepted).

```bash
# one set, one scale
python test.py --scale 4 --device cuda --lr-dir /path/to/benchmark/Set5/LR_bicubic/X4 --hr-dir /path/to/benchmark/Set5/HR --ssim
# all 20 cells, compared with the paper
python reproduce.py --benchmark /path/to/benchmark --out /path/to/new-output --device cuda
# your own LR images (no HR)
python test.py --scale 4 --device cuda --lr-dir /path/to/lr --save-images
```

Results (`metrics.json`, images with `--save-images`) go to `./runs/<timestamp>` or `--run-dir` (must be new).

- Default `--mode journal` = chop for ×2/×3/×4 and direct for ×8, as used for the paper's result images.
- PSNR: clamp/round to 8-bit, Y channel, border = scale, mean of per-image dB. `--ssim`: Gaussian 11×11, σ 1.5, Y, border = scale.
- `--mode chop --chop-min-size 20000` lowers memory for large images, but a different chop size can change PSNR
  (channel attention uses spatial means) — do not mix with journal-mode numbers.
- LR/HR names `0001x4.png` / `0001.png` or identical names; HR bottom/right borders are cropped to a multiple of the scale.

## Train

DIV2K 800 pairs with bicubic LR. Fine-tune example:

```bash
python train.py --scale 4 --device cuda \
  --lr-dir DIV2K/DIV2K_train_LR_bicubic/X4 --hr-dir DIV2K/DIV2K_train_HR \
  --val-lr-dir benchmark/Set5/LR_bicubic/X4 --val-hr-dir benchmark/Set5/HR \
  --init weights/mfcc_x4.pt --lr 1e-5 --batch-size 4 --accumulate 4 --lr-patch 48 --epochs 100 --steps-per-epoch 1000 --workers 0
```

From scratch: drop `--init`, start with `--lr 1e-4 --epochs 600` (paper settings; retraining is not guaranteed to give
the released weights, which were trained in several stages — see `weights/manifest.json`). `--lr-patch` is the LR patch
size for all scales; batch 4 × accumulate 4 = effective batch 16; Adam (0.9, 0.999), L1, flip/transpose augmentation;
LR halves every 200 epochs. Each run saves `last.pt` (model/optimizer/scheduler/scaler/RNG), `best.pt` (with validation),
`metrics.jsonl`, `config.json`. Resume with `--resume <run>/last.pt` and the same scale/batch/patch/schedule/seed/AMP settings.

Check memory first: `python profile_gpu.py --device cuda --scale 4 --batch-size 4 --lr-patch 48 --steps 10`.

## Changes from the original code

Positional `bias` that became a boolean stride fixed; strict state-dict loading with shared-alias checks; no deprecated
scheduler `verbose`; main guards on all entry points; direct image saving instead of nested multiprocessing; no binary
dataset cache or `rm -rf`; atomic checkpoints; `torch.load(weights_only=True)`.

## License and citation

MIT (© 2023 the authors); the code derives in part from EDSR-PyTorch (MIT, © 2018 Sanghyun Son) and code by Yiqun Mei (MIT, © 2020), whose notices are kept in [LICENSE](LICENSE). Cite with [CITATION.cff](CITATION.cff):

```bibtex
@article{ruangsang2023mfcc,
  author  = {Ruangsang, Watchara and Aramvith, Supavadee and Onoye, Takao},
  title   = {Multi-FusNet of Cross Channel Network for Image Super-Resolution},
  journal = {IEEE Access}, volume = {11}, pages = {56287--56299}, year = {2023},
  doi     = {10.1109/ACCESS.2023.3282571}
}
```
