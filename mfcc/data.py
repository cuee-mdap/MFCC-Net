"""Paired RGB images; reads source images without writing dataset caches."""
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

EXTENSIONS = {'.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff'}


def image_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Image directory not found: {directory}')
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError(f'No image files in {directory}')
    if len({p.stem.lower() for p in files}) != len(files):
        raise ValueError(f'Duplicate image stems in {directory}')
    return files


def read_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


def tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255


def save_image(value, path):
    array = value.detach().cpu().squeeze(0).clamp(0, 1).mul(255).round()
    Image.fromarray(array.byte().permute(1, 2, 0).numpy()).save(path)


class Pairs(Dataset):
    def __init__(self, lr_dir, hr_dir, scale, patch=None, augment=False, limit=None):
        self.scale, self.patch, self.augment = scale, patch, augment
        self.hr = image_files(hr_dir)
        lr_by_name = {p.stem.lower(): p for p in image_files(lr_dir)}
        self.lr = []
        for hr in self.hr:
            names = [f'{hr.stem}x{scale}'.lower(), hr.stem.lower()]
            matches = [lr_by_name[n] for n in names if n in lr_by_name]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one LR match for {hr.name}: {matches}')
            self.lr.append(matches[0])
        if limit is not None:
            if limit < 1:
                raise ValueError('limit must be positive')
            self.lr, self.hr = self.lr[:limit], self.hr[:limit]

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, index):
        lr, hr = read_image(self.lr[index]), read_image(self.hr[index])
        h, w = lr.shape[:2]
        s = self.scale
        # Permit only the bottom/right mod-crop remainder used by SR benchmarks.
        dh, dw = hr.shape[0] - h * s, hr.shape[1] - w * s
        if not (0 <= dh < s and 0 <= dw < s):
            raise ValueError(f'Incompatible LR/HR shapes: {self.lr[index]} {lr.shape} / {hr.shape}')
        hr = hr[:h*s, :w*s]
        if self.patch is not None:
            p = self.patch
            if p < 1 or min(h, w) < p:
                raise ValueError(f'LR patch {p} does not fit {self.lr[index]}')
            x, y = random.randrange(w-p+1), random.randrange(h-p+1)
            lr, hr = lr[y:y+p, x:x+p], hr[y*s:(y+p)*s, x*s:(x+p)*s]
        if self.augment:
            hf, vf, tr = [random.random() < 0.5 for _ in range(3)]
            def apply(a):
                if hf: a = a[:, ::-1]
                if vf: a = a[::-1]
                if tr: a = a.transpose(1, 0, 2)
                return a
            lr, hr = apply(lr), apply(hr)
        return tensor(lr), tensor(hr), self.hr[index].stem


class EpochSamples(Dataset):
    """Repeat image indices, then let DataLoader shuffle them like the old trainer."""
    def __init__(self, pairs, count):
        self.pairs, self.count = pairs, count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return self.pairs[index % len(self.pairs)]


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
