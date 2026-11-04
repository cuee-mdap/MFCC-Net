import hashlib
import json
import os
from pathlib import Path
import platform
from datetime import datetime, timezone
import random
import sys

import numpy as np
import torch
from PIL import __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]


def local_work():
    return Path('runs')


def new_run(kind, requested=None):
    path = Path(requested) if requested else local_work()/f'{kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def device_for(name):
    if name == 'auto': name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: check NVIDIA GPU/driver and install a CUDA-enabled torch build')
    return torch.device(name)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def environment(device):
    return {'python': sys.version.split()[0], 'torch': torch.__version__,
            'numpy': np.__version__, 'pillow': pillow_version,
            'platform': platform.platform(), 'device': str(device),
            'cuda_build': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'cpu_threads': torch.get_num_threads()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def checkpoint_for(scale, override=None):
    if override: return Path(override).resolve()
    manifest = json.loads((ROOT/'weights'/'manifest.json').read_text(encoding='utf-8'))
    entry = manifest['checkpoints'][str(scale)]
    path = ROOT/'weights'/entry['file']
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}; copy the verified Journal weight or pass --checkpoint')
    if sha256(path) != entry['sha256']:
        raise ValueError(f'Journal checkpoint hash mismatch: {path}')
    return path


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')


def atomic_save(payload, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)
