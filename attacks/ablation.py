import os
import csv
import gc
import math
import time
import random
import shutil
import logging
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

DATA_ROOT = './data'
CKPT_ROOT = './checkpoints'
OUT_DIR = './results_clean/ablation'
MAIN_AUTOSPARSE_SUMMARY = './results_clean/autosparse/final_summary.csv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 10
IMG_SIZE = 64
EPOCHS = 20
BATCH_SIZE = 64
WORKERS = 4

AE_EPOCHS = 4
AE_LR = 1e-3
AE_MAX_BATCHES = 200
SCORE_BATCHES = 200

LR_STAGE1 = 1e-3
WEIGHT_DECAY = 1e-4
SPARSE_RATIO = 0.06
MASK_MODE = 'channel'
AE_REPAIR_STRENGTH = 8.0
SAMPLE_BETA = 0.50
HEAD_LR_MULT = 5.0
TRAIN_FINAL_HEAD_ONLY = True
AE_RESIDUAL_LAMBDA = 0.02

DEFAULT_BUDGETS = [0.01, 0.05, 0.10, 0.15]
FINAL_SEEDS = [42, 43, 44]
CI_T_DF2 = 4.302652729911275

# Selector baselines used in the ablation.
DEFAULT_SELECTORS = ['random', 'magnitude', 'gradient']
VALID_SELECTORS = tuple(DEFAULT_SELECTORS)

RE_LABEL = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]
CURRENT_SEED = 42
CHANNEL_RATIO = 0.15
CURRENT_SELECTOR = 'random'

CHECKPOINTS = [
    {'method':'SOPHON-IL','name':'sophon_il_cifar10','type':'sophon','arch':'resnet18','dataset':'CIFAR10','task':'CIFAR10','path':os.path.join(CKPT_ROOT,'sophon_il','sophon_il_cifar10.pth')},
    {'method':'SOPHON-IL','name':'sophon_il_cinic','type':'sophon','arch':'resnet18','dataset':'CINIC','task':'CINIC','path':os.path.join(CKPT_ROOT,'sophon_il','sophon_il_cinic.pth')},
    {'method':'SOPHON-IL','name':'sophon_il_mnist','type':'sophon','arch':'resnet18','dataset':'MNIST','task':'MNIST','path':os.path.join(CKPT_ROOT,'sophon_il','sophon_il_mnist.pth')},
    {'method':'SOPHON-IL','name':'sophon_il_stl10','type':'sophon','arch':'resnet18','dataset':'STL10','task':'STL10','path':os.path.join(CKPT_ROOT,'sophon_il','sophon_il_stl10.pth')},
    {'method':'SOPHON-IL','name':'sophon_il_svhn','type':'sophon','arch':'resnet18','dataset':'SVHN','task':'SVHN','path':os.path.join(CKPT_ROOT,'sophon_il','sophon_il_svhn.pth')},
    {'method':'SOPHON-KL','name':'sophon_kl_cifar10','type':'sophon','arch':'resnet18','dataset':'CIFAR10','task':'CIFAR10','path':os.path.join(CKPT_ROOT,'sophon_kl','sophon_kl_cifar10.pth')},
    {'method':'SOPHON-KL','name':'sophon_kl_cinic','type':'sophon','arch':'resnet18','dataset':'CINIC','task':'CINIC','path':os.path.join(CKPT_ROOT,'sophon_kl','sophon_kl_cinic.pth')},
    {'method':'SOPHON-KL','name':'sophon_kl_mnist','type':'sophon','arch':'resnet18','dataset':'MNIST','task':'MNIST','path':os.path.join(CKPT_ROOT,'sophon_kl','sophon_kl_mnist.pth')},
    {'method':'SOPHON-KL','name':'sophon_kl_stl10','type':'sophon','arch':'resnet18','dataset':'STL10','task':'STL10','path':os.path.join(CKPT_ROOT,'sophon_kl','sophon_kl_stl10.pth')},
    {'method':'SOPHON-KL','name':'sophon_kl_svhn','type':'sophon','arch':'resnet18','dataset':'SVHN','task':'SVHN','path':os.path.join(CKPT_ROOT,'sophon_kl','sophon_kl_svhn.pth')},
    {'method':'HNTL','name':'hntl_cifar_stl_vgg13','type':'vgg','arch':'vgg13','dataset':'STL10','task':'CIFAR10_to_STL10','path':os.path.join(CKPT_ROOT,'hntl_cifar_stl_vgg13.pth')},
    {'method':'CUPI','name':'cupi_cifar_stl_vgg13','type':'vgg','arch':'vgg13','dataset':'STL10','task':'CIFAR10_to_STL10','path':os.path.join(CKPT_ROOT,'cupi_cifar_stl_vgg13.pth')},
    {'method':'CUTI','name':'cuti_cifar_stl_vgg19','type':'vgg','arch':'vgg19','dataset':'STL10','task':'CIFAR10_to_STL10','path':os.path.join(CKPT_ROOT,'cuti_cifar_stl_vgg19_ntlbench.pth')},
]

FINAL_EPOCH_CSV = os.path.join(OUT_DIR, 'final_epoch_results.csv')
FINAL_SEED_CSV = os.path.join(OUT_DIR, 'final_seed_results.csv')
FINAL_SUMMARY_CSV = os.path.join(OUT_DIR, 'final_summary.csv')
COMPARISON_SUMMARY_CSV = os.path.join(OUT_DIR, 'comparison_with_autosparse.csv')
PROBE_CSV = os.path.join(OUT_DIR, 'probe_results.csv')
BUDGET_DETAILS_CSV = os.path.join(OUT_DIR, 'budget_details.csv')
SCORE_DIR = os.path.join(OUT_DIR, 'selector_scores')
PRED_DIR = os.path.join(OUT_DIR, 'predictions')
LOG_PATH = os.path.join(OUT_DIR, 'autosparse_ablation.log')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def strip_module(state):
    out = {}
    for k, v in state.items():
        while k.startswith('module.'):
            k = k[7:]
        out[k] = v
    return out


def load_state(path):
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    cleaned = {}

    for k, v in state.items():
        while k.startswith("module."):
            k = k[len("module."):]

        if k.startswith("classifier1."):
            k = "classifier." + k[len("classifier1."):]

        if k.startswith("classifier.3."):
            k = "classifier.2." + k[len("classifier.3."):]
        elif k.startswith("classifier.6."):
            k = "classifier.4." + k[len("classifier.6."):]

        cleaned[k] = v

    return cleaned


class STL10Mapped(datasets.STL10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, RE_LABEL[target]


def get_loaders(dataset):
    # AutoSparse preprocessing.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    tf_rgb_train = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    tf_rgb_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    tf_mnist = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize,
    ])
    d = dataset.upper()
    if d == 'CIFAR10':
        train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=tf_rgb_train)
        test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=tf_rgb_test)
    elif d == 'MNIST':
        train_set = datasets.MNIST(DATA_ROOT, train=True, download=False, transform=tf_mnist)
        test_set = datasets.MNIST(DATA_ROOT, train=False, download=False, transform=tf_mnist)
    elif d == 'SVHN':
        train_set = datasets.SVHN(DATA_ROOT, split='train', download=False, transform=tf_rgb_train)
        test_set = datasets.SVHN(DATA_ROOT, split='test', download=False, transform=tf_rgb_test)
    elif d == 'STL10':
        train_set = STL10Mapped(DATA_ROOT, split='train', download=False, transform=tf_rgb_train)
        test_set = STL10Mapped(DATA_ROOT, split='test', download=False, transform=tf_rgb_test)
    elif d == 'CINIC':
        cinic_root = os.path.join(DATA_ROOT, 'CINIC10')
        train_dir = os.path.join(cinic_root, 'train')
        valid_dir = os.path.join(cinic_root, 'valid')
        test_dir = os.path.join(cinic_root, 'test')
        if os.path.exists(train_dir) and os.path.exists(valid_dir):
            train_set = datasets.ImageFolder(train_dir, transform=tf_rgb_train)
            test_set = datasets.ImageFolder(valid_dir, transform=tf_rgb_test)
        elif os.path.exists(train_dir) and os.path.exists(test_dir):
            train_set = datasets.ImageFolder(train_dir, transform=tf_rgb_train)
            test_set = datasets.ImageFolder(test_dir, transform=tf_rgb_test)
        else:
            raise FileNotFoundError(f'CINIC folders not found inside {cinic_root}')
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    g = torch.Generator()
    g.manual_seed(CURRENT_SEED)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, generator=g,
        num_workers=WORKERS, pin_memory=(DEVICE == 'cuda'), drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=(DEVICE == 'cuda'),
    )
    return train_loader, test_loader, len(train_set), len(test_set)


class CustomVGG(nn.Module):
    def __init__(self, arch, num_classes=10):
        super().__init__()

        if arch == "vgg13":
            base = models.vgg13(weights=None)
        elif arch == "vgg19":
            base = models.vgg19(weights=None)
        else:
            raise ValueError(f"Unknown VGG arch: {arch}")

        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def build_model(info):
    if info['type'] == 'sophon':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif info['type'] == 'vgg':
        model = CustomVGG(info['arch'], NUM_CLASSES)
    else:
        raise ValueError(info['type'])
    state = load_state(info['path'])
    missing, unexpected = model.load_state_dict(state, strict=False)
    return model, list(missing), list(unexpected)


def safe_name(name):
    return name.replace('.', '__')


def get_sample_beta(epoch, total_epochs, max_beta):
    warmup = max(1, total_epochs // 4)
    if epoch <= warmup:
        return 0.0
    return max_beta * min(1.0, (epoch - warmup) / max(1, total_epochs - warmup))


class GenericConvHook:
    def __init__(self, model):
        self.outputs = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inp, out):
            if out.ndim == 4:
                self.outputs[name] = F.adaptive_avg_pool2d(out.detach(), 1).flatten(1)
        return hook

    def clear(self):
        self.outputs = {}

    def remove(self):
        for h in self.handles:
            h.remove()


class TrainConvHook:
    def __init__(self, model):
        self.outputs = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inp, out):
            if out.ndim == 4:
                self.outputs[name] = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return hook

    def clear(self):
        self.outputs = {}

    def remove(self):
        for h in self.handles:
            h.remove()


class AEProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = max(32, dim // 2)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return self.net(x)


def build_ae_bank(model, train_loader):
    hooker = GenericConvHook(model)
    model.eval()
    x, _ = next(iter(train_loader))
    x = x.to(DEVICE)
    with torch.no_grad():
        hooker.clear()
        model(x)
    ae_bank = nn.ModuleDict()
    layer_dims = {}
    for name, feat in hooker.outputs.items():
        dim = feat.shape[1]
        ae = AEProbe(dim).to(DEVICE)
        torch.manual_seed(CURRENT_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(CURRENT_SEED)
        for p in ae.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)
            else:
                nn.init.zeros_(p)
        ae_bank[safe_name(name)] = ae
        layer_dims[name] = dim
    hooker.remove()
    if not layer_dims:
        raise RuntimeError('No Conv2d layers found for AE probe.')
    return ae_bank, layer_dims


@torch.no_grad()
def compute_ae_channel_scores(model, ae_bank, score_loader, hooker):
    # Compute reconstruction scores using the AutoSparse probing pipeline.
    model.eval()
    ae_bank.eval()
    err_sum = {}
    count = 0
    for i, (x, _) in enumerate(tqdm(score_loader, desc='AE channel scores', leave=False)):
        if i >= SCORE_BATCHES:
            break
        x = x.to(DEVICE, non_blocking=True)
        hooker.clear()
        model(x)
        for name, feat in hooker.outputs.items():
            key = safe_name(name)
            if key not in ae_bank:
                continue
            rec = ae_bank[key](feat)
            err = (feat - rec).pow(2).mean(dim=0)
            err_sum[name] = err.detach().cpu() if name not in err_sum else err_sum[name] + err.detach().cpu()
        count += 1
    scores = {}
    for name, err in err_sum.items():
        err = err / max(count, 1)
        scores[name] = (err / (err.max() + 1e-8)).float()
    return scores


def reset_cuda_peak():
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_memory_mb():
    if DEVICE != 'cuda':
        return 0.0, 0.0
    torch.cuda.synchronize()
    return (
        torch.cuda.max_memory_allocated() / (1024 ** 2),
        torch.cuda.max_memory_reserved() / (1024 ** 2),
    )


def gpu_name():
    return torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'


def train_ae_probe(model, train_loader, logger, info):
    # Use the AutoSparse AE probe while replacing only the channel-ranking method.
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    reset_cuda_peak()
    probe_start = time.perf_counter()
    ae_bank, layer_dims = build_ae_bank(model, train_loader)
    hooker = GenericConvHook(model)
    score_loader = DataLoader(
        train_loader.dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=(DEVICE == 'cuda'),
    )
    optimizer = torch.optim.AdamW(ae_bank.parameters(), lr=AE_LR, weight_decay=WEIGHT_DECAY)
    ae_train_time = 0.0
    ae_last_loss = 0.0
    ae_batches_total = 0
    for ep in range(1, AE_EPOCHS + 1):
        ae_bank.train()
        total_loss = 0.0
        count = 0
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for i, (x, _) in enumerate(tqdm(train_loader, desc=f'AE probe E{ep}', leave=False)):
            if i >= AE_MAX_BATCHES:
                break
            x = x.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                hooker.clear()
                model(x)
                feats = {k: v.detach() for k, v in hooker.outputs.items()}
            loss = 0.0
            used = 0
            for name, feat in feats.items():
                key = safe_name(name)
                if key not in ae_bank:
                    continue
                loss = loss + F.mse_loss(ae_bank[key](feat), feat)
                used += 1
            if used == 0:
                raise RuntimeError('AE probe captured zero Conv activations.')
            loss = loss / used
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            count += 1
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        ae_train_time += elapsed
        ae_last_loss = total_loss / max(count, 1)
        ae_batches_total += count
        logger.info(
            'AE PROBE | %s | selector=%s | seed=%d | budget=%.0f%% | epoch=%02d/%d | loss=%.6f | time=%.2fs',
            info['name'], CURRENT_SELECTOR, CURRENT_SEED, CHANNEL_RATIO * 100.0,
            ep, AE_EPOCHS, ae_last_loss, elapsed,
        )
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    score_start = time.perf_counter()
    original_ae_scores = compute_ae_channel_scores(model, ae_bank, score_loader, hooker)
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    ae_score_time = time.perf_counter() - score_start
    hooker.remove()
    probe_time = time.perf_counter() - probe_start
    probe_peak_alloc_mb, probe_peak_reserved_mb = peak_memory_mb()
    ae_bank.eval()
    for p in ae_bank.parameters():
        p.requires_grad = False
    logger.info(
        'AE PROBE RESULT | %s | selector=%s | seed=%d | budget=%.0f%% | conv_layers=%d | AE_train=%.2fs | AE_score=%.2fs | total=%.2fs | peak=%.1fMB',
        info['name'], CURRENT_SELECTOR, CURRENT_SEED, CHANNEL_RATIO * 100.0,
        len(layer_dims), ae_train_time, ae_score_time, probe_time, probe_peak_alloc_mb,
    )
    del optimizer
    return ae_bank, original_ae_scores, len(layer_dims), {
        'ae_train_time_sec': ae_train_time,
        'ae_score_time_sec': ae_score_time,
        'probe_time_sec': probe_time,
        'probe_peak_alloc_mb': probe_peak_alloc_mb,
        'probe_peak_reserved_mb': probe_peak_reserved_mb,
        'ae_last_loss': ae_last_loss,
        'ae_batches_used': ae_batches_total,
    }


def normalize_channel_score(score):
    score = score.detach().float().cpu()
    if not torch.isfinite(score).all():
        raise RuntimeError('Non-finite channel score.')
    mx = float(score.max()) if score.numel() else 0.0
    return score / (mx + 1e-8) if mx > 0 else torch.zeros_like(score)


def validate_scores(model, scores):
    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    missing, wrong = [], []
    for name, module in convs:
        if name not in scores:
            missing.append(name)
        elif int(scores[name].numel()) != int(module.out_channels):
            wrong.append((name, int(scores[name].numel()), int(module.out_channels)))
    if missing or wrong:
        raise RuntimeError(f'Invalid selector scores | missing={missing} | wrong={wrong}')
    return len(convs)


def random_channel_scores(model, seed):
    # Random channel ranking with a private generator, so recovery RNG is untouched.
    g = torch.Generator(device='cpu')
    g.manual_seed(1000003 + int(seed))
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            scores[name] = normalize_channel_score(torch.rand(module.out_channels, generator=g))
    validate_scores(model, scores)
    return scores


def magnitude_channel_scores(model):
    # Mean absolute kernel magnitude per Conv output channel.
    scores = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                s = module.weight.detach().abs().flatten(1).mean(dim=1)
                scores[name] = normalize_channel_score(s)
    validate_scores(model, scores)
    return scores


def gradient_channel_scores(info, train_dataset):
    # Mean absolute target-loss gradient per Conv output channel.
    score_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=(DEVICE == 'cuda'), drop_last=False,
    )
    model, _, _ = build_model(info)
    model = model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    convs = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight.requires_grad = True
            convs.append((name, module))
    if not convs:
        raise RuntimeError('Gradient selector found no Conv2d layers.')
    grad_sum = {name: torch.zeros(module.out_channels, dtype=torch.float64) for name, module in convs}
    count = 0
    for i, (x, y) in enumerate(tqdm(score_loader, desc='Gradient channel scores', leave=False)):
        if i >= SCORE_BATCHES:
            break
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        for name, module in convs:
            if module.weight.grad is None:
                raise RuntimeError(f'No gradient for Conv layer {name}')
            g = module.weight.grad.detach().abs().flatten(1).mean(dim=1).double().cpu()
            grad_sum[name] += g
        count += 1
    if count == 0:
        raise RuntimeError('Gradient selector used zero batches.')
    scores = {name: normalize_channel_score((total / count).float()) for name, total in grad_sum.items()}
    validate_scores(model, scores)
    del model
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return scores, count


def compute_selector_scores(selector, info, train_dataset, seed, logger):
    # Measure selector-specific overhead separately. This matters most for the
    # gradient baseline, which needs a backward pass through a fresh model.
    reset_cuda_peak()
    model, _, _ = build_model(info)
    model = model.to(DEVICE)
    model.eval()
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    gradient_batches = 0
    if selector == 'random':
        scores = random_channel_scores(model, seed)
    elif selector == 'magnitude':
        scores = magnitude_channel_scores(model)
    elif selector == 'gradient':
        del model
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        scores, gradient_batches = gradient_channel_scores(info, train_dataset)
        model, _, _ = build_model(info)
        model = model.to(DEVICE)
        model.eval()
    else:
        raise ValueError(f'Unknown selector {selector}')
    num_layers = validate_scores(model, scores)
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    score_time = time.perf_counter() - started
    selector_peak_alloc_mb, selector_peak_reserved_mb = peak_memory_mb()
    logger.info(
        'SELECTOR SCORE | %s | selector=%s | seed=%d | budget=%.0f%% | layers=%d | '
        'gradient_batches=%d | time=%.2fs | peak=%.1fMB',
        info['name'], selector, seed, CHANNEL_RATIO * 100.0, num_layers,
        gradient_batches, score_time, selector_peak_alloc_mb,
    )
    del model
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return (
        scores, num_layers, score_time, gradient_batches,
        selector_peak_alloc_mb, selector_peak_reserved_mb,
    )


def save_selector_scores(info, selector, seed, budget, scores):
    os.makedirs(SCORE_DIR, exist_ok=True)
    rows = []
    for layer, score in scores.items():
        order = torch.argsort(score, descending=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(1, len(order) + 1)
        for ch in range(score.numel()):
            rows.append({
                'method': info['method'], 'checkpoint_name': info['name'],
                'dataset': info['dataset'], 'arch': info['arch'], 'selector': selector,
                'seed': seed, 'budget': budget, 'budget_percent': 100.0 * budget,
                'layer': layer, 'channel': ch, 'normalized_score': float(score[ch]),
                'rank_within_layer': int(rank[ch]),
            })
    path = os.path.join(SCORE_DIR, f"{info['name']}__{selector}__b{int(round(budget*100))}__seed{seed}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def make_selector_mask(weight, channel_score):
    with torch.no_grad():
        out_c = weight.shape[0]
        if channel_score is None:
            raise RuntimeError('Missing selector score.')
        score = channel_score[:out_c].to(weight.device)
        if score.numel() != out_c:
            raise RuntimeError(f'Score/channel mismatch: {score.numel()} vs {out_c}')
        if MASK_MODE == 'channel':
            k = max(1, int(out_c * CHANNEL_RATIO))
            k = min(k, out_c)
            idx = torch.topk(score, k=k, largest=True).indices
            mask = torch.zeros_like(weight)
            mask[idx, :, :, :] = 1.0
            return mask
        score = score.view(out_c, 1, 1, 1)
        priority = weight.detach().abs() * (1.0 + AE_REPAIR_STRENGTH * score.pow(2))
        flat = priority.flatten()
        k = max(1, int(flat.numel() * SPARSE_RATIO))
        th = torch.topk(flat, k=k, largest=True).values[-1]
        return (priority >= th).float()


def apply_masks(model, channel_scores):
    for p in model.parameters():
        p.requires_grad = False
    masks, originals, hooks = {}, {}, []
    effective = 0
    selected_conv_weight_params = 0
    trainable_conv_bias_params = 0
    final_head_params = 0
    selected_channels_total = 0
    candidate_channels_total = 0
    trained_conv_layers = 0
    detail_rows = []
    linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    final_linear_name = linear_layers[-1][0] if linear_layers else None
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    selection_start = time.perf_counter()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name not in channel_scores:
                raise RuntimeError(f'Missing selector score for {name}')
            mask = make_selector_mask(module.weight.data, channel_scores[name])
            module.weight.requires_grad = True
            selected = int(mask.sum().item())
            effective += selected
            selected_conv_weight_params += selected
            trained_conv_layers += 1
            selected_channels = int((mask.view(mask.shape[0], -1).sum(dim=1) > 0).sum().item())
            selected_channels_total += selected_channels
            candidate_channels_total += module.out_channels
            def hook_fn(grad, m=mask):
                return grad * m.to(grad.device)
            hooks.append(module.weight.register_hook(hook_fn))
            masks[module.weight] = mask.to(module.weight.device)
            originals[module.weight] = module.weight.detach().clone()
            bias_params = 0
            if module.bias is not None:
                module.bias.requires_grad = True
                bmask = torch.ones_like(module.bias.data)
                masks[module.bias] = bmask
                originals[module.bias] = module.bias.detach().clone()
                effective += module.bias.numel()
                trainable_conv_bias_params += module.bias.numel()
                bias_params = module.bias.numel()
            detail_rows.append({
                'layer': name, 'module_type': 'Conv2d', 'out_channels': module.out_channels,
                'selected_channels': selected_channels, 'weight_tensor_params': module.weight.numel(),
                'selected_weight_params': selected, 'bias_params': bias_params, 'is_final_head': 0,
            })
        if isinstance(module, nn.Linear) and TRAIN_FINAL_HEAD_ONLY and name == final_linear_name:
            module.weight.requires_grad = True
            effective += module.weight.numel()
            final_head_params += module.weight.numel()
            bias_params = 0
            if module.bias is not None:
                module.bias.requires_grad = True
                effective += module.bias.numel()
                final_head_params += module.bias.numel()
                bias_params = module.bias.numel()
            detail_rows.append({
                'layer': name, 'module_type': 'Linear', 'out_channels': module.out_features,
                'selected_channels': module.out_features, 'weight_tensor_params': module.weight.numel(),
                'selected_weight_params': module.weight.numel(), 'bias_params': bias_params, 'is_final_head': 1,
            })
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    selection_time = time.perf_counter() - selection_start
    return masks, originals, effective, trained_conv_layers, {
        'selected_conv_weight_params': selected_conv_weight_params,
        'trainable_conv_bias_params': trainable_conv_bias_params,
        'final_head_params': final_head_params,
        'selected_channels_total': selected_channels_total,
        'candidate_channels_total': candidate_channels_total,
        'selection_time_sec': selection_time,
        'detail_rows': detail_rows,
        'hooks': hooks,
    }


def restore_unselected(masks, originals):
    with torch.no_grad():
        for p, m in masks.items():
            p.data.copy_(p.data * m + originals[p] * (1.0 - m))


def cleanup_hooks(mask_stats):
    for h in mask_stats.get('hooks', []):
        h.remove()


@torch.no_grad()
def evaluate(model, loader, return_predictions=False):
    model.eval()
    ce = nn.CrossEntropyLoss()
    correct = total = 0
    loss_sum = 0.0
    rows = []
    sample_index = 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y_dev = y.to(DEVICE, non_blocking=True)
        out = model(x)
        loss_sum += ce(out, y_dev).item()
        pred = out.argmax(1)
        correct += (pred == y_dev).sum().item()
        total += y_dev.size(0)
        if return_predictions:
            for yt, yp in zip(y.cpu().tolist(), pred.cpu().tolist()):
                rows.append({'sample_index': sample_index, 'y_true': int(yt), 'y_pred': int(yp)})
                sample_index += 1
    acc = 100.0 * correct / max(total, 1)
    avg_loss = loss_sum / max(len(loader), 1)
    if return_predictions:
        return acc, avg_loss, pd.DataFrame(rows)
    return acc, avg_loss


def append_csv(path, row, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in columns})


EPOCH_COLUMNS = [
    'method','checkpoint_name','dataset','task','arch','checkpoint','attack','selector','seed',
    'budget','budget_percent','epoch','epochs','lr','head_lr','batch_size','weight_decay',
    'epoch_acc','epoch_loss','best_acc_so_far','best_epoch_so_far','epoch_train_time_sec',
    'epoch_test_time_sec','cumulative_train_time_sec','effective_trainable','total_params',
    'effective_percent','selected_conv_weight_params','trainable_conv_bias_params','final_head_params',
    'selected_channels_total','candidate_channels_total','num_conv_layers_scored','num_conv_layers_trained',
    'selection_time_sec','gpu_name','missing_keys','unexpected_keys',
]
SEED_COLUMNS = [
    'method','checkpoint_name','dataset','task','arch','checkpoint','attack','selector','seed','budget',
    'budget_percent','epoch20_acc','epoch20_loss','best_acc','best_epoch','lr','head_lr','batch_size',
    'weight_decay','epochs','ae_train_time_sec','ae_score_time_sec','ae_probe_time_sec',
    'selector_score_time_sec','gradient_score_batches','selection_time_sec','train_time_sec','test_time_sec',
    'total_attack_time_sec','probe_peak_alloc_mb','probe_peak_reserved_mb',
    'selector_peak_alloc_mb','selector_peak_reserved_mb','recovery_peak_alloc_mb',
    'recovery_peak_reserved_mb','peak_attack_alloc_mb','peak_attack_reserved_mb','effective_trainable',
    'total_params','effective_percent','selected_conv_weight_params','trainable_conv_bias_params',
    'final_head_params','selected_channels_total','candidate_channels_total','num_conv_layers_scored',
    'num_conv_layers_trained','train_examples','test_examples','score_file','prediction_file','gpu_name',
    'missing_keys','unexpected_keys',
]
PROBE_COLUMNS = [
    'method','checkpoint_name','dataset','task','arch','checkpoint','selector','seed','budget','budget_percent',
    'ae_epochs','ae_lr','ae_max_batches','ae_score_batches','ae_batches_used','ae_last_loss','num_conv_layers',
    'ae_train_time_sec','ae_score_time_sec','ae_probe_time_sec','selector_score_time_sec','gradient_score_batches',
    'probe_peak_alloc_mb','probe_peak_reserved_mb','selector_peak_alloc_mb','selector_peak_reserved_mb',
    'score_file','gpu_name',
]
BUDGET_DETAIL_COLUMNS = [
    'method','checkpoint_name','dataset','task','arch','checkpoint','selector','seed','budget','budget_percent',
    'layer','module_type','out_channels','selected_channels','weight_tensor_params','selected_weight_params',
    'bias_params','is_final_head',
]


def write_budget_details(info, selector, seed, budget, detail_rows):
    for d in detail_rows:
        append_csv(BUDGET_DETAILS_CSV, {
            'method': info['method'], 'checkpoint_name': info['name'], 'dataset': info['dataset'],
            'task': info['task'], 'arch': info['arch'], 'checkpoint': info['path'], 'selector': selector,
            'seed': seed, 'budget': budget, 'budget_percent': 100.0 * budget, **d,
        }, BUDGET_DETAIL_COLUMNS)


def train_recovery(info, selector, model, missing, unexpected, channel_scores, num_scored_layers,
                   train_loader, test_loader, ae_bank, probe_stats, selector_score_time,
                   gradient_score_batches, selector_peak_alloc_mb, selector_peak_reserved_mb,
                   score_file, n_train, n_test, logger):
    # Continue with the same loader and RNG state after AE probing.
    reset_cuda_peak()
    masks, originals, effective, trained_conv_layers, mask_stats = apply_masks(model, channel_scores)
    model = model.to(DEVICE)
    ae_bank = ae_bank.to(DEVICE)
    ae_bank.eval()
    for p in ae_bank.parameters():
        p.requires_grad = False
    write_budget_details(info, selector, CURRENT_SEED, CHANNEL_RATIO, mask_stats['detail_rows'])
    train_hook = TrainConvHook(model)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * effective / total
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if '.fc.' in name or name.startswith('fc.') or 'classifier' in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    groups = []
    if backbone_params:
        groups.append({'params': backbone_params, 'lr': LR_STAGE1})
    if head_params:
        groups.append({'params': head_params, 'lr': LR_STAGE1 * HEAD_LR_MULT})
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR_STAGE1 * 0.05,
    )
    best_acc, best_epoch = -1.0, -1
    final_acc = final_loss = None
    cumulative_train_time = 0.0
    cumulative_test_time = 0.0
    logger.info(
        'RECOVERY | %s | selector=%s | seed=%d | budget=%.0f%% | effective=%d/%d (%.6f%%) | conv=%d/%d',
        info['name'], selector, CURRENT_SEED, CHANNEL_RATIO * 100.0,
        effective, total, pct, num_scored_layers, trained_conv_layers,
    )
    for epoch in range(1, EPOCHS + 1):
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        train_start = time.perf_counter()
        model.train()
        for x, y in tqdm(
            train_loader,
            desc=f"{info['name']} {selector} B{int(CHANNEL_RATIO*100)} S{CURRENT_SEED} E{epoch}",
            leave=False,
        ):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            train_hook.clear()
            logits = model(x)
            ae_loss = 0.0
            used = 0
            for name, feat in train_hook.outputs.items():
                key = safe_name(name)
                if key in ae_bank:
                    ae_loss = ae_loss + F.mse_loss(ae_bank[key](feat), feat)
                    used += 1
            if used > 0:
                ae_loss = ae_loss / used
            with torch.no_grad():
                conf = F.softmax(logits, dim=1).max(dim=1).values
                beta = get_sample_beta(epoch, EPOCHS, SAMPLE_BETA)
                sw = 1.0 + beta * (1.0 - conf)
                sw = sw / sw.mean().clamp_min(1e-8)
            loss_each = F.cross_entropy(logits, y, reduction='none')
            loss = (loss_each * sw).mean()
            if used > 0 and AE_RESIDUAL_LAMBDA > 0.0:
                loss = loss + AE_RESIDUAL_LAMBDA * ae_loss
            loss.backward()
            optimizer.step()
            restore_unselected(masks, originals)
        scheduler.step()
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        epoch_train_time = time.perf_counter() - train_start
        cumulative_train_time += epoch_train_time
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        test_start = time.perf_counter()
        acc, loss_val = evaluate(model, test_loader)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        epoch_test_time = time.perf_counter() - test_start
        cumulative_test_time += epoch_test_time
        final_acc, final_loss = acc, loss_val
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
        append_csv(FINAL_EPOCH_CSV, {
            'method': info['method'], 'checkpoint_name': info['name'], 'dataset': info['dataset'],
            'task': info['task'], 'arch': info['arch'], 'checkpoint': info['path'],
            'attack': 'autosparse_ablation', 'selector': selector, 'seed': CURRENT_SEED,
            'budget': CHANNEL_RATIO, 'budget_percent': 100.0 * CHANNEL_RATIO,
            'epoch': epoch, 'epochs': EPOCHS, 'lr': LR_STAGE1, 'head_lr': LR_STAGE1 * HEAD_LR_MULT,
            'batch_size': BATCH_SIZE, 'weight_decay': WEIGHT_DECAY,
            'epoch_acc': round(acc, 4), 'epoch_loss': round(loss_val, 6),
            'best_acc_so_far': round(best_acc, 4), 'best_epoch_so_far': best_epoch,
            'epoch_train_time_sec': round(epoch_train_time, 4),
            'epoch_test_time_sec': round(epoch_test_time, 4),
            'cumulative_train_time_sec': round(cumulative_train_time, 4),
            'effective_trainable': effective, 'total_params': total, 'effective_percent': round(pct, 6),
            'selected_conv_weight_params': mask_stats['selected_conv_weight_params'],
            'trainable_conv_bias_params': mask_stats['trainable_conv_bias_params'],
            'final_head_params': mask_stats['final_head_params'],
            'selected_channels_total': mask_stats['selected_channels_total'],
            'candidate_channels_total': mask_stats['candidate_channels_total'],
            'num_conv_layers_scored': num_scored_layers,
            'num_conv_layers_trained': trained_conv_layers,
            'selection_time_sec': round(mask_stats['selection_time_sec'], 6),
            'gpu_name': gpu_name(), 'missing_keys': len(missing), 'unexpected_keys': len(unexpected),
        }, EPOCH_COLUMNS)
        logger.info(
            'ABLATION | %s | selector=%s | seed=%d | budget=%.0f%% | epoch=%02d/%d | acc=%.2f%% | loss=%.4f | best=%.2f%%@%d | train=%.1fs | test=%.2fs',
            info['name'], selector, CURRENT_SEED, CHANNEL_RATIO * 100.0,
            epoch, EPOCHS, acc, loss_val, best_acc, best_epoch, epoch_train_time, epoch_test_time,
        )

    final_acc2, final_loss2, pred_df = evaluate(model, test_loader, return_predictions=True)
    if abs(final_acc2 - final_acc) > 1e-9:
        raise RuntimeError(f'Final evaluation mismatch: {final_acc} vs {final_acc2}')
    os.makedirs(PRED_DIR, exist_ok=True)
    pred_path = os.path.join(
        PRED_DIR,
        f"{info['name']}__{selector}__b{int(round(CHANNEL_RATIO*100))}__seed{CURRENT_SEED}.csv",
    )
    pred_df.to_csv(pred_path, index=False)
    recovery_peak_alloc_mb, recovery_peak_reserved_mb = peak_memory_mb()
    peak_attack_alloc_mb = max(
        probe_stats['probe_peak_alloc_mb'],
        selector_peak_alloc_mb,
        recovery_peak_alloc_mb,
    )
    peak_attack_reserved_mb = max(
        probe_stats['probe_peak_reserved_mb'],
        selector_peak_reserved_mb,
        recovery_peak_reserved_mb,
    )
    total_attack_time = (
        probe_stats['probe_time_sec'] + selector_score_time
        + mask_stats['selection_time_sec'] + cumulative_train_time
    )
    append_csv(FINAL_SEED_CSV, {
        'method': info['method'], 'checkpoint_name': info['name'], 'dataset': info['dataset'],
        'task': info['task'], 'arch': info['arch'], 'checkpoint': info['path'],
        'attack': 'autosparse_ablation', 'selector': selector, 'seed': CURRENT_SEED,
        'budget': CHANNEL_RATIO, 'budget_percent': 100.0 * CHANNEL_RATIO,
        'epoch20_acc': round(final_acc, 4), 'epoch20_loss': round(final_loss, 6),
        'best_acc': round(best_acc, 4), 'best_epoch': best_epoch,
        'lr': LR_STAGE1, 'head_lr': LR_STAGE1 * HEAD_LR_MULT,
        'batch_size': BATCH_SIZE, 'weight_decay': WEIGHT_DECAY, 'epochs': EPOCHS,
        'ae_train_time_sec': round(probe_stats['ae_train_time_sec'], 4),
        'ae_score_time_sec': round(probe_stats['ae_score_time_sec'], 4),
        'ae_probe_time_sec': round(probe_stats['probe_time_sec'], 4),
        'selector_score_time_sec': round(selector_score_time, 4),
        'gradient_score_batches': gradient_score_batches,
        'selection_time_sec': round(mask_stats['selection_time_sec'], 6),
        'train_time_sec': round(cumulative_train_time, 4),
        'test_time_sec': round(cumulative_test_time, 4),
        'total_attack_time_sec': round(total_attack_time, 4),
        'probe_peak_alloc_mb': round(probe_stats['probe_peak_alloc_mb'], 4),
        'probe_peak_reserved_mb': round(probe_stats['probe_peak_reserved_mb'], 4),
        'selector_peak_alloc_mb': round(selector_peak_alloc_mb, 4),
        'selector_peak_reserved_mb': round(selector_peak_reserved_mb, 4),
        'recovery_peak_alloc_mb': round(recovery_peak_alloc_mb, 4),
        'recovery_peak_reserved_mb': round(recovery_peak_reserved_mb, 4),
        'peak_attack_alloc_mb': round(peak_attack_alloc_mb, 4),
        'peak_attack_reserved_mb': round(peak_attack_reserved_mb, 4),
        'effective_trainable': effective, 'total_params': total, 'effective_percent': round(pct, 6),
        'selected_conv_weight_params': mask_stats['selected_conv_weight_params'],
        'trainable_conv_bias_params': mask_stats['trainable_conv_bias_params'],
        'final_head_params': mask_stats['final_head_params'],
        'selected_channels_total': mask_stats['selected_channels_total'],
        'candidate_channels_total': mask_stats['candidate_channels_total'],
        'num_conv_layers_scored': num_scored_layers, 'num_conv_layers_trained': trained_conv_layers,
        'train_examples': n_train, 'test_examples': n_test, 'score_file': score_file,
        'prediction_file': pred_path, 'gpu_name': gpu_name(),
        'missing_keys': len(missing), 'unexpected_keys': len(unexpected),
    }, SEED_COLUMNS)
    logger.info(
        'RESULT | %s | selector=%s | seed=%d | budget=%.0f%% | epoch20=%.4f%% | best=%.4f%%@%d | AE_probe=%.2fs | selector_score=%.2fs | train=%.2fs | attack=%.2fs',
        info['name'], selector, CURRENT_SEED, CHANNEL_RATIO * 100.0,
        final_acc, best_acc, best_epoch, probe_stats['probe_time_sec'],
        selector_score_time, cumulative_train_time, total_attack_time,
    )
    train_hook.remove()
    cleanup_hooks(mask_stats)
    del optimizer, masks, originals
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()


def run_one(info, selector, seed, budget, logger):
    global CURRENT_SEED, CHANNEL_RATIO, CURRENT_SELECTOR
    CURRENT_SEED = seed
    CHANNEL_RATIO = budget
    CURRENT_SELECTOR = selector
    set_seed(seed)
    train_loader, test_loader, n_train, n_test = get_loaders(info['dataset'])
    model, _, _ = build_model(info)
    model = model.to(DEVICE)

    # Keep the submitted AE probe/recovery mechanism; skip only AE as a selector.
    ae_bank, original_ae_scores, _, probe_stats = train_ae_probe(model, train_loader, logger, info)

    # Alternative scoring must not perturb the RNG stream used by recovery.
    recovery_rng_state = capture_rng_state()
    (
        selector_scores, num_scored_layers, selector_score_time, gradient_batches,
        selector_peak_alloc_mb, selector_peak_reserved_mb,
    ) = compute_selector_scores(
        selector, info, train_loader.dataset, seed, logger,
    )
    restore_rng_state(recovery_rng_state)

    score_file = save_selector_scores(info, selector, seed, budget, selector_scores)
    append_csv(PROBE_CSV, {
        'method': info['method'], 'checkpoint_name': info['name'], 'dataset': info['dataset'],
        'task': info['task'], 'arch': info['arch'], 'checkpoint': info['path'],
        'selector': selector, 'seed': seed, 'budget': budget, 'budget_percent': 100.0 * budget,
        'ae_epochs': AE_EPOCHS, 'ae_lr': AE_LR, 'ae_max_batches': AE_MAX_BATCHES,
        'ae_score_batches': SCORE_BATCHES, 'ae_batches_used': probe_stats['ae_batches_used'],
        'ae_last_loss': probe_stats['ae_last_loss'], 'num_conv_layers': num_scored_layers,
        'ae_train_time_sec': round(probe_stats['ae_train_time_sec'], 4),
        'ae_score_time_sec': round(probe_stats['ae_score_time_sec'], 4),
        'ae_probe_time_sec': round(probe_stats['probe_time_sec'], 4),
        'selector_score_time_sec': round(selector_score_time, 4),
        'gradient_score_batches': gradient_batches,
        'probe_peak_alloc_mb': round(probe_stats['probe_peak_alloc_mb'], 4),
        'probe_peak_reserved_mb': round(probe_stats['probe_peak_reserved_mb'], 4),
        'selector_peak_alloc_mb': round(selector_peak_alloc_mb, 4),
        'selector_peak_reserved_mb': round(selector_peak_reserved_mb, 4),
        'score_file': score_file, 'gpu_name': gpu_name(),
    }, PROBE_COLUMNS)

    # Reload the hardened checkpoint and continue with the same training loader.
    del model, original_ae_scores
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    model, missing, unexpected = build_model(info)
    model = model.to(DEVICE)
    train_recovery(
        info, selector, model, missing, unexpected, selector_scores, num_scored_layers,
        train_loader, test_loader, ae_bank, probe_stats, selector_score_time,
        gradient_batches, selector_peak_alloc_mb, selector_peak_reserved_mb,
        score_file, n_train, n_test, logger,
    )
    del model, ae_bank, selector_scores
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()


def load_seed_rows():
    if not os.path.exists(FINAL_SEED_CSV):
        return []
    return pd.read_csv(FINAL_SEED_CSV).to_dict('records')


def done_keys():
    return {
        (str(r['selector']), str(r['checkpoint_name']), round(float(r['budget']), 6), int(r['seed']))
        for r in load_seed_rows()
    }


def remove_partial_rows(path, selector, checkpoint_name, budget, seed):
    # Resume safety: if a run was interrupted before its final seed row was
    # written, remove only that run's partial epoch/probe/budget bookkeeping.
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return
    required = {'selector', 'checkpoint_name', 'budget', 'seed'}
    if df.empty or not required.issubset(set(df.columns)):
        return
    same = (
        (df['selector'].astype(str) == str(selector))
        & (df['checkpoint_name'].astype(str) == str(checkpoint_name))
        & ((df['budget'].astype(float) - float(budget)).abs() < 1e-12)
        & (df['seed'].astype(int) == int(seed))
    )
    if same.any():
        df.loc[~same].to_csv(path, index=False)


def rebuild_summary():
    rows = load_seed_rows()
    groups = {}
    for r in rows:
        key = (str(r['selector']), str(r['checkpoint_name']), round(float(r['budget']), 6))
        groups.setdefault(key, []).append(r)
    out = []
    for (selector, name, budget), vals in groups.items():
        latest = {}
        for v in vals:
            seed = int(v['seed'])
            if seed in FINAL_SEEDS:
                latest[seed] = v
        if set(latest) != set(FINAL_SEEDS):
            continue
        vals = [latest[s] for s in FINAL_SEEDS]
        acc = np.array([float(v['epoch20_acc']) for v in vals], dtype=float)
        mean = float(acc.mean())
        sd = float(acc.std(ddof=1))
        half = CI_T_DF2 * sd / math.sqrt(len(FINAL_SEEDS))
        base = vals[0]
        def avg(key):
            return float(np.mean([float(v[key]) for v in vals]))
        out.append({
            'method': base['method'], 'checkpoint_name': base['checkpoint_name'],
            'dataset': base['dataset'], 'task': base['task'], 'arch': base['arch'],
            'checkpoint': base['checkpoint'], 'attack': 'autosparse_ablation', 'selector': selector,
            'budget': budget, 'budget_percent': 100.0 * budget,
            'lr': LR_STAGE1, 'head_lr': LR_STAGE1 * HEAD_LR_MULT,
            'n_seeds': len(FINAL_SEEDS), 'seeds': '|'.join(str(s) for s in FINAL_SEEDS),
            'mean_test_acc': mean, 'sample_sd': sd, 'ci95_low': mean - half, 'ci95_high': mean + half,
            'mean_ae_probe_time_sec': avg('ae_probe_time_sec'),
            'mean_selector_score_time_sec': avg('selector_score_time_sec'),
            'mean_selector_peak_alloc_mb': avg('selector_peak_alloc_mb'),
            'mean_selection_time_sec': avg('selection_time_sec'),
            'mean_train_time_sec': avg('train_time_sec'),
            'mean_total_attack_time_sec': avg('total_attack_time_sec'),
            'mean_peak_attack_alloc_mb': avg('peak_attack_alloc_mb'),
            'effective_trainable': int(float(base['effective_trainable'])),
            'total_params': int(float(base['total_params'])),
            'effective_percent': float(base['effective_percent']),
            'selected_conv_weight_params': int(float(base['selected_conv_weight_params'])),
            'trainable_conv_bias_params': int(float(base['trainable_conv_bias_params'])),
            'final_head_params': int(float(base['final_head_params'])),
            'selected_channels_total': int(float(base['selected_channels_total'])),
            'candidate_channels_total': int(float(base['candidate_channels_total'])),
            'num_conv_layers_scored': int(float(base['num_conv_layers_scored'])),
            'num_conv_layers_trained': int(float(base['num_conv_layers_trained'])),
            'gpu_name': base['gpu_name'],
        })
    columns = [
        'method','checkpoint_name','dataset','task','arch','checkpoint','attack','selector',
        'budget','budget_percent','lr','head_lr','n_seeds','seeds','mean_test_acc','sample_sd',
        'ci95_low','ci95_high','mean_ae_probe_time_sec','mean_selector_score_time_sec',
        'mean_selector_peak_alloc_mb','mean_selection_time_sec','mean_train_time_sec','mean_total_attack_time_sec',
        'mean_peak_attack_alloc_mb','effective_trainable','total_params','effective_percent',
        'selected_conv_weight_params','trainable_conv_bias_params','final_head_params',
        'selected_channels_total','candidate_channels_total','num_conv_layers_scored',
        'num_conv_layers_trained','gpu_name',
    ]
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(out, columns=columns).to_csv(FINAL_SUMMARY_CSV, index=False)
    build_comparison_summary()


def build_comparison_summary():
    # Automatically joins completed main AutoSparse means with ablation means.
    # Support both 'budget' and 'sparsity' column names.
    pieces = []
    if os.path.exists(MAIN_AUTOSPARSE_SUMMARY):
        main = pd.read_csv(MAIN_AUTOSPARSE_SUMMARY)
        if not main.empty:
            if 'budget' in main.columns:
                budget_values = main['budget'].astype(float)
            elif 'sparsity' in main.columns:
                budget_values = main['sparsity'].astype(float)
            else:
                budget_values = None
            required = {'checkpoint_name','mean_test_acc','sample_sd','ci95_low','ci95_high'}
            if budget_values is not None and required.issubset(set(main.columns)):
                keep = pd.DataFrame({
                    'checkpoint_name': main['checkpoint_name'].astype(str),
                    'selector': 'ae_reconstruction',
                    'budget': budget_values,
                    'budget_percent': budget_values * 100.0,
                    'mean_test_acc': main['mean_test_acc'].astype(float),
                    'sample_sd': main['sample_sd'].astype(float),
                    'ci95_low': main['ci95_low'].astype(float),
                    'ci95_high': main['ci95_high'].astype(float),
                })
                pieces.append(keep)
    if os.path.exists(FINAL_SUMMARY_CSV):
        abl = pd.read_csv(FINAL_SUMMARY_CSV)
        if not abl.empty:
            pieces.append(abl[[
                'checkpoint_name','selector','budget','budget_percent',
                'mean_test_acc','sample_sd','ci95_low','ci95_high'
            ]])
    if pieces:
        comp = pd.concat(pieces, ignore_index=True)
        order = {'ae_reconstruction':0, 'random':1, 'magnitude':2, 'gradient':3}
        comp['_selector_order'] = comp['selector'].map(order).fillna(99)
        comp = comp.sort_values(['checkpoint_name','budget','_selector_order']).drop(columns=['_selector_order'])
        comp.to_csv(COMPARISON_SUMMARY_CSV, index=False)


def setup_logger():
    os.makedirs(OUT_DIR, exist_ok=True)
    logger = logging.getLogger('autosparse_ablation')
    logger.handlers = []
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(message)s')
    fh = logging.FileHandler(LOG_PATH)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def confirm_paths(infos):
    print(f"Checking {len(infos)} checkpoints...")
    for info in infos:
        if not os.path.exists(info['path']):
            raise FileNotFoundError(info['path'])
        print("Checkpoint:", info["path"])
    print("All selected checkpoints found.")


def select_infos(only):
    if only == 'all':
        return CHECKPOINTS
    selected = [x for x in CHECKPOINTS if x['name'] == only]
    if not selected:
        valid = ', '.join(x['name'] for x in CHECKPOINTS)
        raise ValueError(f'Unknown checkpoint {only}. Valid: {valid}')
    return selected


def clear_outputs():
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)


def parse_budgets(values):
    budgets = []
    for value in values:
        value = float(value)
        if value <= 0.0 or value > 100.0:
            raise ValueError('Budgets must be percentages in (0,100].')
        budgets.append(value / 100.0)
    return budgets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', default='all', help='all or one checkpoint_name')
    parser.add_argument('--selectors', nargs='+', default=DEFAULT_SELECTORS, choices=VALID_SELECTORS)
    parser.add_argument('--budgets', nargs='+', type=float, default=[1.0, 5.0, 10.0, 15.0],
                        help='Channel budgets in percent, e.g. 1 5 10 15')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    budgets = parse_budgets(args.budgets)
    selectors = list(dict.fromkeys(args.selectors))
    if args.overwrite:
        clear_outputs()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(SCORE_DIR, exist_ok=True)
    infos = select_infos(args.only)
    confirm_paths(infos)
    logger = setup_logger()

    logger.info('=' * 100)
    logger.info('AUTOSPARSE SELECTOR ABLATION')
    logger.info('AE-selected recovery is NOT repeated here; existing results_clean/autosparse is reused in comparison_with_autosparse.csv')
    logger.info('Selectors: %s', selectors)
    logger.info('Budgets: %s', [int(b * 100) for b in budgets])
    logger.info('Seeds: %s', FINAL_SEEDS)
    logger.info('Selection unit: convolutional output channel')
    logger.info('Only selection ranking changes: random vs magnitude vs gradient')
    logger.info('AE recovery preserved: epochs=%d lr=%g max_batches=%d score_batches=%d residual_lambda=%g',
                AE_EPOCHS, AE_LR, AE_MAX_BATCHES, SCORE_BATCHES, AE_RESIDUAL_LAMBDA)
    logger.info('Recovery preserved: AdamW backbone_lr=%g head_lr=%g wd=%g batch=%d epochs=%d cosine_eta_min=%g beta=%g',
                LR_STAGE1, LR_STAGE1 * HEAD_LR_MULT, WEIGHT_DECAY, BATCH_SIZE, EPOCHS,
                LR_STAGE1 * 0.05, SAMPLE_BETA)
    logger.info('Each selector x checkpoint x budget x seed is independent')
    logger.info('Epoch 20 is primary; predictions, mean, sample SD and 95%% CI are saved')
    logger.info('GPU: %s', gpu_name())
    logger.info('=' * 100)

    done = done_keys()
    for info in infos:
        for selector in selectors:
            for budget in budgets:
                for seed in FINAL_SEEDS:
                    key = (selector, info['name'], round(float(budget), 6), seed)
                    if key in done:
                        logger.info('SKIP | %s | selector=%s | budget=%.0f%% | seed=%d',
                                    info['name'], selector, budget * 100.0, seed)
                        continue
                    logger.info('RUN | %s | selector=%s | budget=%.0f%% | seed=%d',
                                info['name'], selector, budget * 100.0, seed)
                    remove_partial_rows(FINAL_EPOCH_CSV, selector, info['name'], budget, seed)
                    remove_partial_rows(PROBE_CSV, selector, info['name'], budget, seed)
                    remove_partial_rows(BUDGET_DETAILS_CSV, selector, info['name'], budget, seed)
                    run_one(info, selector, seed, budget, logger)
                    done.add(key)
                    rebuild_summary()

    rebuild_summary()
    logger.info('FINAL MEAN ± STD (95%% CI)')
    if os.path.exists(FINAL_SUMMARY_CSV):
        summary = pd.read_csv(FINAL_SUMMARY_CSV)
        for _, row in summary.iterrows():
            logger.info(
                '%s | selector=%s | budget=%.0f%% | %.4f ± %.4f | CI=[%.4f, %.4f] | effective=%.6f%% | AEprobe=%.2fs | selector_score=%.2fs | train=%.2fs | attack=%.2fs | peak=%.1fMB',
                row['checkpoint_name'], row['selector'], float(row['budget']) * 100.0,
                row['mean_test_acc'], row['sample_sd'], row['ci95_low'], row['ci95_high'],
                row['effective_percent'], row['mean_ae_probe_time_sec'],
                row['mean_selector_score_time_sec'], row['mean_train_time_sec'],
                row['mean_total_attack_time_sec'], row['mean_peak_attack_alloc_mb'],
            )

    print("Done.")
    print('Epoch results     :', FINAL_EPOCH_CSV)
    print('Seed results      :', FINAL_SEED_CSV)
    print('Ablation summary  :', FINAL_SUMMARY_CSV)
    print('Combined summary  :', COMPARISON_SUMMARY_CSV)
    print('Probe results     :', PROBE_CSV)
    print('Budget details    :', BUDGET_DETAILS_CSV)
    print('Selector scores   :', SCORE_DIR)
    print('Predictions       :', PRED_DIR)
    print('Log               :', LOG_PATH)


if __name__ == '__main__':
    main()
