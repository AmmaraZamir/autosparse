import os
import csv
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
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

DATA_ROOT = "./data"
OUT_DIR = "./results_clean/dora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANKS = [4, 8, 16, 32, 64]
EPOCHS = 20
BATCH_SIZE = 64
WORKERS = 4
WEIGHT_DECAY = 1e-4
LEARNING_RATES = [1e-3]
SWEEP_SEED = 41
FINAL_SEEDS = [42, 43, 44]
SPLIT_SEED = 2026
VAL_FRACTION = 0.10
NUM_CLASSES = 10

DORA_ALPHA_SOPHON = 32
DORA_ALPHA_VGG = 16

ATTACK = "dora"

T95_DF2 = 4.302652729911275
RE_LABEL = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]

CHECKPOINTS = [
    {"method":"SOPHON-IL","name":"sophon_il_cifar10","type":"sophon","arch":"resnet18","dataset":"CIFAR10","task":"CIFAR10","path":"./checkpoints/sophon_il/sophon_il_cifar10.pth"},
    {"method":"SOPHON-IL","name":"sophon_il_cinic","type":"sophon","arch":"resnet18","dataset":"CINIC","task":"CINIC","path":"./checkpoints/sophon_il/sophon_il_cinic.pth"},
    {"method":"SOPHON-IL","name":"sophon_il_mnist","type":"sophon","arch":"resnet18","dataset":"MNIST","task":"MNIST","path":"./checkpoints/sophon_il/sophon_il_mnist.pth"},
    {"method":"SOPHON-IL","name":"sophon_il_stl10","type":"sophon","arch":"resnet18","dataset":"STL10","task":"STL10","path":"./checkpoints/sophon_il/sophon_il_stl10.pth"},
    {"method":"SOPHON-IL","name":"sophon_il_svhn","type":"sophon","arch":"resnet18","dataset":"SVHN","task":"SVHN","path":"./checkpoints/sophon_il/sophon_il_svhn.pth"},
    {"method":"SOPHON-KL","name":"sophon_kl_cifar10","type":"sophon","arch":"resnet18","dataset":"CIFAR10","task":"CIFAR10","path":"./checkpoints/sophon_kl/sophon_kl_cifar10.pth"},
    {"method":"SOPHON-KL","name":"sophon_kl_cinic","type":"sophon","arch":"resnet18","dataset":"CINIC","task":"CINIC","path":"./checkpoints/sophon_kl/sophon_kl_cinic.pth"},
    {"method":"SOPHON-KL","name":"sophon_kl_mnist","type":"sophon","arch":"resnet18","dataset":"MNIST","task":"MNIST","path":"./checkpoints/sophon_kl/sophon_kl_mnist.pth"},
    {"method":"SOPHON-KL","name":"sophon_kl_stl10","type":"sophon","arch":"resnet18","dataset":"STL10","task":"STL10","path":"./checkpoints/sophon_kl/sophon_kl_stl10.pth"},
    {"method":"SOPHON-KL","name":"sophon_kl_svhn","type":"sophon","arch":"resnet18","dataset":"SVHN","task":"SVHN","path":"./checkpoints/sophon_kl/sophon_kl_svhn.pth"},
    {"method":"HNTL","name":"hntl_cifar_stl_vgg13","type":"vgg","arch":"vgg13","dataset":"STL10","task":"CIFAR10_to_STL10","path":"./checkpoints/hntl_cifar_stl_vgg13.pth"},
    {"method":"CUPI","name":"cupi_cifar_stl_vgg13","type":"vgg","arch":"vgg13","dataset":"STL10","task":"CIFAR10_to_STL10","path":"./checkpoints/cupi_cifar_stl_vgg13.pth"},
    {"method":"CUTI","name":"cuti_cifar_stl_vgg19","type":"vgg","arch":"vgg19","dataset":"STL10","task":"CIFAR10_to_STL10","path":"./checkpoints/cuti_cifar_stl_vgg19.pth"},
]


class STL10Mapped(datasets.STL10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, RE_LABEL[target]


class CustomVGG(nn.Module):
    def __init__(self, arch, num_classes=10):
        super().__init__()
        if arch == "vgg13":
            base = models.vgg13(weights=None)
        elif arch == "vgg19":
            base = models.vgg19(weights=None)
        else:
            raise ValueError(arch)
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


class DoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.scale = alpha / rank

        for p in self.linear.parameters():
            p.requires_grad = False

        out_f, in_f = linear.weight.shape
        self.A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, rank))

        with torch.no_grad():
            mag = linear.weight.norm(dim=1)
        self.magnitude = nn.Parameter(mag.clone())

    def forward(self, x):
        delta = self.B @ self.A
        adapted = self.linear.weight + self.scale * delta
        direction = adapted / (adapted.norm(dim=1, keepdim=True) + 1e-6)
        weight = self.magnitude.unsqueeze(1) * direction
        return F.linear(x, weight, self.linear.bias)


class DoRAConv2d(nn.Module):
    def __init__(self, conv, rank, alpha):
        super().__init__()
        self.conv = conv
        self.scale = alpha / rank

        for p in self.conv.parameters():
            p.requires_grad = False

        out_ch = conv.out_channels
        kh, kw = conv.kernel_size
        flat_dim = conv.in_channels * kh * kw

        self.A = nn.Parameter(torch.randn(rank, flat_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_ch, rank))

        with torch.no_grad():
            mag = conv.weight.view(out_ch, -1).norm(dim=1)
        self.magnitude = nn.Parameter(mag.clone())

    def forward(self, x):
        out_ch = self.conv.out_channels
        base = self.conv.weight.view(out_ch, -1)
        delta = self.B @ self.A
        adapted = base + self.scale * delta
        direction = adapted / (adapted.norm(dim=1, keepdim=True) + 1e-6)
        weight_flat = self.magnitude.unsqueeze(1) * direction
        weight = weight_flat.view_as(self.conv.weight)

        return F.conv2d(
            x,
            weight,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def strip_module(state):
    out = {}
    for k, v in state.items():
        while k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out


def load_sophon_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    return strip_module(state)


def load_vgg_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    state = strip_module(state)
    fixed = {}
    for k, v in state.items():
        if k.startswith("classifier1."):
            k = k.replace("classifier1.", "classifier.")
        fixed[k] = v
    return fixed


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def replace_all_dora(module, rank, alpha, prefix=""):
    count = 0
    names = []
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d):
            setattr(module, name, DoRAConv2d(child, rank, alpha))
            count += 1
            names.append(full)
        elif isinstance(child, nn.Linear):
            setattr(module, name, DoRALinear(child, rank, alpha))
            count += 1
            names.append(full)
        else:
            c, n = replace_all_dora(child, rank, alpha, full)
            count += c
            names.extend(n)
    return count, names


def build_model(info, rank, seed):

    if info["type"] == "sophon":
        model = build_resnet18()
        state = load_sophon_state(info["path"])
        alpha = DORA_ALPHA_SOPHON
    elif info["type"] == "vgg":
        model = CustomVGG(info["arch"], NUM_CLASSES)
        state = load_vgg_state(info["path"])
        alpha = DORA_ALPHA_VGG
    else:
        raise ValueError(info["type"])

    missing, unexpected = model.load_state_dict(state, strict=False)
    freeze_all(model)

    n_dora, layers = replace_all_dora(model, rank, alpha)
    return model, list(missing), list(unexpected), n_dora, layers, alpha


def get_dataset_pair(dataset):
    tf_rgb = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    tf_mnist = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    d = dataset.upper()
    if d == "CIFAR10":
        train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=tf_rgb)
        test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=tf_rgb)
    elif d == "MNIST":
        train_set = datasets.MNIST(DATA_ROOT, train=True, download=False, transform=tf_mnist)
        test_set = datasets.MNIST(DATA_ROOT, train=False, download=False, transform=tf_mnist)
    elif d == "SVHN":
        train_set = datasets.SVHN(DATA_ROOT, split="train", download=False, transform=tf_rgb)
        test_set = datasets.SVHN(DATA_ROOT, split="test", download=False, transform=tf_rgb)
    elif d == "STL10":
        train_set = STL10Mapped(DATA_ROOT, split="train", download=False, transform=tf_rgb)
        test_set = STL10Mapped(DATA_ROOT, split="test", download=False, transform=tf_rgb)
    elif d == "CINIC":
        cinic_root = os.path.join(DATA_ROOT, "CINIC10")
        train_dir = os.path.join(cinic_root, "train")
        valid_dir = os.path.join(cinic_root, "valid")
        test_dir = os.path.join(cinic_root, "test")
        if os.path.exists(train_dir) and os.path.exists(valid_dir):
            train_set = datasets.ImageFolder(train_dir, transform=tf_rgb)
            test_set = datasets.ImageFolder(valid_dir, transform=tf_rgb)
        elif os.path.exists(train_dir) and os.path.exists(test_dir):
            train_set = datasets.ImageFolder(train_dir, transform=tf_rgb)
            test_set = datasets.ImageFolder(test_dir, transform=tf_rgb)
        else:
            raise FileNotFoundError(f"CINIC folders not found inside {cinic_root}")
    else:
        raise ValueError(dataset)
    return train_set, test_set


def make_train_loader(dataset, seed):
     return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True,
        drop_last=True,
     )


def make_eval_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
    )


def fixed_split(train_set):
    n_total = len(train_set)
    n_val = int(round(n_total * VAL_FRACTION))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(SPLIT_SEED)
    return random_split(train_set, [n_train, n_val], generator=g)


def param_stats(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    base_model_params = total - trainable
    pct_total = 100.0 * trainable / total
    pct_base = 100.0 * trainable / base_model_params if base_model_params else 0.0
    return trainable, total, base_model_params, pct_total, pct_base


def gpu_name():
    return torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"


def reset_peak_memory():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_memory_mb():
    if DEVICE != "cuda":
        return 0.0, 0.0
    torch.cuda.synchronize()
    alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserv = torch.cuda.max_memory_reserved() / (1024 ** 2)
    return alloc, reserv


@torch.no_grad()
def evaluate(model, loader, collect_predictions=False):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    loss_sum = 0.0
    y_true = []
    y_pred = []

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y_dev = y.to(DEVICE, non_blocking=True)
        out = model(x)
        pred = out.argmax(1)
        loss_sum += ce(out, y_dev).item()
        correct += (pred == y_dev).sum().item()
        total += y_dev.size(0)
        if collect_predictions:
            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    acc = 100.0 * correct / total
    loss = loss_sum / total
    return acc, loss, elapsed, y_true, y_pred


def train_epochs(model, train_loader, lr, logger, prefix, epoch_csv=None, epoch_context=None, test_loader=None):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    optimizer_steps = 0
    samples_seen = 0
    total_train_time = 0.0
    max_train_alloc = 0.0
    max_train_reserv = 0.0
    reset_peak_memory()

    for epoch in range(1, EPOCHS + 1):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in tqdm(train_loader, desc=f"{prefix} E{epoch}", leave=False):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = ce(out, y)
            loss.backward()
            optimizer.step()
            bs = y.size(0)
            running_loss += loss.item() * bs
            seen += bs
            optimizer_steps += 1
            samples_seen += bs

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start
        total_train_time += epoch_time
        alloc, reserv = peak_memory_mb()
        max_train_alloc = max(max_train_alloc, alloc)
        max_train_reserv = max(max_train_reserv, reserv)
        epoch_loss = running_loss / max(seen, 1)

        epoch_test_acc = None
        epoch_test_loss = None
        epoch_test_time = 0.0

        if test_loader is not None:
            epoch_test_acc, epoch_test_loss, epoch_test_time, _, _ = evaluate(
                model, test_loader, False
            )
            reset_peak_memory()

        logger.info(
            f"{prefix} | epoch={epoch:02d}/{EPOCHS} | loss={epoch_loss:.6f} | "
            f"test_acc={epoch_test_acc if epoch_test_acc is not None else 'NA'} | "
            f"train={epoch_time:.2f}s | peak_alloc={alloc:.1f}MB | peak_reserved={reserv:.1f}MB"
        )

        if epoch_csv is not None:
            row = dict(epoch_context or {})
            row.update({
                "epoch": epoch,
                "epoch_loss": epoch_loss,
                "epoch_train_time_sec": epoch_time,
                "cumulative_train_time_sec": total_train_time,
                "epoch_test_acc": epoch_test_acc,
                "epoch_test_loss": epoch_test_loss,
                "epoch_test_time_sec": epoch_test_time,
                "peak_allocated_mb": alloc,
                "peak_reserved_mb": reserv,
                "optimizer_steps_so_far": optimizer_steps,
                "samples_seen_so_far": samples_seen,
            })
            append_dict_row(epoch_csv, row)

    return total_train_time, max_train_alloc, max_train_reserv, optimizer_steps, samples_seen


def run_sweep(info, attack, rank, lr, train_subset, val_subset, logger, paths):
    model, missing, unexpected, n_dora, layers, alpha = build_model(info, rank, SWEEP_SEED)
    model = model.to(DEVICE)
    trainable, total, base_model_params, pct, pct_base = param_stats(model)
    train_loader = make_train_loader(train_subset, SWEEP_SEED)
    val_loader = make_eval_loader(val_subset)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    ce = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_epoch = -1
    epoch20_acc = None
    total_train_time = 0.0
    total_val_time = 0.0
    optimizer_steps = 0
    samples_seen = 0
    reset_peak_memory()

    logger.info(
        f"SWEEP | {info['name']} | {attack} | r={rank} | lr={lr} | "
        f"train={len(train_subset)} | val={len(val_subset)} | "
        f"trainable={trainable}/{total} ({pct:.6f}%) | dora_layers={n_dora}"
    )

    for epoch in range(1, EPOCHS + 1):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in tqdm(train_loader, desc=f"SWEEP {info['name']} {attack} R{rank} lr={lr} E{epoch}", leave=False):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = ce(out, y)
            loss.backward()
            optimizer.step()
            bs = y.size(0)
            running_loss += loss.item() * bs
            seen += bs
            optimizer_steps += 1
            samples_seen += bs

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        train_time = time.perf_counter() - start
        total_train_time += train_time

        val_acc, val_loss, val_time, _, _ = evaluate(model, val_loader, False)
        total_val_time += val_time
        alloc, reserv = peak_memory_mb()
        epoch_loss = running_loss / max(seen, 1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
        if epoch == EPOCHS:
            epoch20_acc = val_acc

        append_dict_row(paths["sweep_epoch"], {
            "method": info["method"],
            "checkpoint_name": info["name"],
            "dataset": info["dataset"],
            "task": info["task"],
            "arch": info["arch"],
            "attack": attack,
            "rank": rank,
            "dora_alpha": alpha,
            "lr": lr,
            "seed": SWEEP_SEED,
            "split_seed": SPLIT_SEED,
            "epoch": epoch,
            "epoch_loss": epoch_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc_so_far": best_acc,
            "best_epoch_so_far": best_epoch,
            "epoch_train_time_sec": train_time,
            "val_time_sec": val_time,
            "cumulative_train_time_sec": total_train_time,
            "cumulative_val_time_sec": total_val_time,
            "peak_allocated_mb": alloc,
            "peak_reserved_mb": reserv,
            "trainable_params": trainable,
            "total_params": total,
            "base_model_params": base_model_params,
            "trainable_percent": pct,
            "trainable_percent_of_base": pct_base,
            "num_dora_layers": n_dora,
            "targeted_layers": " | ".join(layers),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "optimizer_steps_so_far": optimizer_steps,
            "samples_seen_so_far": samples_seen,
            "gpu": gpu_name(),
        })

        logger.info(
            f"SWEEP | {info['name']} | {attack} | r={rank} | lr={lr} | epoch={epoch:02d}/{EPOCHS} | "
            f"val={val_acc:.4f} | best={best_acc:.4f}@{best_epoch} | train={train_time:.2f}s | "
            f"valtime={val_time:.2f}s | peak_alloc={alloc:.1f}MB | peak_reserved={reserv:.1f}MB"
        )

    append_dict_row(paths["sweep_summary"], {
        "method": info["method"],
        "checkpoint_name": info["name"],
        "dataset": info["dataset"],
        "task": info["task"],
        "arch": info["arch"],
        "attack": attack,
        "rank": rank,
        "dora_alpha": alpha,
        "lr": lr,
        "seed": SWEEP_SEED,
        "split_seed": SPLIT_SEED,
        "train_size": len(train_subset),
        "val_size": len(val_subset),
        "epoch20_val_acc": epoch20_acc,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "train_time_sec": total_train_time,
        "val_time_sec": total_val_time,
        "sweep_total_time_sec": total_train_time + total_val_time,
        "peak_allocated_mb": alloc,
        "peak_reserved_mb": reserv,
        "trainable_params": trainable,
        "total_params": total,
        "base_model_params": base_model_params,
        "trainable_percent": pct,
        "trainable_percent_of_base": pct_base,
        "num_dora_layers": n_dora,
        "targeted_layers": " | ".join(layers),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "optimizer_steps": optimizer_steps,
        "samples_seen": samples_seen,
        "gpu": gpu_name(),
    })

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return epoch20_acc


def run_final(info, attack, rank, lr, seed, full_train_set, test_set, logger, paths):
    model, missing, unexpected, n_dora, layers, alpha = build_model(info, rank, seed)
    model = model.to(DEVICE)
    trainable, total, base_model_params, pct, pct_base = param_stats(model)
    train_loader = make_train_loader(full_train_set, seed)
    test_loader = make_eval_loader(test_set)

    prefix = f"FINAL | {info['name']} | {attack} | r={rank} | seed={seed} | lr={lr}"
    logger.info(
        f"{prefix} | full_train={len(full_train_set)} | "
        f"trainable={trainable}/{total} ({pct:.6f}%) | dora_layers={n_dora}"
    )

    train_time, alloc, reserv, optimizer_steps, samples_seen = train_epochs(
        model,
        train_loader,
        lr,
        logger,
        prefix,
        epoch_csv=paths["final_epoch"],
        epoch_context={
            "method": info["method"],
            "checkpoint_name": info["name"],
            "dataset": info["dataset"],
            "task": info["task"],
            "arch": info["arch"],
            "attack": attack,
            "rank": rank,
            "dora_alpha": alpha,
            "lr": lr,
            "seed": seed,
        },
        test_loader=test_loader,
    )

    test_acc, test_loss, test_time, y_true, y_pred = evaluate(model, test_loader, True)

    pred_name = f"{info['name']}__{attack}__r{rank}__seed{seed}.csv"
    pred_path = os.path.join(paths["pred_dir"], pred_name)
    pd.DataFrame({
        "sample_index": np.arange(len(y_true), dtype=np.int64),
        "y_true": y_true,
        "y_pred": y_pred,
    }).to_csv(pred_path, index=False)

    append_dict_row(paths["final_seed"], {
        "method": info["method"],
        "checkpoint_name": info["name"],
        "dataset": info["dataset"],
        "task": info["task"],
        "arch": info["arch"],
        "attack": attack,
        "rank": rank,
        "dora_alpha": alpha,
        "lr": lr,
        "seed": seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train_time_sec": train_time,
        "test_time_sec": test_time,
        "total_attack_time_sec": train_time,
        "probing_time_sec": 0.0,
        "intrinsic_selection_time_sec": 0.0,
        "peak_allocated_mb": alloc,
        "peak_reserved_mb": reserv,
        "trainable_params": trainable,
        "total_params": total,
        "base_model_params": base_model_params,
        "trainable_percent": pct,
        "trainable_percent_of_base": pct_base,
        "num_dora_layers": n_dora,
        "targeted_layers": " | ".join(layers),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "optimizer_steps": optimizer_steps,
        "samples_seen": samples_seen,
        "gpu": gpu_name(),
        "predictions_file": pred_path,
    })

    logger.info(
        f"RESULT | {info['name']} | {attack} | r={rank} | seed={seed} | lr={lr} | "
        f"test={test_acc:.4f}% | train={train_time:.2f}s | testtime={test_time:.2f}s | "
        f"peak_alloc={alloc:.1f}MB | budget={pct:.6f}% | pred={pred_path}"
    )

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        "test_acc": test_acc,
        "train_time_sec": train_time,
        "test_time_sec": test_time,
        "total_attack_time_sec": train_time,
        "probing_time_sec": 0.0,
        "intrinsic_selection_time_sec": 0.0,
        "peak_allocated_mb": alloc,
        "peak_reserved_mb": reserv,
        "trainable_params": trainable,
        "total_params": total,
        "base_model_params": base_model_params,
        "trainable_percent": pct,
        "trainable_percent_of_base": pct_base,
        "num_dora_layers": n_dora,
        "alpha": alpha,
    }


def setup_logger(path):
    logger = logging.getLogger("dora")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def append_dict_row(path, row):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_done(path, keys):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set(tuple(row[k] for k in keys) for _, row in df.iterrows())


def paths_for_run():
    pred_dir = os.path.join(OUT_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    return {
        "log": os.path.join(OUT_DIR, "dora.log"),
        "sweep_epoch": os.path.join(OUT_DIR, "lr_sweep_epoch_results.csv"),
        "sweep_summary": os.path.join(OUT_DIR, "lr_sweep_summary.csv"),
        "chosen": os.path.join(OUT_DIR, "chosen_lr.csv"),
        "final_epoch": os.path.join(OUT_DIR, "final_epoch_results.csv"),
        "final_seed": os.path.join(OUT_DIR, "final_seed_results.csv"),
        "final_summary": os.path.join(OUT_DIR, "final_summary.csv"),
        "pred_dir": pred_dir,
    }


def confirm_paths():
    print("Checking checkpoints...")
    for info in CHECKPOINTS:
        if not os.path.exists(info["path"]):
            raise FileNotFoundError(info["path"])
        print("Checkpoint:", info["path"])
    print("All checkpoints found.")


def choose_lr_from_summary(paths, info, attack, rank):
    df = pd.read_csv(paths["sweep_summary"])
    sub = df[
        (df["checkpoint_name"] == info["name"])
        & (df["attack"] == attack)
        & (df["rank"] == rank)
    ].copy()
    if len(sub) != len(LEARNING_RATES):
        return None
    sub = sub.sort_values(["epoch20_val_acc", "lr"], ascending=[False, True])
    return float(sub.iloc[0]["lr"])


def write_chosen(paths, info, attack, rank, lr):
    df = pd.read_csv(paths["sweep_summary"])
    sub = df[
        (df["checkpoint_name"] == info["name"])
        & (df["attack"] == attack)
        & (df["rank"] == rank)
    ].copy()
    chosen = sub[np.isclose(sub["lr"].astype(float), lr)].iloc[0]
    append_dict_row(paths["chosen"], {
        "method": info["method"],
        "checkpoint_name": info["name"],
        "dataset": info["dataset"],
        "task": info["task"],
        "arch": info["arch"],
        "attack": attack,
        "rank": rank,
        "chosen_lr": lr,
        "selection_metric": "epoch20_val_acc",
        "selected_epoch20_val_acc": float(chosen["epoch20_val_acc"]),
        "selected_best_val_acc": float(chosen["best_val_acc"]),
        "selected_best_epoch": int(chosen["best_epoch"]),
        "sweep_seed": SWEEP_SEED,
        "split_seed": SPLIT_SEED,
        "lr_candidates": " | ".join(str(x) for x in LEARNING_RATES),
        "all_lr_sweep_time_sec": float(sub["sweep_total_time_sec"].sum()),
        "gpu": gpu_name(),
    })


def already_chosen(paths, info, attack, rank):
    if not os.path.exists(paths["chosen"]):
        return False
    df = pd.read_csv(paths["chosen"])
    return bool(((df["checkpoint_name"] == info["name"]) & (df["attack"] == attack) & (df["rank"] == rank)).any())


def already_final_seed(paths, info, attack, rank, seed):
    if not os.path.exists(paths["final_seed"]):
        return False
    df = pd.read_csv(paths["final_seed"])
    return bool(((df["checkpoint_name"] == info["name"]) & (df["attack"] == attack) & (df["rank"] == rank) & (df["seed"] == seed)).any())


def summarize_config(paths, info, attack, rank, lr, logger):
    df = pd.read_csv(paths["final_seed"])
    sub = df[
        (df["checkpoint_name"] == info["name"])
        & (df["attack"] == attack)
        & (df["rank"] == rank)
    ].copy()
    if len(sub) < len(FINAL_SEEDS):
        return

    sub = sub[sub["seed"].isin(FINAL_SEEDS)].drop_duplicates(subset=["seed"], keep="last")
    if len(sub) != len(FINAL_SEEDS):
        return

    acc = sub["test_acc"].astype(float).to_numpy()
    mean = float(np.mean(acc))
    sd = float(np.std(acc, ddof=1))
    half = T95_DF2 * sd / math.sqrt(len(acc))
    ci_low = mean - half
    ci_high = mean + half

    row = {
        "method": info["method"],
        "checkpoint_name": info["name"],
        "dataset": info["dataset"],
        "task": info["task"],
        "arch": info["arch"],
        "attack": attack,
        "rank": rank,
        "dora_alpha": int(sub.iloc[0]["dora_alpha"]),
        "chosen_lr": lr,
        "n_seeds": len(FINAL_SEEDS),
        "seeds": " | ".join(str(s) for s in FINAL_SEEDS),
        "mean_test_acc": mean,
        "sample_sd": sd,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "mean_train_time_sec": float(sub["train_time_sec"].mean()),
        "mean_test_time_sec": float(sub["test_time_sec"].mean()),
        "mean_total_attack_time_sec": float(sub["total_attack_time_sec"].mean()),
        "mean_peak_allocated_mb": float(sub["peak_allocated_mb"].mean()),
        "mean_peak_reserved_mb": float(sub["peak_reserved_mb"].mean()),
        "trainable_params": int(sub.iloc[0]["trainable_params"]),
        "total_params": int(sub.iloc[0]["total_params"]),
        "base_model_params": int(sub.iloc[0]["base_model_params"]),
        "trainable_percent": float(sub.iloc[0]["trainable_percent"]),
        "trainable_percent_of_base": float(sub.iloc[0]["trainable_percent_of_base"]),
        "num_dora_layers": int(sub.iloc[0]["num_dora_layers"]),
        "gpu": str(sub.iloc[0]["gpu"]),
    }

    if os.path.exists(paths["final_summary"]):
        old = pd.read_csv(paths["final_summary"])
        mask = (
            (old["checkpoint_name"] == info["name"])
            & (old["attack"] == attack)
            & (old["rank"] == rank)
        )
        old = old[~mask]
        old = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        old.to_csv(paths["final_summary"], index=False)
    else:
        pd.DataFrame([row]).to_csv(paths["final_summary"], index=False)

    logger.info(
        f"SUMMARY | {info['name']} | {attack} | r={rank} | "
        f"{mean:.4f} ± {sd:.4f} (95% CI {ci_low:.4f}–{ci_high:.4f}) | "
        f"LR={lr} | time={row['mean_train_time_sec']:.2f}s | "
        f"peak={row['mean_peak_allocated_mb']:.1f}MB | budget={row['trainable_percent']:.6f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite and os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = paths_for_run()
    logger = setup_logger(paths["log"])
    confirm_paths()

    logger.info("DoRA evaluation")
    logger.info("Attack: backbone-wide DoRA")
    logger.info(f"Ranks: {RANKS}")
    logger.info("Fixed LR: 0.001")
    logger.info("Seeds: %s", FINAL_SEEDS)
    logger.info(
        f"AdamW | weight_decay={WEIGHT_DECAY} | batch={BATCH_SIZE} | "
        f"epochs={EPOCHS} | constant_lr"
    )
    logger.info("Full target training set with fixed learning rate")
    logger.info("Predictions are saved for each seed")

    logger.info("Intrinsic preparation/probing/selection time: 0 for DoRA")

    logger.info(f"GPU: {gpu_name()}")

    dataset_cache = {}
    attack = ATTACK
    fixed_lr = 1e-3

    for seed in FINAL_SEEDS:
        set_seed(seed)
        for info in CHECKPOINTS:
            if args.only != "all" and args.only != info["name"]:
                continue

            if info["dataset"] not in dataset_cache:
                full_train, test_set = get_dataset_pair(info["dataset"])
                dataset_cache[info["dataset"]] = (full_train, test_set)

            full_train, test_set = dataset_cache[info["dataset"]]

            for rank in RANKS:
                if already_final_seed(paths, info, attack, rank, seed):
                    logger.info(
                        f"SKIP FINAL existing | {info['name']} | {attack} | "
                        f"r={rank} | seed={seed}"
                    )
                    continue

                run_final(
                    info,
                    attack,
                    rank,
                    fixed_lr,
                    seed,
                    full_train,
                    test_set,
                    logger,
                    paths,
                )

# Summaries are generated only after all three complete seed runs.
    for info in CHECKPOINTS:
        if args.only != "all" and args.only != info["name"]:
            continue
        for rank in RANKS:
            summarize_config(
                paths,
                info,
                attack,
                rank,
                fixed_lr,
                logger,
            )

    if os.path.exists(paths["final_summary"]):
        print("Final mean ± std (95% CI):")
        sdf = pd.read_csv(paths["final_summary"])
        for _, r in sdf.iterrows():
            print(
                f"{r['checkpoint_name']} | DoRA | "
                f"r={int(r['rank'])}: "
                f"{r['mean_test_acc']:.4f} ± {r['sample_sd']:.4f} "
                f"(95% CI {r['ci95_low']:.4f}–{r['ci95_high']:.4f}) | "
                f"LR=0.001 | "
                f"time={r['mean_train_time_sec']:.2f}s | "
                f"peak={r['mean_peak_allocated_mb']:.1f}MB | "
                f"budget={r['trainable_percent']:.6f}%"
            )

    print("Outputs:")
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
