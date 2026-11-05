import os
import csv
import gc
import math
import time
import random
import argparse
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

DATA_ROOT = "./data"
OUT_DIR = "./results_clean/full_ft"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
EPOCHS = 20
BATCH_SIZE = 64
WORKERS = 4
WEIGHT_DECAY = 1e-4
LEARNING_RATES = [1e-3]
FINAL_SEEDS = [42, 43, 44]
SWEEP_SEED = 41
SPLIT_SEED = 2026
VAL_FRACTION = 0.10
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

SWEEP_EPOCH_COLUMNS = [
    "method","checkpoint_name","dataset","task","arch","checkpoint",
    "phase","seed","lr_initial","lr_current","epoch","epochs",
    "train_loss","val_acc","val_loss",
    "epoch_train_time_sec","epoch_val_time_sec","cumulative_time_sec",
    "trainable_params","total_params","trainable_percent",
    "peak_allocated_mb","peak_reserved_mb","optimizer_steps_so_far",
    "samples_seen_so_far","probe_time_sec","missing_keys","unexpected_keys",
    "gpu_name"
]

SWEEP_SUMMARY_COLUMNS = [
    "method","checkpoint_name","dataset","task","arch","checkpoint",
    "seed","lr","final_val_acc","final_val_loss","best_val_acc","best_val_epoch",
    "train_time_sec","validation_time_sec","total_time_sec","probe_time_sec",
    "peak_allocated_mb","peak_reserved_mb",
    "trainable_params","total_params","trainable_percent",
    "optimizer_steps","samples_seen","train_examples","val_examples","gpu_name"
]

CHOSEN_COLUMNS = [
    "method","checkpoint_name","dataset","task","arch","checkpoint",
    "selected_lr","selection_metric","selected_final_val_acc",
    "selected_best_val_acc","selected_best_val_epoch",
    "lr_sweep_total_time_sec","lr_sweep_total_optimizer_steps",
    "lr_sweep_total_samples_seen","trainable_params","total_params",
    "trainable_percent","gpu_name"
]

FINAL_EPOCH_COLUMNS = [
    "method","checkpoint_name","dataset","task","arch","checkpoint",
    "phase","seed","selected_lr","lr_current","epoch","epochs","train_loss",
    "epoch_train_time_sec","cumulative_train_time_sec",
    "epoch_test_acc","epoch_test_loss","epoch_test_time_sec",
    "trainable_params","total_params","trainable_percent",
    "peak_allocated_mb","peak_reserved_mb","optimizer_steps_so_far",
    "samples_seen_so_far","probe_time_sec","missing_keys","unexpected_keys","gpu_name"
]

FINAL_SEED_COLUMNS = [
    "method","checkpoint_name","dataset","task","arch","checkpoint",
    "seed","selected_lr","test_acc","test_loss",
    "train_time_sec","test_time_sec","total_time_sec","probe_time_sec",
    "peak_allocated_mb","peak_reserved_mb",
    "trainable_params","total_params","trainable_percent",
    "optimizer_steps","samples_seen","train_examples","test_examples",
    "prediction_file","gpu_name"
]

FINAL_SUMMARY_COLUMNS = [
    "method","checkpoint_name","dataset","task","arch","checkpoint",
    "selected_lr","n_seeds","seeds","mean_test_acc","std_test_acc",
    "ci95_low","ci95_high","ci95_half_width",
    "mean_train_time_sec","std_train_time_sec","mean_total_time_sec",
    "mean_peak_allocated_mb","mean_peak_reserved_mb",
    "trainable_params","total_params","trainable_percent",
    "optimizer_steps_per_run","samples_seen_per_run","probe_time_sec","gpu_name"
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
            raise ValueError(f"Unknown VGG arch: {arch}")

        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def strip_module(state):
    out = {}
    for k, v in state.items():
        while k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out


def load_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    state = strip_module(state)
    fixed = {}
    for k, v in state.items():
        if k.startswith("classifier1."):
            k = k.replace("classifier1.", "classifier.")
        fixed[k] = v
    return fixed


def build_model(info):
    if info["type"] == "sophon":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif info["type"] == "vgg":
        model = CustomVGG(info["arch"], NUM_CLASSES)
    else:
        raise ValueError(info["type"])

    state = load_state(info["path"])
    missing, unexpected = model.load_state_dict(state, strict=False)

    for p in model.parameters():
        p.requires_grad = True

    return model, missing, unexpected


def get_datasets(dataset):
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
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_set, test_set


def split_train_validation(train_set):
    n_total = len(train_set)
    n_val = max(1, int(round(n_total * VAL_FRACTION)))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_subset, val_subset = random_split(
        train_set, [n_train, n_val], generator=generator
    )
    return train_subset, val_subset


def make_loader(dataset, shuffle, seed, drop_last):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=drop_last,
    )


def gpu_name():
    if DEVICE == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"


def reset_gpu_metrics():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_memory():
    if DEVICE != "cuda":
        return 0.0, 0.0
    torch.cuda.synchronize()
    return (
        torch.cuda.max_memory_allocated() / (1024 ** 2),
        torch.cuda.max_memory_reserved() / (1024 ** 2),
    )


def append_row(path, columns, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in columns})


def rewrite_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def setup_logger():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "full_ft.log")
    logger = logging.getLogger("full_ft")
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


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ce = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss_sum = 0.0
    batches = 0

    started = time.perf_counter()
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        out = model(x)
        loss = ce(out, y)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item()
        batches += 1

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return 100.0 * correct / total, loss_sum / max(1, batches), elapsed


@torch.no_grad()
def save_predictions(model, loader, path):
    model.eval()
    rows = []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        pred = model(x).argmax(1).cpu().numpy()
        for yt, yp in zip(y.numpy(), pred):
            rows.append([int(yt), int(yp)])
    pd.DataFrame([[i, yt, yp] for i, (yt, yp) in enumerate(rows)], columns=["sample_index", "y_true", "y_pred"]).to_csv(path, index=False)


def train_epoch(model, loader, optimizer, ce, desc):
    model.train()
    total_loss = 0.0
    batches = 0
    started = time.perf_counter()

    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = ce(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return total_loss / max(1, batches), elapsed


def model_budget(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    return trainable, total, pct


def paths():
    return {
        "sweep_epoch": os.path.join(OUT_DIR, "lr_sweep_epoch_results.csv"),
        "sweep_summary": os.path.join(OUT_DIR, "lr_sweep_summary.csv"),
        "chosen": os.path.join(OUT_DIR, "chosen_lr.csv"),
        "final_epoch": os.path.join(OUT_DIR, "final_epoch_results.csv"),
        "final_seed": os.path.join(OUT_DIR, "final_seed_results.csv"),
        "final_summary": os.path.join(OUT_DIR, "final_summary.csv"),
        "pred_dir": os.path.join(OUT_DIR, "predictions"),
    }


def select_infos(only):
    if only == "all":
        return CHECKPOINTS
    selected = [x for x in CHECKPOINTS if x["name"] == only]
    if not selected:
        valid = ", ".join(x["name"] for x in CHECKPOINTS)
        raise ValueError(f"Unknown checkpoint {only}. Valid: {valid}")
    return selected


def confirm_paths(infos):
    print(f"\nCHECKING {len(infos)} CHECKPOINTS")
    for info in infos:
        if not os.path.exists(info["path"]):
            raise FileNotFoundError(info["path"])
        print("FOUND:", info["path"])
    print("ALL SELECTED CHECKPOINTS FOUND\n")


def run_sweep_one(info, lr, train_set, val_set, logger, p):
    set_seed(SWEEP_SEED)
    train_loader = make_loader(train_set, True, SWEEP_SEED, True)
    val_loader = make_loader(val_set, False, SWEEP_SEED, False)

    model, missing, unexpected = build_model(info)
    model = model.to(DEVICE)
    trainable, total, pct = model_budget(model)

    optimizer = torch.optim.AdamW(
        [x for x in model.parameters() if x.requires_grad],
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    ce = nn.CrossEntropyLoss()

    reset_gpu_metrics()
    cumulative = 0.0
    train_time = 0.0
    val_time = 0.0
    best_val = -1.0
    best_epoch = -1
    final_val_acc = None
    final_val_loss = None
    steps_per_epoch = len(train_loader)
    samples_per_epoch = steps_per_epoch * BATCH_SIZE

    logger.info(
        "SWEEP | %s | lr=%g | train=%d | val=%d | trainable=%d/%d (%.6f%%)",
        info["name"], lr, len(train_set), len(val_set), trainable, total, pct,
    )

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, etrain = train_epoch(
            model, train_loader, optimizer, ce,
            f"SWEEP {info['name']} lr={lr:g} E{epoch}"
        )
        val_acc, val_loss, eval_time = evaluate(model, val_loader)
        train_time += etrain
        val_time += eval_time
        cumulative += etrain + eval_time

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch

        final_val_acc = val_acc
        final_val_loss = val_loss
        peak_alloc, peak_reserved = peak_memory()

        append_row(p["sweep_epoch"], SWEEP_EPOCH_COLUMNS, {
            "method": info["method"], "checkpoint_name": info["name"],
            "dataset": info["dataset"], "task": info["task"], "arch": info["arch"],
            "checkpoint": info["path"], "phase": "lr_sweep", "seed": SWEEP_SEED,
            "lr_initial": lr, "lr_current": current_lr, "epoch": epoch, "epochs": EPOCHS,
            "train_loss": round(train_loss, 8), "val_acc": round(val_acc, 6),
            "val_loss": round(val_loss, 8), "epoch_train_time_sec": round(etrain, 4),
            "epoch_val_time_sec": round(eval_time, 4), "cumulative_time_sec": round(cumulative, 4),
            "trainable_params": trainable, "total_params": total,
            "trainable_percent": round(pct, 6), "peak_allocated_mb": round(peak_alloc, 3),
            "peak_reserved_mb": round(peak_reserved, 3),
            "optimizer_steps_so_far": steps_per_epoch * epoch,
            "samples_seen_so_far": samples_per_epoch * epoch,
            "probe_time_sec": 0.0, "missing_keys": len(missing),
            "unexpected_keys": len(unexpected), "gpu_name": gpu_name(),
        })

        logger.info(
            "SWEEP | %s | lr=%g | epoch=%02d/%d | val=%.4f | best=%.4f@%d | "
            "train=%.2fs | valtime=%.2fs | peak_alloc=%.1fMB | peak_reserved=%.1fMB",
            info["name"], lr, epoch, EPOCHS, val_acc, best_val, best_epoch,
            etrain, eval_time, peak_alloc, peak_reserved,
        )

    peak_alloc, peak_reserved = peak_memory()
    row = {
        "method": info["method"], "checkpoint_name": info["name"],
        "dataset": info["dataset"], "task": info["task"], "arch": info["arch"],
        "checkpoint": info["path"], "seed": SWEEP_SEED, "lr": lr,
        "final_val_acc": round(final_val_acc, 6), "final_val_loss": round(final_val_loss, 8),
        "best_val_acc": round(best_val, 6), "best_val_epoch": best_epoch,
        "train_time_sec": round(train_time, 4), "validation_time_sec": round(val_time, 4),
        "total_time_sec": round(train_time + val_time, 4), "probe_time_sec": 0.0,
        "peak_allocated_mb": round(peak_alloc, 3), "peak_reserved_mb": round(peak_reserved, 3),
        "trainable_params": trainable, "total_params": total,
        "trainable_percent": round(pct, 6),
        "optimizer_steps": steps_per_epoch * EPOCHS,
        "samples_seen": samples_per_epoch * EPOCHS,
        "train_examples": len(train_set), "val_examples": len(val_set), "gpu_name": gpu_name(),
    }
    append_row(p["sweep_summary"], SWEEP_SUMMARY_COLUMNS, row)

    del optimizer, model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return row


def run_sweep(infos, logger, p):
    chosen_rows = []
    dataset_cache = {}

    for info in infos:
        if info["dataset"] not in dataset_cache:
            full_train, test_set = get_datasets(info["dataset"])
            train_set, val_set = split_train_validation(full_train)
            dataset_cache[info["dataset"]] = (full_train, train_set, val_set, test_set)
        _, train_set, val_set, _ = dataset_cache[info["dataset"]]

        rows = []
        for lr in LEARNING_RATES:
            rows.append(run_sweep_one(info, lr, train_set, val_set, logger, p))

        winner = sorted(rows, key=lambda r: (-float(r["final_val_acc"]), float(r["lr"])))[0]
        chosen = {
            "method": info["method"], "checkpoint_name": info["name"],
            "dataset": info["dataset"], "task": info["task"], "arch": info["arch"],
            "checkpoint": info["path"], "selected_lr": winner["lr"],
            "selection_metric": "epoch20_validation_accuracy",
            "selected_final_val_acc": winner["final_val_acc"],
            "selected_best_val_acc": winner["best_val_acc"],
            "selected_best_val_epoch": winner["best_val_epoch"],
            "lr_sweep_total_time_sec": round(sum(float(x["total_time_sec"]) for x in rows), 4),
            "lr_sweep_total_optimizer_steps": sum(int(x["optimizer_steps"]) for x in rows),
            "lr_sweep_total_samples_seen": sum(int(x["samples_seen"]) for x in rows),
            "trainable_params": winner["trainable_params"], "total_params": winner["total_params"],
            "trainable_percent": winner["trainable_percent"], "gpu_name": winner["gpu_name"],
        }
        chosen_rows.append(chosen)
        logger.info(
            "CHOSEN | %s | lr=%g | epoch20 val=%.4f%%",
            info["name"], float(winner["lr"]), float(winner["final_val_acc"]),
        )

    rewrite_csv(p["chosen"], CHOSEN_COLUMNS, chosen_rows)
    return chosen_rows


def load_chosen(p, infos):
    return {info["name"]: 1e-3 for info in infos}


def run_final_seed(info, lr, seed, full_train, test_set, logger, p):
    set_seed(seed)
    train_loader = make_loader(full_train, True, seed, True)
    test_loader = make_loader(test_set, False, seed, False)

    model, missing, unexpected = build_model(info)
    model = model.to(DEVICE)
    trainable, total, pct = model_budget(model)

    optimizer = torch.optim.AdamW(
        [x for x in model.parameters() if x.requires_grad],
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    ce = nn.CrossEntropyLoss()

    reset_gpu_metrics()
    train_time = 0.0
    max_train_alloc = 0.0
    max_train_reserved = 0.0
    steps_per_epoch = len(train_loader)
    samples_per_epoch = steps_per_epoch * BATCH_SIZE

    logger.info(
        "FINAL | %s | seed=%d | lr=%g | full_train=%d | trainable=%d/%d (%.6f%%)",
        info["name"], seed, lr, len(full_train), trainable, total, pct,
    )

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_loss, etrain = train_epoch(
            model, train_loader, optimizer, ce,
            f"FINAL {info['name']} seed={seed} E{epoch}"
        )
        train_time += etrain
        peak_alloc, peak_reserved = peak_memory()

        max_train_alloc = max(max_train_alloc, peak_alloc)
        max_train_reserved = max(max_train_reserved, peak_reserved)

        with torch.no_grad():
            epoch_test_acc, epoch_test_loss, epoch_test_time = evaluate(
                model, test_loader
            )

        append_row(p["final_epoch"], FINAL_EPOCH_COLUMNS, {
            "method": info["method"], "checkpoint_name": info["name"],
            "dataset": info["dataset"], "task": info["task"], "arch": info["arch"],
            "checkpoint": info["path"], "phase": "final", "seed": seed,
            "selected_lr": lr, "lr_current": current_lr, "epoch": epoch, "epochs": EPOCHS,
            "train_loss": round(train_loss, 8), "epoch_train_time_sec": round(etrain, 4),
            "cumulative_train_time_sec": round(train_time, 4),
            "epoch_test_acc": round(epoch_test_acc, 6),
            "epoch_test_loss": round(epoch_test_loss, 8),
            "epoch_test_time_sec": round(epoch_test_time, 4),
            "trainable_params": trainable, "total_params": total,
            "trainable_percent": round(pct, 6), "peak_allocated_mb": round(peak_alloc, 3),
            "peak_reserved_mb": round(peak_reserved, 3),
            "optimizer_steps_so_far": steps_per_epoch * epoch,
            "samples_seen_so_far": samples_per_epoch * epoch,
            "probe_time_sec": 0.0, "missing_keys": len(missing),
            "unexpected_keys": len(unexpected), "gpu_name": gpu_name(),
        })

        logger.info(
            "FINAL | %s | seed=%d | epoch=%02d/%d | loss=%.6f | "
            "test=%.4f%% | train=%.2fs | peak_alloc=%.1fMB | peak_reserved=%.1fMB",
            info["name"], seed, epoch, EPOCHS, train_loss,
            epoch_test_acc, etrain, peak_alloc, peak_reserved,
        )

    test_acc, test_loss, test_time = evaluate(model, test_loader)
    os.makedirs(p["pred_dir"], exist_ok=True)
    pred_path = os.path.join(p["pred_dir"], f"{info['name']}_seed{seed}_predictions.csv")
    save_predictions(model, test_loader, pred_path)

    peak_alloc = max_train_alloc
    peak_reserved = max_train_reserved
    row = {
        "method": info["method"], "checkpoint_name": info["name"],
        "dataset": info["dataset"], "task": info["task"], "arch": info["arch"],
        "checkpoint": info["path"], "seed": seed, "selected_lr": lr,
        "test_acc": round(test_acc, 6), "test_loss": round(test_loss, 8),
        "train_time_sec": round(train_time, 4), "test_time_sec": round(test_time, 4),
        "total_time_sec": round(train_time, 4), "probe_time_sec": 0.0,
        "peak_allocated_mb": round(peak_alloc, 3), "peak_reserved_mb": round(peak_reserved, 3),
        "trainable_params": trainable, "total_params": total,
        "trainable_percent": round(pct, 6),
        "optimizer_steps": steps_per_epoch * EPOCHS,
        "samples_seen": samples_per_epoch * EPOCHS,
        "train_examples": len(full_train), "test_examples": len(test_set),
        "prediction_file": pred_path, "gpu_name": gpu_name(),
    }
    append_row(p["final_seed"], FINAL_SEED_COLUMNS, row)

    logger.info(
        "RESULT | %s | seed=%d | lr=%g | test=%.4f%% | train=%.2fs | test=%.2fs | "
        "peak_alloc=%.1fMB | budget=%.4f%%",
        info["name"], seed, lr, test_acc, train_time, test_time, peak_alloc, pct,
    )

    del optimizer, model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return row


def aggregate_final(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["checkpoint_name"]].append(row)

    out = []
    for name, items in grouped.items():
        items = sorted(items, key=lambda x: int(x["seed"]))
        acc = np.array([float(x["test_acc"]) for x in items], dtype=float)
        train_times = np.array([float(x["train_time_sec"]) for x in items], dtype=float)
        total_times = np.array([float(x["total_time_sec"]) for x in items], dtype=float)
        alloc = np.array([float(x["peak_allocated_mb"]) for x in items], dtype=float)
        reserved = np.array([float(x["peak_reserved_mb"]) for x in items], dtype=float)

        mean_acc = float(acc.mean())
        std_acc = float(acc.std(ddof=1)) if len(acc) > 1 else 0.0
        if len(acc) == 3:
            half = T95_DF2 * std_acc / math.sqrt(3.0)
        elif len(acc) > 1:
            half = 1.96 * std_acc / math.sqrt(len(acc))
        else:
            half = 0.0

        first = items[0]
        out.append({
            "method": first["method"], "checkpoint_name": name,
            "dataset": first["dataset"], "task": first["task"], "arch": first["arch"],
            "checkpoint": first["checkpoint"], "selected_lr": first["selected_lr"],
            "n_seeds": len(items), "seeds": "|".join(str(x["seed"]) for x in items),
            "mean_test_acc": round(mean_acc, 6), "std_test_acc": round(std_acc, 6),
            "ci95_low": round(mean_acc - half, 6), "ci95_high": round(mean_acc + half, 6),
            "ci95_half_width": round(half, 6),
            "mean_train_time_sec": round(float(train_times.mean()), 4),
            "std_train_time_sec": round(float(train_times.std(ddof=1)) if len(train_times) > 1 else 0.0, 4),
            "mean_total_time_sec": round(float(total_times.mean()), 4),
            "mean_peak_allocated_mb": round(float(alloc.mean()), 3),
            "mean_peak_reserved_mb": round(float(reserved.mean()), 3),
            "trainable_params": first["trainable_params"], "total_params": first["total_params"],
            "trainable_percent": first["trainable_percent"],
            "optimizer_steps_per_run": first["optimizer_steps"],
            "samples_seen_per_run": first["samples_seen"],
            "probe_time_sec": 0.0, "gpu_name": first["gpu_name"],
        })
    return out


def run_final(infos, logger, p):
    chosen = load_chosen(p, infos)
    dataset_cache = {}
    rows = []

    for info in infos:
        if info["dataset"] not in dataset_cache:
            dataset_cache[info["dataset"]] = get_datasets(info["dataset"])
        full_train, test_set = dataset_cache[info["dataset"]]
        lr = chosen[info["name"]]
        for seed in FINAL_SEEDS:
            rows.append(run_final_seed(info, lr, seed, full_train, test_set, logger, p))

    summaries = aggregate_final(rows)
    rewrite_csv(p["final_summary"], FINAL_SUMMARY_COLUMNS, summaries)
    return summaries


def clear_outputs(p, phase):
    targets = []
    if phase in {"sweep", "all"}:
        targets += [p["sweep_epoch"], p["sweep_summary"], p["chosen"]]
    if phase in {"final", "all"}:
        targets += [p["final_epoch"], p["final_seed"], p["final_summary"]]
    for x in targets:
        if os.path.exists(x):
            os.remove(x)
    if phase in {"final", "all"} and os.path.isdir(p["pred_dir"]):
        for name in os.listdir(p["pred_dir"]):
            os.remove(os.path.join(p["pred_dir"], name))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    p = paths()
    logger = setup_logger()
    infos = select_infos(args.only)
    confirm_paths(infos)

    logger.info("=" * 100)
    logger.info("FULL-FT FIXED-LR MULTI-SEED EVALUATION")
    logger.info("Fixed LR: 0.001")
    logger.info("Final seeds: %s", FINAL_SEEDS)
    logger.info(
        "AdamW | weight_decay=%g | batch=%d | epochs=%d | constant LR",
        WEIGHT_DECAY, BATCH_SIZE, EPOCHS
    )
    logger.info("Full target training set; no LR sweep or validation selection")
    logger.info("Full-FT budget: 100%% trainable | probing time=0")
    logger.info("GPU: %s", gpu_name())
    logger.info("=" * 100)

    summaries = run_final(infos, logger, p)

    print("\nFINAL MEAN ± STD (95% CI)")
    for r in summaries:
        print(
            f"{r['checkpoint_name']}: "
            f"{r['mean_test_acc']:.4f} ± {r['std_test_acc']:.4f} "
            f"(95% CI {r['ci95_low']:.4f}–{r['ci95_high']:.4f}) | "
            f"LR=0.001 | time={r['mean_train_time_sec']:.2f}s | "
            f"peak={r['mean_peak_allocated_mb']:.1f}MB | "
            f"budget={r['trainable_percent']:.4f}%"
        )


if __name__ == "__main__":
    main()
