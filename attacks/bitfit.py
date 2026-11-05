
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
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

DATA_ROOT = "./data"
OUT_DIR = "./results_clean/bitfit"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATES = [1e-3]
EPOCHS = 20
BATCH_SIZE = 64
WORKERS = 4
WEIGHT_DECAY = 1e-4

SWEEP_SEED = 41
FINAL_SEEDS = [42, 43, 44]
SPLIT_SEED = 2026
VAL_FRACTION = 0.10

NUM_CLASSES = 10
ATTACK = "bitfit"
CI_T_DF2 = 4.302652729911275

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

SWEEP_EPOCH_CSV = os.path.join(OUT_DIR, "lr_sweep_epoch_results.csv")
SWEEP_SUMMARY_CSV = os.path.join(OUT_DIR, "lr_sweep_summary.csv")
CHOSEN_LR_CSV = os.path.join(OUT_DIR, "chosen_lr.csv")
FINAL_EPOCH_CSV = os.path.join(OUT_DIR, "final_epoch_results.csv")
FINAL_SEED_CSV = os.path.join(OUT_DIR, "final_seed_results.csv")
FINAL_SUMMARY_CSV = os.path.join(OUT_DIR, "final_summary.csv")
PARAM_DETAILS_CSV = os.path.join(OUT_DIR, "parameter_details.csv")
PRED_DIR = os.path.join(OUT_DIR, "predictions")
LOG_PATH = os.path.join(OUT_DIR, "bitfit.log")


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


class STL10Mapped(datasets.STL10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, RE_LABEL[target]


def get_transform(dataset):
    if dataset.upper() == "MNIST":
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])


def get_dataset_pair(dataset):
    tf = get_transform(dataset)
    d = dataset.upper()

    if d == "CIFAR10":
        train = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=tf)
        test = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=tf)
    elif d == "MNIST":
        train = datasets.MNIST(DATA_ROOT, train=True, download=False, transform=tf)
        test = datasets.MNIST(DATA_ROOT, train=False, download=False, transform=tf)
    elif d == "SVHN":
        train = datasets.SVHN(DATA_ROOT, split="train", download=False, transform=tf)
        test = datasets.SVHN(DATA_ROOT, split="test", download=False, transform=tf)
    elif d == "STL10":
        train = STL10Mapped(DATA_ROOT, split="train", download=False, transform=tf)
        test = STL10Mapped(DATA_ROOT, split="test", download=False, transform=tf)
    elif d == "CINIC":
        root = os.path.join(DATA_ROOT, "CINIC10")
        train_dir = os.path.join(root, "train")
        valid_dir = os.path.join(root, "valid")
        test_dir = os.path.join(root, "test")

        if os.path.isdir(train_dir) and os.path.isdir(valid_dir):
            train = datasets.ImageFolder(train_dir, transform=tf)
            test = datasets.ImageFolder(valid_dir, transform=tf)
        elif os.path.isdir(train_dir) and os.path.isdir(test_dir):
            train = datasets.ImageFolder(train_dir, transform=tf)
            test = datasets.ImageFolder(test_dir, transform=tf)
        else:
            raise FileNotFoundError(f"CINIC folders not found in {root}")
    else:
        raise ValueError(dataset)

    return train, test


def split_train_val(train_set):
    # Use the same fixed 90/10 split as the other PEFT methods.
    n_total = len(train_set)
    n_val = int(round(n_total * VAL_FRACTION))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(SPLIT_SEED)
    return random_split(train_set, [n_train, n_val], generator=g)


def make_loader(ds, train, seed=None):
    if train:
        if seed is None:
            raise ValueError("A seed is required for a shuffled training loader.")
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            pin_memory=True,
            drop_last=True,
        )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        drop_last=False,
    )


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
        model = build_resnet18()
    elif info["type"] == "vgg":
        model = CustomVGG(info["arch"], NUM_CLASSES)
    else:
        raise ValueError(info["type"])

    state = load_state(info["path"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    return model, list(missing), list(unexpected)


def classifier_bias_names(model):
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    if hasattr(model, "classifier"):
        classifier_modules = {n for n in linear_names if n.startswith("classifier")}
    else:
        classifier_modules = {linear_names[-1]} if linear_names else set()

    out = set()
    for module_name in classifier_modules:
        out.add(module_name + ".bias")
    return out


def apply_bitfit(model):
    """Enable training only for bias parameters."""
    for p in model.parameters():
        p.requires_grad = False

    classifier_biases = classifier_bias_names(model)

    trainable_names = []
    trainable_bias_params = 0
    classifier_bias_params = 0
    detail_rows = []

    for name, p in model.named_parameters():
        is_bias = name.endswith(".bias")
        if not is_bias:
            continue

        p.requires_grad = True
        n = p.numel()
        trainable_names.append(name)
        trainable_bias_params += n

        is_classifier = name in classifier_biases
        if is_classifier:
            classifier_bias_params += n

        detail_rows.append({
            "parameter_name": name,
            "num_params": n,
            "is_classifier": int(is_classifier),
        })

    total_model_params = sum(p.numel() for p in model.parameters())

    if trainable_bias_params == 0:
        raise RuntimeError("No bias parameters found for BitFit.")

    return {
        "trainable_names": trainable_names,
        "trainable_bias_params": trainable_bias_params,
        "classifier_bias_params": classifier_bias_params,
        "total_model_params": total_model_params,
        "trainable_percent": 100.0 * trainable_bias_params / total_model_params,
        "detail_rows": detail_rows,
    }


@torch.no_grad()
def evaluate(model, loader, return_predictions=False):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")

    correct = 0
    total = 0
    loss_sum = 0.0

    sample_index = []
    y_true = []
    y_pred = []
    cursor = 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y_dev = y.to(DEVICE, non_blocking=True)

        out = model(x)
        loss_sum += ce(out, y_dev).item()
        pred = out.argmax(1)

        correct += (pred == y_dev).sum().item()
        total += y_dev.size(0)

        if return_predictions:
            b = y.size(0)
            sample_index.extend(range(cursor, cursor + b))
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            cursor += b

    acc = 100.0 * correct / total
    avg_loss = loss_sum / max(total, 1)

    if return_predictions:
        return acc, avg_loss, pd.DataFrame({
            "sample_index": sample_index,
            "y_true": y_true,
            "y_pred": y_pred,
        })

    return acc, avg_loss


def setup_logger():
    logger = logging.getLogger("bitfit")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(message)s")

    fh = logging.FileHandler(LOG_PATH)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def append_row(path, row, columns):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_dataframe(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def gpu_name():
    return torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"


def reset_cuda_peak():
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


def write_parameter_details(info, phase, seed, lr, stats):
    columns = [
        "method","checkpoint_name","dataset","task","arch",
        "phase","seed","lr","parameter_name","num_params","is_classifier"
    ]

    for d in stats["detail_rows"]:
        row = {
            "method": info["method"],
            "checkpoint_name": info["name"],
            "dataset": info["dataset"],
            "task": info["task"],
            "arch": info["arch"],
            "phase": phase,
            "seed": seed,
            "lr": lr,
            **d,
        }
        append_row(PARAM_DETAILS_CSV, row, columns)


def train_epoch(model, loader, optimizer):
    model.train()
    ce = nn.CrossEntropyLoss()

    loss_sum = 0.0
    steps = 0
    samples = 0

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = ce(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        steps += 1
        samples += y.size(0)


    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return loss_sum / max(1, steps), elapsed, steps, samples


SWEEP_EPOCH_COLUMNS = [
    "method","checkpoint_name","dataset","task","arch","checkpoint",
    "attack","lr","sweep_seed","split_seed","epoch","epochs",
    "val_acc","val_loss","best_val_acc","best_val_epoch",
    "epoch_train_time_sec","val_time_sec",
    "peak_alloc_mb","peak_reserved_mb",
    "trainable_bias_params","classifier_bias_params","total_model_params",
    "trainable_percent","num_trainable_bias_tensors",
    "optimizer_steps","samples_seen","gpu_name",
    "missing_keys","unexpected_keys",
]

FINAL_EPOCH_COLUMNS = [
    "method","checkpoint_name","dataset","task","arch","checkpoint",
    "attack","seed","lr","epoch","epochs","train_loss",
    "epoch_train_time_sec","cumulative_train_time_sec",
    "epoch_test_acc","epoch_test_loss","epoch_test_time_sec",
    "peak_alloc_mb","peak_reserved_mb",
    "trainable_bias_params","classifier_bias_params","total_model_params",
    "trainable_percent","num_trainable_bias_tensors",
    "optimizer_steps","samples_seen","gpu_name",
    "missing_keys","unexpected_keys",
]


def run_sweep_one(info, lr, train_subset, val_subset, logger):
    set_seed(SWEEP_SEED)

    train_loader = make_loader(train_subset, True, SWEEP_SEED)
    val_loader = make_loader(val_subset, False)

    model, missing, unexpected = build_model(info)
    model = model.to(DEVICE)

    reset_cuda_peak()
    stats = apply_bitfit(model)
    write_parameter_details(info, "sweep", SWEEP_SEED, lr, stats)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    logger.info(
        f"SWEEP | {info['name']} | lr={lr} | "
        f"train={len(train_subset)} | val={len(val_subset)} | "
        f"trainable_bias={stats['trainable_bias_params']}/{stats['total_model_params']} "
        f"({stats['trainable_percent']:.6f}%) | "
        f"classifier_bias={stats['classifier_bias_params']} | "
        f"bias_tensors={len(stats['trainable_names'])}"
    )

    best_val = -1.0
    best_epoch = -1
    epoch20_val = None

    total_train_time = 0.0
    total_val_time = 0.0
    total_steps = 0
    total_samples = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_time, steps, samples = train_epoch(
            model, train_loader, optimizer
        )
        total_train_time += train_time
        total_steps += steps
        total_samples += samples

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        tv = time.perf_counter()
        val_acc, val_loss = evaluate(model, val_loader)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        val_time = time.perf_counter() - tv
        total_val_time += val_time

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch

        if epoch == EPOCHS:
            epoch20_val = val_acc

        peak_alloc, peak_reserved = peak_memory()

        row = {
            "method": info["method"],
            "checkpoint_name": info["name"],
            "dataset": info["dataset"],
            "task": info["task"],
            "arch": info["arch"],
            "checkpoint": info["path"],
            "attack": ATTACK,
            "lr": lr,
            "sweep_seed": SWEEP_SEED,
            "split_seed": SPLIT_SEED,
            "epoch": epoch,
            "epochs": EPOCHS,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "best_val_acc": best_val,
            "best_val_epoch": best_epoch,
            "epoch_train_time_sec": train_time,
            "val_time_sec": val_time,
            "peak_alloc_mb": peak_alloc,
            "peak_reserved_mb": peak_reserved,
            "trainable_bias_params": stats["trainable_bias_params"],
            "classifier_bias_params": stats["classifier_bias_params"],
            "total_model_params": stats["total_model_params"],
            "trainable_percent": stats["trainable_percent"],
            "num_trainable_bias_tensors": len(stats["trainable_names"]),
            "optimizer_steps": total_steps,
            "samples_seen": total_samples,
            "gpu_name": gpu_name(),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        }
        append_row(SWEEP_EPOCH_CSV, row, SWEEP_EPOCH_COLUMNS)

        logger.info(
            f"SWEEP | {info['name']} | lr={lr} | epoch={epoch:02d}/{EPOCHS} | "
            f"val={val_acc:.4f} | best={best_val:.4f}@{best_epoch} | "
            f"train={train_time:.2f}s | valtime={val_time:.2f}s | "
            f"peak_alloc={peak_alloc:.1f}MB | peak_reserved={peak_reserved:.1f}MB"
        )

    peak_alloc, peak_reserved = peak_memory()

    row = {
        "method": info["method"],
        "checkpoint_name": info["name"],
        "dataset": info["dataset"],
        "task": info["task"],
        "arch": info["arch"],
        "checkpoint": info["path"],
        "attack": ATTACK,
        "lr": lr,
        "sweep_seed": SWEEP_SEED,
        "split_seed": SPLIT_SEED,
        "epoch20_val_acc": epoch20_val,
        "best_val_acc": best_val,
        "best_val_epoch": best_epoch,
        "sweep_train_time_sec": total_train_time,
        "sweep_val_time_sec": total_val_time,
        "sweep_total_time_sec": total_train_time + total_val_time,
        "peak_alloc_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
        "trainable_bias_params": stats["trainable_bias_params"],
        "classifier_bias_params": stats["classifier_bias_params"],
        "total_model_params": stats["total_model_params"],
        "trainable_percent": stats["trainable_percent"],
        "num_trainable_bias_tensors": len(stats["trainable_names"]),
        "optimizer_steps": total_steps,
        "samples_seen": total_samples,
        "gpu_name": gpu_name(),
    }

    del model, optimizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return row


def choose_lrs(rows):
    chosen = []
    groups = {}

    for r in rows:
        groups.setdefault(r["checkpoint_name"], []).append(r)

    for _, vals in groups.items():
        vals = sorted(
            vals,
            key=lambda r: (-float(r["epoch20_val_acc"]), float(r["lr"]))
        )
        winner = dict(vals[0])
        winner["chosen_lr"] = winner["lr"]
        winner["selection_metric"] = "epoch20_validation_accuracy"
        winner["lr_candidates"] = "|".join(str(x) for x in LEARNING_RATES)
        chosen.append(winner)

    return chosen


def run_final_one(info, lr, seed, full_train, test_set, logger):

    train_loader = make_loader(full_train, True, seed)
    test_loader = make_loader(test_set, False)

    model, missing, unexpected = build_model(info)
    model = model.to(DEVICE)

    reset_cuda_peak()
    stats = apply_bitfit(model)
    write_parameter_details(info, "final", seed, lr, stats)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    logger.info(
        f"FINAL | {info['name']} | seed={seed} | lr={lr} | "
        f"full_train={len(full_train)} | "
        f"trainable_bias={stats['trainable_bias_params']}/{stats['total_model_params']} "
        f"({stats['trainable_percent']:.6f}%) | "
        f"classifier_bias={stats['classifier_bias_params']} | "
        f"bias_tensors={len(stats['trainable_names'])}"
    )

    total_train_time = 0.0
    total_steps = 0
    total_samples = 0
    max_train_alloc = 0.0
    max_train_reserved = 0.0

    reset_cuda_peak()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_time, steps, samples = train_epoch(
            model, train_loader, optimizer
        )

        total_train_time += train_time
        total_steps += steps
        total_samples += samples

        peak_alloc, peak_reserved = peak_memory()
        max_train_alloc = max(max_train_alloc, peak_alloc)
        max_train_reserved = max(max_train_reserved, peak_reserved)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        epoch_test_start = time.perf_counter()

        with torch.no_grad():
            epoch_test_acc, epoch_test_loss = evaluate(
                model, test_loader, return_predictions=False
            )

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        epoch_test_time = time.perf_counter() - epoch_test_start

        # Reset memory tracking before the next training epoch.
        reset_cuda_peak()

        row = {
            "method": info["method"],
            "checkpoint_name": info["name"],
            "dataset": info["dataset"],
            "task": info["task"],
            "arch": info["arch"],
            "checkpoint": info["path"],
            "attack": ATTACK,
            "seed": seed,
            "lr": lr,
            "epoch": epoch,
            "epochs": EPOCHS,
            "train_loss": train_loss,
            "epoch_train_time_sec": train_time,
            "cumulative_train_time_sec": total_train_time,
            "epoch_test_acc": epoch_test_acc,
            "epoch_test_loss": epoch_test_loss,
            "epoch_test_time_sec": epoch_test_time,
            "peak_alloc_mb": peak_alloc,
            "peak_reserved_mb": peak_reserved,
            "trainable_bias_params": stats["trainable_bias_params"],
            "classifier_bias_params": stats["classifier_bias_params"],
            "total_model_params": stats["total_model_params"],
            "trainable_percent": stats["trainable_percent"],
            "num_trainable_bias_tensors": len(stats["trainable_names"]),
            "optimizer_steps": total_steps,
            "samples_seen": total_samples,
            "gpu_name": gpu_name(),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        }
        append_row(FINAL_EPOCH_CSV, row, FINAL_EPOCH_COLUMNS)

        logger.info(
            f"FINAL | {info['name']} | seed={seed} | lr={lr} | "
            f"epoch={epoch:02d}/{EPOCHS} | loss={train_loss:.6f} | "
            f"test_acc={epoch_test_acc:.4f}% | "
            f"train={train_time:.2f}s | peak_alloc={peak_alloc:.1f}MB | "
            f"peak_reserved={peak_reserved:.1f}MB"
        )

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    tt = time.perf_counter()

    test_acc, test_loss, pred_df = evaluate(
        model, test_loader, return_predictions=True
    )

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    test_time = time.perf_counter() - tt

    pred_path = os.path.join(
        PRED_DIR,
        f"{info['name']}__bitfit__seed{seed}.csv"
    )
    pred_df.to_csv(pred_path, index=False)

    peak_alloc = max_train_alloc
    peak_reserved = max_train_reserved

    seed_row = {
        "method": info["method"],
        "checkpoint_name": info["name"],
        "dataset": info["dataset"],
        "task": info["task"],
        "arch": info["arch"],
        "checkpoint": info["path"],
        "attack": ATTACK,
        "seed": seed,
        "lr": lr,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train_time_sec": total_train_time,
        "test_time_sec": test_time,
        "total_attack_time_sec": total_train_time,
        "peak_alloc_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
        "trainable_bias_params": stats["trainable_bias_params"],
        "classifier_bias_params": stats["classifier_bias_params"],
        "total_model_params": stats["total_model_params"],
        "trainable_percent": stats["trainable_percent"],
        "num_trainable_bias_tensors": len(stats["trainable_names"]),
        "optimizer_steps": total_steps,
        "samples_seen": total_samples,
        "gpu_name": gpu_name(),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "predictions_file": pred_path,
    }

    logger.info(
        f"RESULT | {info['name']} | seed={seed} | lr={lr} | "
        f"test={test_acc:.4f}% | train={total_train_time:.2f}s | "
        f"testtime={test_time:.2f}s | attack_total={total_train_time:.2f}s | "
        f"peak_alloc={peak_alloc:.1f}MB | "
        f"trainable={stats['trainable_percent']:.6f}%"
    )

    del model, optimizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return seed_row


def summarize_final(seed_rows):
    groups = {}
    for r in seed_rows:
        groups.setdefault(r["checkpoint_name"], []).append(r)

    out = []
    for _, vals in groups.items():
        latest = {}
        for v in vals:
            seed = int(v["seed"])
            if seed in FINAL_SEEDS:
                latest[seed] = v
        if set(latest) != set(FINAL_SEEDS):
            continue

        vals = [latest[s] for s in FINAL_SEEDS]
        acc = np.array([float(v["test_acc"]) for v in vals], dtype=float)
        mean = float(acc.mean())
        sd = float(acc.std(ddof=1))
        half = CI_T_DF2 * sd / math.sqrt(len(FINAL_SEEDS))

        base = vals[0]
        allocs = [float(v["peak_alloc_mb"]) for v in vals]
        reservs = [float(v["peak_reserved_mb"]) for v in vals]

        out.append({
            "method": base["method"],
            "checkpoint_name": base["checkpoint_name"],
            "dataset": base["dataset"],
            "task": base["task"],
            "arch": base["arch"],
            "checkpoint": base["checkpoint"],
            "attack": ATTACK,
            "lr": base["lr"],
            "n_seeds": len(FINAL_SEEDS),
            "seeds": "|".join(str(s) for s in FINAL_SEEDS),
            "mean_test_acc": mean,
            "sample_sd": sd,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "mean_train_time_sec": float(np.mean([float(v["train_time_sec"]) for v in vals])),
            "mean_test_time_sec": float(np.mean([float(v["test_time_sec"]) for v in vals])),
            "mean_total_attack_time_sec": float(np.mean([float(v["total_attack_time_sec"]) for v in vals])),
            "mean_peak_alloc_mb": float(np.mean(allocs)),
            "mean_peak_reserved_mb": float(np.mean(reservs)),
            "max_peak_alloc_mb": float(np.max(allocs)),
            "max_peak_reserved_mb": float(np.max(reservs)),
            "trainable_bias_params": base["trainable_bias_params"],
            "classifier_bias_params": base["classifier_bias_params"],
            "total_model_params": base["total_model_params"],
            "trainable_percent": base["trainable_percent"],
            "num_trainable_bias_tensors": base["num_trainable_bias_tensors"],
            "gpu_name": base["gpu_name"],
        })
    return out

def confirm_paths():
    print("Checking checkpoints...")
    for info in CHECKPOINTS:
        if not os.path.exists(info["path"]):
            raise FileNotFoundError(info["path"])
        print("Checkpoint:", info["path"])
    print("All checkpoints found.")


def load_csv_records(path):
    if not os.path.exists(path):
        return []
    return pd.read_csv(path).to_dict("records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    confirm_paths()
    logger = setup_logger()

    logger.info("BitFit evaluation")
    logger.info("All bias parameters are trainable; all other parameters remain frozen")
    logger.info("Learning rate: 0.001")
    logger.info("Seeds: %s", FINAL_SEEDS)
    logger.info(
        "AdamW: weight_decay=%g, batch=%d, epochs=%d, constant LR",
        WEIGHT_DECAY, BATCH_SIZE, EPOCHS,
    )
    logger.info("Full target training set with fixed learning rate")
    logger.info("Predictions are saved for each seed")
    logger.info("Classifier bias parameters are recorded separately")
    logger.info("No probing or parameter-selection stage is required")
    logger.info(f"GPU: {gpu_name()}")

    fixed_lr = 1e-3
    dataset_cache = {}

    seed_rows = load_csv_records(FINAL_SEED_CSV)
    done = {
        (r["checkpoint_name"], int(r["seed"]))
        for r in seed_rows
    }

    for seed in FINAL_SEEDS:
        set_seed(seed)

        for info in CHECKPOINTS:
            if args.only != "all" and args.only != info["name"]:
                continue

            if info["dataset"] not in dataset_cache:
                dataset_cache[info["dataset"]] = get_dataset_pair(info["dataset"])

            full_train, test_set = dataset_cache[info["dataset"]]

            key = (info["name"], seed)

            if key in done:
                logger.info(f"Skip existing run: {key}")
                continue

            row = run_final_one(
                info, fixed_lr, seed, full_train, test_set, logger
            )
            seed_rows.append(row)
            done.add(key)
            write_dataframe(FINAL_SEED_CSV, seed_rows)

    summary = summarize_final(seed_rows)
    write_dataframe(FINAL_SUMMARY_CSV, summary)

    logger.info("Final mean ± std (95% CI)")
    for r in summary:
        logger.info(
            f"{r['checkpoint_name']}: "
            f"{r['mean_test_acc']:.4f} ± {r['sample_sd']:.4f} "
            f"(95% CI {r['ci95_low']:.4f}–{r['ci95_high']:.4f}) | "
            f"LR=0.001 | "
            f"trainable={r['trainable_percent']:.6f}% | "
            f"attack_time={r['mean_total_attack_time_sec']:.2f}s | "
            f"peak={r['mean_peak_alloc_mb']:.1f}MB"
        )


if __name__ == "__main__":
    main()
