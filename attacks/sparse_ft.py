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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


DATA_ROOT = "./data"
CKPT_ROOT = "./checkpoints"
OUT_DIR = "./results_clean/sparseft"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SparseFT budgets and evaluation seeds.
SPARSITIES = [0.01, 0.05, 0.10, 0.15]
FINAL_SEEDS = [42, 43, 44]

EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 64
WORKERS = 4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 10

CI_T_DF2 = 4.302652729911275
ATTACK = "strong_sparseft"

RE_LABEL = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]

CHECKPOINTS = [
    {"method":"SOPHON-IL","name":"sophon_il_cifar10","type":"sophon","arch":"resnet18","dataset":"CIFAR10","task":"CIFAR10","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_cifar10.pth")},
    {"method":"SOPHON-IL","name":"sophon_il_cinic","type":"sophon","arch":"resnet18","dataset":"CINIC","task":"CINIC","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_cinic.pth")},
    {"method":"SOPHON-IL","name":"sophon_il_mnist","type":"sophon","arch":"resnet18","dataset":"MNIST","task":"MNIST","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_mnist.pth")},
    {"method":"SOPHON-IL","name":"sophon_il_stl10","type":"sophon","arch":"resnet18","dataset":"STL10","task":"STL10","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_stl10.pth")},
    {"method":"SOPHON-IL","name":"sophon_il_svhn","type":"sophon","arch":"resnet18","dataset":"SVHN","task":"SVHN","path":os.path.join(CKPT_ROOT,"sophon_il","sophon_il_svhn.pth")},

    {"method":"SOPHON-KL","name":"sophon_kl_cifar10","type":"sophon","arch":"resnet18","dataset":"CIFAR10","task":"CIFAR10","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_cifar10.pth")},
    {"method":"SOPHON-KL","name":"sophon_kl_cinic","type":"sophon","arch":"resnet18","dataset":"CINIC","task":"CINIC","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_cinic.pth")},
    {"method":"SOPHON-KL","name":"sophon_kl_mnist","type":"sophon","arch":"resnet18","dataset":"MNIST","task":"MNIST","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_mnist.pth")},
    {"method":"SOPHON-KL","name":"sophon_kl_stl10","type":"sophon","arch":"resnet18","dataset":"STL10","task":"STL10","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_stl10.pth")},
    {"method":"SOPHON-KL","name":"sophon_kl_svhn","type":"sophon","arch":"resnet18","dataset":"SVHN","task":"SVHN","path":os.path.join(CKPT_ROOT,"sophon_kl","sophon_kl_svhn.pth")},

    {"method":"HNTL","name":"hntl_cifar_stl_vgg13","type":"vgg","arch":"vgg13","dataset":"STL10","task":"CIFAR10_to_STL10","path":os.path.join(CKPT_ROOT,"hntl_cifar_stl_vgg13.pth")},
    {"method":"CUPI","name":"cupi_cifar_stl_vgg13","type":"vgg","arch":"vgg13","dataset":"STL10","task":"CIFAR10_to_STL10","path":os.path.join(CKPT_ROOT,"cupi_cifar_stl_vgg13.pth")},
    {"method":"CUTI","name":"cuti_cifar_stl_vgg19","type":"vgg","arch":"vgg19","dataset":"STL10","task":"CIFAR10_to_STL10","path":os.path.join(CKPT_ROOT,"cuti_cifar_stl_vgg19.pth")},
]

FINAL_EPOCH_CSV = os.path.join(OUT_DIR, "final_epoch_results.csv")
FINAL_SEED_CSV = os.path.join(OUT_DIR, "final_seed_results.csv")
FINAL_SUMMARY_CSV = os.path.join(OUT_DIR, "final_summary.csv")
BUDGET_DETAILS_CSV = os.path.join(OUT_DIR, "budget_details.csv")
PRED_DIR = os.path.join(OUT_DIR, "predictions")
LOG_PATH = os.path.join(OUT_DIR, "sparse_ft.log")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class STL10Mapped(datasets.STL10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, RE_LABEL[target]


def get_loaders(dataset):
    # Dataset preprocessing.
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
        train_set = datasets.CIFAR10(
            DATA_ROOT, train=True, download=False, transform=tf_rgb
        )
        test_set = datasets.CIFAR10(
            DATA_ROOT, train=False, download=False, transform=tf_rgb
        )

    elif d == "MNIST":
        train_set = datasets.MNIST(
            DATA_ROOT, train=True, download=False, transform=tf_mnist
        )
        test_set = datasets.MNIST(
            DATA_ROOT, train=False, download=False, transform=tf_mnist
        )

    elif d == "SVHN":
        train_set = datasets.SVHN(
            DATA_ROOT, split="train", download=False, transform=tf_rgb
        )
        test_set = datasets.SVHN(
            DATA_ROOT, split="test", download=False, transform=tf_rgb
        )

    elif d == "STL10":
        train_set = STL10Mapped(
            DATA_ROOT, split="train", download=False, transform=tf_rgb
        )
        test_set = STL10Mapped(
            DATA_ROOT, split="test", download=False, transform=tf_rgb
        )

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
            raise FileNotFoundError(
                f"CINIC folders not found inside {cinic_root}"
            )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # The run seed is set before creating the data loader.
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
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


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def make_sparse_mask(param, sparsity):
    # Select the largest-magnitude weights in each layer; ties at the threshold are retained.
    flat = param.detach().abs().flatten()
    k = max(1, int(flat.numel() * sparsity))
    threshold = torch.topk(flat, k=k, largest=True).values[-1]
    return (param.detach().abs() >= threshold).float()


def apply_sparseft(model, sparsity):
    # Apply magnitude masks to convolutional and linear weights; biases remain trainable.
    freeze_all(model)

    masks = {}
    originals = {}
    hooks = []
    trainable_params = 0
    targeted_layers = []
    detail_rows = []

    selection_start = time.perf_counter()

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        if hasattr(module, "weight") and module.weight is not None:
            mask = make_sparse_mask(
                module.weight.data, sparsity
            ).to(module.weight.device)

            module.weight.requires_grad = True

            def hook_fn(grad, m=mask):
                return grad * m.to(grad.device)

            hooks.append(module.weight.register_hook(hook_fn))

            masks[module.weight] = mask
            originals[module.weight] = module.weight.detach().clone()

            selected = int(mask.sum().item())
            trainable_params += selected
            targeted_layers.append(name + ".weight")

            detail_rows.append({
                "layer": name + ".weight",
                "module_type": module.__class__.__name__,
                "tensor_params": module.weight.numel(),
                "selected_params": selected,
                "fully_trainable": 0,
            })

        if hasattr(module, "bias") and module.bias is not None:
            module.bias.requires_grad = True

            bmask = torch.ones_like(module.bias.data).to(module.bias.device)
            masks[module.bias] = bmask
            originals[module.bias] = module.bias.detach().clone()

            trainable_params += module.bias.numel()
            targeted_layers.append(name + ".bias")

            detail_rows.append({
                "layer": name + ".bias",
                "module_type": module.__class__.__name__,
                "tensor_params": module.bias.numel(),
                "selected_params": module.bias.numel(),
                "fully_trainable": 1,
            })

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    selection_time = time.perf_counter() - selection_start

    return masks, originals, hooks, trainable_params, targeted_layers, detail_rows, selection_time


def restore_unselected(masks, originals):
    # Restore unselected weights after each update.
    with torch.no_grad():
        for p, m in masks.items():
            m = m.to(p.device)
            orig = originals[p].to(p.device)
            p.data.copy_(p.data * m + orig * (1.0 - m))


@torch.no_grad()
def evaluate(model, loader, return_predictions=False):
    model.eval()
    ce = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    loss_sum = 0.0

    rows = []
    sample_index = 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y_dev = y.to(DEVICE, non_blocking=True)

        out = model(x)
        loss = ce(out, y_dev)
        pred = out.argmax(1)

        correct += (pred == y_dev).sum().item()
        total += y_dev.size(0)
        loss_sum += loss.item()

        if return_predictions:
            y_cpu = y.cpu().tolist()
            p_cpu = pred.cpu().tolist()
            for yt, yp in zip(y_cpu, p_cpu):
                rows.append((sample_index, int(yt), int(yp)))
                sample_index += 1

    acc = 100.0 * correct / total
    avg_loss = loss_sum / len(loader)

    if return_predictions:
        return acc, avg_loss, rows
    return acc, avg_loss


def gpu_name():
    return torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"


def reset_cuda_peak():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_memory_mb():
    if DEVICE != "cuda":
        return 0.0, 0.0
    torch.cuda.synchronize()
    return (
        torch.cuda.max_memory_allocated() / (1024 ** 2),
        torch.cuda.max_memory_reserved() / (1024 ** 2),
    )


def append_csv(path, row, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in columns})


EPOCH_COLUMNS = [
    "seed","method","checkpoint_name","dataset","task","arch","checkpoint",
    "attack","sparsity","sparsity_percent","epoch","epochs","lr",
    "batch_size","weight_decay","epoch_acc","epoch_loss",
    "best_acc_so_far","best_epoch_so_far",
    "epoch_train_time_sec","epoch_test_time_sec",
    "cumulative_train_time_sec","cumulative_test_time_sec",
    "selection_time_sec",
    "trainable_params","total_params","trainable_percent",
    "targeted_layers","peak_alloc_mb","peak_reserved_mb",
    "gpu_name","missing_keys","unexpected_keys",
]

SEED_COLUMNS = [
    "seed","method","checkpoint_name","dataset","task","arch","checkpoint",
    "attack","sparsity","sparsity_percent","epochs","lr",
    "batch_size","weight_decay",
    "epoch20_acc","epoch20_loss","best_acc","best_epoch",
    "selection_time_sec","train_time_sec","test_time_sec",
    "total_attack_time_sec",
    "trainable_params","total_params","trainable_percent",
    "peak_alloc_mb","peak_reserved_mb",
    "train_examples","test_examples","prediction_file",
    "gpu_name","missing_keys","unexpected_keys",
]

BUDGET_DETAIL_COLUMNS = [
    "seed","method","checkpoint_name","dataset","task","arch",
    "checkpoint","attack","sparsity","sparsity_percent",
    "layer","module_type","tensor_params","selected_params",
    "selected_percent_tensor","fully_trainable",
]


def write_predictions(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "y_true", "y_pred"])
        writer.writerows(rows)


def remove_partial_rows(path, checkpoint_name, sparsity, seed):
    # Remove partial records before rerunning an interrupted configuration.
    if not os.path.exists(path):
        return

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)

    if not columns:
        return

    kept = []
    for r in rows:
        same = (
            r.get("checkpoint_name") == checkpoint_name
            and abs(float(r.get("sparsity", -1)) - float(sparsity)) < 1e-12
            and int(float(r.get("seed", -1))) == int(seed)
        )
        if not same:
            kept.append(r)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(kept)


def save_budget_details(info, sparsity, seed, detail_rows):
    for d in detail_rows:
        tensor_params = int(d["tensor_params"])
        selected = int(d["selected_params"])
        append_csv(
            BUDGET_DETAILS_CSV,
            {
                "seed": seed,
                "method": info["method"],
                "checkpoint_name": info["name"],
                "dataset": info["dataset"],
                "task": info["task"],
                "arch": info["arch"],
                "checkpoint": info["path"],
                "attack": ATTACK,
                "sparsity": sparsity,
                "sparsity_percent": 100.0 * sparsity,
                "layer": d["layer"],
                "module_type": d["module_type"],
                "tensor_params": tensor_params,
                "selected_params": selected,
                "selected_percent_tensor": (
                    100.0 * selected / tensor_params if tensor_params else 0.0
                ),
                "fully_trainable": d["fully_trainable"],
            },
            BUDGET_DETAIL_COLUMNS,
        )


def train_one(info, sparsity, seed, logger):
    # Each checkpoint, budget, and seed starts from a fresh model state.

    train_loader, test_loader, n_train, n_test = get_loaders(info["dataset"])

    model, missing, unexpected = build_model(info)

    (
        masks,
        originals,
        hooks,
        trainable,
        layers,
        detail_rows,
        selection_time,
    ) = apply_sparseft(model, sparsity)

    save_budget_details(info, sparsity, seed, detail_rows)

    # Select the sparse mask before moving the model to the GPU.
    model = model.to(DEVICE)

    reset_cuda_peak()

    # AdamW with a constant learning rate and no scheduler.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    ce = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100.0 * trainable / total_params

    best_acc = -1.0
    best_epoch = -1
    final_acc = None
    final_loss = None
    cumulative_train_time = 0.0
    cumulative_test_time = 0.0

    logger.info(
        "RUN | %s | %s | seed=%d | sparsity=%.0f%% | "
        "lr=%g constant | batch=%d | epochs=%d | "
        "trainable=%d/%d (%.6f%%)",
        info["method"],
        info["name"],
        seed,
        sparsity * 100.0,
        LR,
        BATCH_SIZE,
        EPOCHS,
        trainable,
        total_params,
        trainable_pct,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        train_start = time.perf_counter()

        for x, y in tqdm(
            train_loader,
            desc=(
                f"{info['name']} {ATTACK} "
                f"S{sparsity} seed{seed} E{epoch}"
            ),
            leave=False,
        ):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = ce(logits, y)

            loss.backward()
            optimizer.step()

            # Restore the frozen weights after each optimizer step.
            restore_unselected(masks, originals)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        epoch_train_time = time.perf_counter() - train_start
        cumulative_train_time += epoch_train_time

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        test_start = time.perf_counter()

        # Evaluate the test set after each epoch.
        acc, loss_val = evaluate(model, test_loader)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        epoch_test_time = time.perf_counter() - test_start
        cumulative_test_time += epoch_test_time

        final_acc = acc
        final_loss = loss_val

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        peak_alloc, peak_reserved = peak_memory_mb()

        append_csv(
            FINAL_EPOCH_CSV,
            {
                "seed": seed,
                "method": info["method"],
                "checkpoint_name": info["name"],
                "dataset": info["dataset"],
                "task": info["task"],
                "arch": info["arch"],
                "checkpoint": info["path"],
                "attack": ATTACK,
                "sparsity": sparsity,
                "sparsity_percent": 100.0 * sparsity,
                "epoch": epoch,
                "epochs": EPOCHS,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "weight_decay": WEIGHT_DECAY,
                "epoch_acc": round(acc, 4),
                "epoch_loss": round(loss_val, 6),
                "best_acc_so_far": round(best_acc, 4),
                "best_epoch_so_far": best_epoch,
                "epoch_train_time_sec": round(epoch_train_time, 4),
                "epoch_test_time_sec": round(epoch_test_time, 4),
                "cumulative_train_time_sec": round(
                    cumulative_train_time, 4
                ),
                "cumulative_test_time_sec": round(
                    cumulative_test_time, 4
                ),
                "selection_time_sec": round(selection_time, 6),
                "trainable_params": trainable,
                "total_params": total_params,
                "trainable_percent": round(trainable_pct, 6),
                "targeted_layers": " | ".join(layers),
                "peak_alloc_mb": round(peak_alloc, 4),
                "peak_reserved_mb": round(peak_reserved, 4),
                "gpu_name": gpu_name(),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            },
            EPOCH_COLUMNS,
        )

        logger.info(
            "%s | %s | %s | seed=%d | sparsity=%.2f | "
            "epoch %03d/%d | acc=%.2f%% | loss=%.4f | "
            "best=%.2f%% @ %d | trainable=%.4f%% | "
            "train=%.2fs | test=%.2fs",
            info["method"],
            info["name"],
            ATTACK,
            seed,
            sparsity,
            epoch,
            EPOCHS,
            acc,
            loss_val,
            best_acc,
            best_epoch,
            trainable_pct,
            epoch_train_time,
            epoch_test_time,
        )

    # Save predictions from the final trained model.
    final_acc_check, final_loss_check, prediction_rows = evaluate(
        model, test_loader, return_predictions=True
    )

    if abs(final_acc_check - final_acc) > 1e-9:
        raise RuntimeError(
            f"Final accuracy mismatch: epoch20={final_acc} "
            f"prediction_eval={final_acc_check}"
        )

    pred_path = os.path.join(
        PRED_DIR,
        (
            f"{info['name']}__strong_sparseft__"
            f"b{int(round(sparsity * 100))}__seed{seed}.csv"
        ),
    )
    write_predictions(pred_path, prediction_rows)

    peak_alloc, peak_reserved = peak_memory_mb()

    # Test inference is excluded from attack time.
    total_attack_time = selection_time + cumulative_train_time

    append_csv(
        FINAL_SEED_CSV,
        {
            "seed": seed,
            "method": info["method"],
            "checkpoint_name": info["name"],
            "dataset": info["dataset"],
            "task": info["task"],
            "arch": info["arch"],
            "checkpoint": info["path"],
            "attack": ATTACK,
            "sparsity": sparsity,
            "sparsity_percent": 100.0 * sparsity,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "epoch20_acc": round(final_acc, 4),
            "epoch20_loss": round(final_loss, 6),
            "best_acc": round(best_acc, 4),
            "best_epoch": best_epoch,
            "selection_time_sec": round(selection_time, 6),
            "train_time_sec": round(cumulative_train_time, 4),
            "test_time_sec": round(cumulative_test_time, 4),
            "total_attack_time_sec": round(total_attack_time, 4),
            "trainable_params": trainable,
            "total_params": total_params,
            "trainable_percent": round(trainable_pct, 6),
            "peak_alloc_mb": round(peak_alloc, 4),
            "peak_reserved_mb": round(peak_reserved, 4),
            "train_examples": n_train,
            "test_examples": n_test,
            "prediction_file": pred_path,
            "gpu_name": gpu_name(),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        },
        SEED_COLUMNS,
    )

    logger.info(
        "Result | %s | seed=%d | sparsity=%.0f%% | "
        "epoch20=%.4f%% | best=%.4f%%@%d | "
        "selection=%.4fs | train=%.2fs | attack=%.2fs",
        info["name"],
        seed,
        sparsity * 100.0,
        final_acc,
        best_acc,
        best_epoch,
        selection_time,
        cumulative_train_time,
        total_attack_time,
    )

    for h in hooks:
        h.remove()

    del model, optimizer, masks, originals
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def done_keys():
    done = set()
    for r in read_csv_rows(FINAL_SEED_CSV):
        done.add((
            r["checkpoint_name"],
            round(float(r["sparsity"]), 6),
            int(float(r["seed"])),
        ))
    return done


def rebuild_summary():
    rows = read_csv_rows(FINAL_SEED_CSV)
    groups = {}

    for r in rows:
        key = (
            r["checkpoint_name"],
            round(float(r["sparsity"]), 6),
        )
        groups.setdefault(key, []).append(r)

    summary_rows = []

    for (checkpoint_name, sparsity), vals in groups.items():
        latest = {}
        for v in vals:
            seed = int(float(v["seed"]))
            if seed in FINAL_SEEDS:
                latest[seed] = v

        if set(latest.keys()) != set(FINAL_SEEDS):
            continue

        vals = [latest[s] for s in FINAL_SEEDS]
        accs = np.array(
            [float(v["epoch20_acc"]) for v in vals],
            dtype=float,
        )

        mean = float(accs.mean())
        sd = float(accs.std(ddof=1))
        half = CI_T_DF2 * sd / math.sqrt(len(FINAL_SEEDS))
        base = vals[0]

        def avg(key):
            return float(np.mean([float(v[key]) for v in vals]))

        summary_rows.append({
            "method": base["method"],
            "checkpoint_name": checkpoint_name,
            "dataset": base["dataset"],
            "task": base["task"],
            "arch": base["arch"],
            "checkpoint": base["checkpoint"],
            "attack": ATTACK,
            "sparsity": sparsity,
            "sparsity_percent": 100.0 * sparsity,
            "lr": LR,
            "n_seeds": len(FINAL_SEEDS),
            "seeds": "|".join(str(s) for s in FINAL_SEEDS),
            "mean_test_acc": mean,
            "sample_sd": sd,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "mean_selection_time_sec": avg("selection_time_sec"),
            "mean_train_time_sec": avg("train_time_sec"),
            "mean_test_time_sec": avg("test_time_sec"),
            "mean_total_attack_time_sec": avg("total_attack_time_sec"),
            "mean_peak_alloc_mb": avg("peak_alloc_mb"),
            "mean_peak_reserved_mb": avg("peak_reserved_mb"),
            "trainable_params": int(float(base["trainable_params"])),
            "total_params": int(float(base["total_params"])),
            "trainable_percent": float(base["trainable_percent"]),
            "gpu_name": base["gpu_name"],
        })

    columns = [
        "method","checkpoint_name","dataset","task","arch","checkpoint",
        "attack","sparsity","sparsity_percent","lr",
        "n_seeds","seeds","mean_test_acc","sample_sd",
        "ci95_low","ci95_high",
        "mean_selection_time_sec","mean_train_time_sec",
        "mean_test_time_sec","mean_total_attack_time_sec",
        "mean_peak_alloc_mb","mean_peak_reserved_mb",
        "trainable_params","total_params","trainable_percent",
        "gpu_name",
    ]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(FINAL_SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary_rows)


def setup_logger():
    os.makedirs(OUT_DIR, exist_ok=True)

    logger = logging.getLogger("sparseft")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(message)s")

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
        if not os.path.exists(info["path"]):
            raise FileNotFoundError(info["path"])
        print("Checkpoint:", info["path"])
    print("All selected checkpoints found.")


def select_infos(only):
    if only == "all":
        return CHECKPOINTS

    selected = [
        info for info in CHECKPOINTS
        if info["name"] == only
    ]

    if not selected:
        valid = ", ".join(info["name"] for info in CHECKPOINTS)
        raise ValueError(
            f"Unknown checkpoint '{only}'. Valid values: {valid}"
        )

    return selected


def clear_outputs():
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite:
        clear_outputs()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    infos = select_infos(args.only)
    confirm_paths(infos)

    logger = setup_logger()

    logger.info("SparseFT evaluation")
    logger.info("Sparsities: %s", [int(s * 100) for s in SPARSITIES])
    logger.info("Seeds: %s", FINAL_SEEDS)
    logger.info(
        "Preprocessing: Resize64 -> ToTensor; MNIST uses Grayscale(3)"
    )
    logger.info(
        "Selection: per-layer magnitude threshold for Conv2d and Linear weights; "
        "biases are trainable"
    )
    logger.info(
        "Optimization: AdamW lr=%g constant | wd=%g | "
        "batch=%d | epochs=%d",
        LR,
        WEIGHT_DECAY,
        BATCH_SIZE,
        EPOCHS,
    )
    logger.info("No LR sweep, validation split, or scheduler")
    logger.info(
        "Seeds 42/43/44 are evaluated for every checkpoint and sparsity setting"
    )
    logger.info("Epoch 20 is used for the final result")
    logger.info("GPU: %s", gpu_name())

    done = done_keys()

    for seed in FINAL_SEEDS:
        set_seed(seed)
        for info in infos:
            for sparsity in SPARSITIES:
                key = (
                    info["name"],
                    round(float(sparsity), 6),
                    seed,
                )

                if key in done:
                    logger.info(
                        "Skip | %s | sparsity=%.0f%% | seed=%d",
                        info["name"],
                        sparsity * 100.0,
                        seed,
                    )
                    continue

                # Remove incomplete records before restarting this configuration.
                remove_partial_rows(
                    FINAL_EPOCH_CSV,
                    info["name"],
                    sparsity,
                    seed,
                )
                remove_partial_rows(
                    BUDGET_DETAILS_CSV,
                    info["name"],
                    sparsity,
                    seed,
                )

                train_one(
                    info=info,
                    sparsity=sparsity,
                    seed=seed,
                    logger=logger,
                )

                done.add(key)
                rebuild_summary()

    rebuild_summary()

    print("Done.")
    print("Epoch results :", FINAL_EPOCH_CSV)
    print("Seed results  :", FINAL_SEED_CSV)
    print("Summary       :", FINAL_SUMMARY_CSV)
    print("Budget details:", BUDGET_DETAILS_CSV)
    print("Predictions   :", PRED_DIR)
    print("Log           :", LOG_PATH)


if __name__ == "__main__":
    main()
