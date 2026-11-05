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

DATA_ROOT = "./data"
CKPT_ROOT = "./checkpoints"
OUT_DIR = "./results_clean/autosparse"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
MASK_MODE = "channel"
AE_REPAIR_STRENGTH = 8.0
SAMPLE_BETA = 0.50
HEAD_LR_MULT = 5.0
TRAIN_CLASSIFIER = False
TRAIN_FINAL_HEAD_ONLY = True
AE_RESIDUAL_LAMBDA = 0.02

BUDGETS = [0.01, 0.05, 0.10, 0.15]
FINAL_SEEDS = [42, 43, 44]
CI_T_DF2 = 4.302652729911275

RE_LABEL = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]

CURRENT_SEED = 42
CHANNEL_RATIO = 0.15

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
PROBE_CSV = os.path.join(OUT_DIR, "probe_results.csv")
BUDGET_DETAILS_CSV = os.path.join(OUT_DIR, "budget_details.csv")
PRED_DIR = os.path.join(OUT_DIR, "predictions")
LOG_PATH = os.path.join(OUT_DIR, "autosparse.log")


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


class STL10Mapped(datasets.STL10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, RE_LABEL[target]


def get_loaders(dataset):
    # Dataset preprocessing.
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

    if d == "CIFAR10":
        train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=tf_rgb_train)
        test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=tf_rgb_test)

    elif d == "MNIST":
        train_set = datasets.MNIST(DATA_ROOT, train=True, download=False, transform=tf_mnist)
        test_set = datasets.MNIST(DATA_ROOT, train=False, download=False, transform=tf_mnist)

    elif d == "SVHN":
        train_set = datasets.SVHN(DATA_ROOT, split="train", download=False, transform=tf_rgb_train)
        test_set = datasets.SVHN(DATA_ROOT, split="test", download=False, transform=tf_rgb_test)

    elif d == "STL10":
        train_set = STL10Mapped(DATA_ROOT, split="train", download=False, transform=tf_rgb_train)
        test_set = STL10Mapped(DATA_ROOT, split="test", download=False, transform=tf_rgb_test)

    elif d == "CINIC":
        cinic_root = os.path.join(DATA_ROOT, "CINIC10")
        train_dir = os.path.join(cinic_root, "train")
        valid_dir = os.path.join(cinic_root, "valid")
        test_dir = os.path.join(cinic_root, "test")

        if os.path.exists(train_dir) and os.path.exists(valid_dir):
            train_set = datasets.ImageFolder(train_dir, transform=tf_rgb_train)
            test_set = datasets.ImageFolder(valid_dir, transform=tf_rgb_test)
        elif os.path.exists(train_dir) and os.path.exists(test_dir):
            train_set = datasets.ImageFolder(train_dir, transform=tf_rgb_train)
            test_set = datasets.ImageFolder(test_dir, transform=tf_rgb_test)
        else:
            raise FileNotFoundError(f"CINIC folders not found inside {cinic_root}")

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Use one seeded shuffled training loader for each run.
    g = torch.Generator()
    g.manual_seed(CURRENT_SEED)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        num_workers=WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=(DEVICE == "cuda"),
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
    if info["type"] == "sophon":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif info["type"] == "vgg":
        model = CustomVGG(info["arch"], NUM_CLASSES)
    else:
        raise ValueError(info["type"])

    state = load_state(info["path"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    return model, list(missing), list(unexpected)


def safe_name(name):
    return name.replace(".", "__")


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
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


def build_ae_bank(model, train_loader):
    hooker = GenericConvHook(model)
    model.eval()

    # Use one training batch to determine the activation size of each convolutional layer.
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

        # Initialize each autoencoder reproducibly.
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
    return ae_bank, layer_dims


@torch.no_grad()
def compute_channel_scores(model, ae_bank, score_loader, hooker):
    model.eval()
    ae_bank.eval()
    err_sum = {}
    count = 0

    for i, (x, _) in enumerate(tqdm(score_loader, desc="AE channel scores", leave=False)):
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
            if name not in err_sum:
                err_sum[name] = err.detach().cpu()
            else:
                err_sum[name] += err.detach().cpu()
        count += 1

    scores = {}
    for name, err in err_sum.items():
        err = err / max(count, 1)
        err = err / (err.max() + 1e-8)
        scores[name] = err.float()
    return scores


def reset_cuda_peak():
    if DEVICE == "cuda":
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


def train_ae_damage(model, train_loader, logger, info):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    reset_cuda_peak()
    probe_start = time.perf_counter()

    ae_bank, layer_dims = build_ae_bank(model, train_loader)
    hooker = GenericConvHook(model)
    score_loader = DataLoader(
        train_loader.dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )
    optimizer = torch.optim.AdamW(ae_bank.parameters(), lr=AE_LR, weight_decay=WEIGHT_DECAY)

    ae_train_time = 0.0
    ae_last_loss = 0.0
    ae_batches_total = 0

    for ep in range(1, AE_EPOCHS + 1):
        ae_bank.train()
        total_loss = 0.0
        count = 0
        start = time.perf_counter()
        for i, (x, _) in enumerate(tqdm(train_loader, desc=f"AE probe E{ep}", leave=False)):
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
                continue
            loss = loss / used
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            count += 1

        elapsed = time.perf_counter() - start
        ae_train_time += elapsed
        ae_last_loss = total_loss / max(count, 1)
        ae_batches_total += count
        logger.info(
            "Probe | %s | seed=%d | budget=%.0f%% | epoch=%02d/%d | loss=%.6f | time=%.2fs",
            info["name"], CURRENT_SEED, CHANNEL_RATIO * 100.0,
            ep, AE_EPOCHS, ae_last_loss, elapsed,
        )

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    score_start = time.perf_counter()
    scores = compute_channel_scores(model, ae_bank, score_loader, hooker)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    score_time = time.perf_counter() - score_start

    hooker.remove()
    probe_time = time.perf_counter() - probe_start
    probe_peak_alloc_mb, probe_peak_reserved_mb = peak_memory_mb()

    return ae_bank, scores, len(layer_dims), {
        "ae_train_time_sec": ae_train_time,
        "score_time_sec": score_time,
        "probe_time_sec": probe_time,
        "probe_peak_alloc_mb": probe_peak_alloc_mb,
        "probe_peak_reserved_mb": probe_peak_reserved_mb,
        "ae_last_loss": ae_last_loss,
        "ae_batches_used": ae_batches_total,
    }


def make_ae_mask(weight, channel_score):
    with torch.no_grad():
        out_c = weight.shape[0]
        if channel_score is None:
            score = torch.ones(out_c, device=weight.device)
        else:
            score = channel_score[:out_c].to(weight.device)
            if score.numel() < out_c:
                pad = torch.ones(out_c - score.numel(), device=weight.device) * score.mean().clamp_min(1e-8)
                score = torch.cat([score, pad], dim=0)

        if MASK_MODE == "channel":
            k = max(1, int(out_c * CHANNEL_RATIO))
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

    masks, originals = {}, {}
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

    selection_start = time.perf_counter()

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            mask = make_ae_mask(module.weight.data, channel_scores.get(name, None))
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

            module.weight.register_hook(hook_fn)
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
                "layer": name,
                "module_type": "Conv2d",
                "out_channels": module.out_channels,
                "selected_channels": selected_channels,
                "weight_tensor_params": module.weight.numel(),
                "selected_weight_params": selected,
                "bias_params": bias_params,
                "is_final_head": 0,
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
                "layer": name,
                "module_type": "Linear",
                "out_channels": module.out_features,
                "selected_channels": module.out_features,
                "weight_tensor_params": module.weight.numel(),
                "selected_weight_params": module.weight.numel(),
                "bias_params": bias_params,
                "is_final_head": 1,
            })

    selection_time = time.perf_counter() - selection_start

    return masks, originals, effective, trained_conv_layers, {
        "selected_conv_weight_params": selected_conv_weight_params,
        "trainable_conv_bias_params": trainable_conv_bias_params,
        "final_head_params": final_head_params,
        "selected_channels_total": selected_channels_total,
        "candidate_channels_total": candidate_channels_total,
        "selection_time_sec": selection_time,
        "detail_rows": detail_rows,
    }


def restore_unselected(masks, originals):
    with torch.no_grad():
        for p, m in masks.items():
            p.data.copy_(p.data * m + originals[p] * (1.0 - m))


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
                rows.append({
                    "sample_index": sample_index,
                    "y_true": int(yt),
                    "y_pred": int(yp),
                })
                sample_index += 1

    acc = 100.0 * correct / total
    avg_loss = loss_sum / len(loader)

    if return_predictions:
        return acc, avg_loss, pd.DataFrame(rows)
    return acc, avg_loss


def append_csv(path, row, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in columns})


EPOCH_COLUMNS = [
    "method","checkpoint_name","dataset","task","arch","checkpoint",
    "attack","seed","budget","budget_percent","epoch","epochs",
    "lr","head_lr","batch_size","weight_decay",
    "epoch_acc","epoch_loss","best_acc_so_far","best_epoch_so_far",
    "epoch_train_time_sec","epoch_test_time_sec","cumulative_train_time_sec",
    "effective_trainable","total_params","effective_percent",
    "selected_conv_weight_params","trainable_conv_bias_params","final_head_params",
    "selected_channels_total","candidate_channels_total","num_conv_layers_scored",
    "num_conv_layers_trained","selection_time_sec","gpu_name",
    "missing_keys","unexpected_keys",
]

SEED_COLUMNS = [
    "method","checkpoint_name","dataset","task","arch","checkpoint",
    "attack","seed","budget","budget_percent","epoch20_acc","epoch20_loss",
    "best_acc","best_epoch","lr","head_lr","batch_size","weight_decay","epochs",
    "ae_train_time_sec","score_time_sec","probe_time_sec","selection_time_sec",
    "train_time_sec","test_time_sec","total_attack_time_sec",
    "probe_peak_alloc_mb","probe_peak_reserved_mb",
    "recovery_peak_alloc_mb","recovery_peak_reserved_mb",
    "peak_attack_alloc_mb","peak_attack_reserved_mb",
    "effective_trainable","total_params","effective_percent",
    "selected_conv_weight_params","trainable_conv_bias_params","final_head_params",
    "selected_channels_total","candidate_channels_total","num_conv_layers_scored",
    "num_conv_layers_trained","train_examples","test_examples","prediction_file",
    "gpu_name","missing_keys","unexpected_keys",
]

PROBE_COLUMNS = [
    "method","checkpoint_name","dataset","task","arch","checkpoint",
    "seed","budget","budget_percent","ae_epochs","ae_lr","ae_max_batches",
    "score_batches","ae_batches_used","ae_last_loss","num_conv_layers",
    "ae_train_time_sec","score_time_sec","probe_time_sec",
    "probe_peak_alloc_mb","probe_peak_reserved_mb","gpu_name",
]

BUDGET_DETAIL_COLUMNS = [
    "method","checkpoint_name","dataset","task","arch","checkpoint",
    "seed","budget","budget_percent","layer","module_type","out_channels",
    "selected_channels","weight_tensor_params","selected_weight_params",
    "bias_params","is_final_head",
]


def gpu_name():
    return torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"


def write_budget_details(info, seed, budget, detail_rows):
    for d in detail_rows:
        append_csv(BUDGET_DETAILS_CSV, {
            "method": info["method"],
            "checkpoint_name": info["name"],
            "dataset": info["dataset"],
            "task": info["task"],
            "arch": info["arch"],
            "checkpoint": info["path"],
            "seed": seed,
            "budget": budget,
            "budget_percent": 100.0 * budget,
            **d,
        }, BUDGET_DETAIL_COLUMNS)


def train_recovery(info, model, missing, unexpected, channel_scores, num_scored_layers,
                   train_loader, test_loader, ae_bank, probe_stats, n_train, n_test, logger):
    # Continue with the same loader and RNG state used during probing.
    reset_cuda_peak()

    masks, originals, effective, trained_conv_layers, mask_stats = apply_masks(model, channel_scores)
    model = model.to(DEVICE)
    ae_bank = ae_bank.to(DEVICE)
    ae_bank.eval()
    for p in ae_bank.parameters():
        p.requires_grad = False

    write_budget_details(info, CURRENT_SEED, CHANNEL_RATIO, mask_stats["detail_rows"])

    train_hook = TrainConvHook(model)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * effective / total

    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".fc." in name or name.startswith("fc.") or "classifier" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_STAGE1},
        {"params": head_params, "lr": LR_STAGE1 * HEAD_LR_MULT},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine learning-rate schedule for recovery.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR_STAGE1 * 0.05,
    )

    best_acc = -1.0
    best_epoch = -1
    final_acc = None
    final_loss = None
    cumulative_train_time = 0.0
    cumulative_test_time = 0.0

    logger.info(
        "Recovery | %s | seed=%d | budget=%.0f%% | trainable=%d/%d (%.6f%%) | conv=%d/%d",
        info["name"], CURRENT_SEED, CHANNEL_RATIO * 100.0,
        effective, total, pct, num_scored_layers, trained_conv_layers,
    )

    for epoch in range(1, EPOCHS + 1):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        train_start = time.perf_counter()
        model.train()

        for x, y in tqdm(
            train_loader,
            desc=f"{info['name']} AutoSparse B{int(CHANNEL_RATIO*100)} S{CURRENT_SEED} E{epoch}",
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

            loss_each = F.cross_entropy(logits, y, reduction="none")
            loss = (loss_each * sw).mean()
            if used > 0 and AE_RESIDUAL_LAMBDA > 0.0:
                loss = loss + AE_RESIDUAL_LAMBDA * ae_loss

            loss.backward()
            optimizer.step()
            restore_unselected(masks, originals)

        scheduler.step()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        epoch_train_time = time.perf_counter() - train_start
        cumulative_train_time += epoch_train_time

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        test_start = time.perf_counter()
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

        append_csv(FINAL_EPOCH_CSV, {
            "method": info["method"],
            "checkpoint_name": info["name"],
            "dataset": info["dataset"],
            "task": info["task"],
            "arch": info["arch"],
            "checkpoint": info["path"],
            "attack": "autosparse",
            "seed": CURRENT_SEED,
            "budget": CHANNEL_RATIO,
            "budget_percent": 100.0 * CHANNEL_RATIO,
            "epoch": epoch,
            "epochs": EPOCHS,
            "lr": LR_STAGE1,
            "head_lr": LR_STAGE1 * HEAD_LR_MULT,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "epoch_acc": round(acc, 4),
            "epoch_loss": round(loss_val, 6),
            "best_acc_so_far": round(best_acc, 4),
            "best_epoch_so_far": best_epoch,
            "epoch_train_time_sec": round(epoch_train_time, 4),
            "epoch_test_time_sec": round(epoch_test_time, 4),
            "cumulative_train_time_sec": round(cumulative_train_time, 4),
            "effective_trainable": effective,
            "total_params": total,
            "effective_percent": round(pct, 6),
            "selected_conv_weight_params": mask_stats["selected_conv_weight_params"],
            "trainable_conv_bias_params": mask_stats["trainable_conv_bias_params"],
            "final_head_params": mask_stats["final_head_params"],
            "selected_channels_total": mask_stats["selected_channels_total"],
            "candidate_channels_total": mask_stats["candidate_channels_total"],
            "num_conv_layers_scored": num_scored_layers,
            "num_conv_layers_trained": trained_conv_layers,
            "selection_time_sec": round(mask_stats["selection_time_sec"], 6),
            "gpu_name": gpu_name(),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        }, EPOCH_COLUMNS)

        logger.info(
            "AutoSparse | %s | seed=%d | budget=%.0f%% | epoch=%02d/%d | "
            "acc=%.2f%% | loss=%.4f | best=%.2f%%@%d | train=%.1fs | test=%.2fs",
            info["name"], CURRENT_SEED, CHANNEL_RATIO * 100.0,
            epoch, EPOCHS, acc, loss_val, best_acc, best_epoch,
            epoch_train_time, epoch_test_time,
        )

    # Save predictions from the epoch-20 model.
    final_acc2, final_loss2, pred_df = evaluate(model, test_loader, return_predictions=True)
    if abs(final_acc2 - final_acc) > 1e-9:
        raise RuntimeError(f"Final evaluation mismatch: {final_acc} vs {final_acc2}")

    os.makedirs(PRED_DIR, exist_ok=True)
    pred_path = os.path.join(
        PRED_DIR,
        f"{info['name']}__autosparse__b{int(round(CHANNEL_RATIO*100))}__seed{CURRENT_SEED}.csv",
    )
    pred_df.to_csv(pred_path, index=False)

    recovery_peak_alloc_mb, recovery_peak_reserved_mb = peak_memory_mb()
    peak_attack_alloc_mb = max(probe_stats["probe_peak_alloc_mb"], recovery_peak_alloc_mb)
    peak_attack_reserved_mb = max(probe_stats["probe_peak_reserved_mb"], recovery_peak_reserved_mb)

    total_attack_time = (
        probe_stats["probe_time_sec"]
        + mask_stats["selection_time_sec"]
        + cumulative_train_time
    )

    append_csv(FINAL_SEED_CSV, {
        "method": info["method"],
        "checkpoint_name": info["name"],
        "dataset": info["dataset"],
        "task": info["task"],
        "arch": info["arch"],
        "checkpoint": info["path"],
        "attack": "autosparse",
        "seed": CURRENT_SEED,
        "budget": CHANNEL_RATIO,
        "budget_percent": 100.0 * CHANNEL_RATIO,
        "epoch20_acc": round(final_acc, 4),
        "epoch20_loss": round(final_loss, 6),
        "best_acc": round(best_acc, 4),
        "best_epoch": best_epoch,
        "lr": LR_STAGE1,
        "head_lr": LR_STAGE1 * HEAD_LR_MULT,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "ae_train_time_sec": round(probe_stats["ae_train_time_sec"], 4),
        "score_time_sec": round(probe_stats["score_time_sec"], 4),
        "probe_time_sec": round(probe_stats["probe_time_sec"], 4),
        "selection_time_sec": round(mask_stats["selection_time_sec"], 6),
        "train_time_sec": round(cumulative_train_time, 4),
        "test_time_sec": round(cumulative_test_time, 4),
        "total_attack_time_sec": round(total_attack_time, 4),
        "probe_peak_alloc_mb": round(probe_stats["probe_peak_alloc_mb"], 4),
        "probe_peak_reserved_mb": round(probe_stats["probe_peak_reserved_mb"], 4),
        "recovery_peak_alloc_mb": round(recovery_peak_alloc_mb, 4),
        "recovery_peak_reserved_mb": round(recovery_peak_reserved_mb, 4),
        "peak_attack_alloc_mb": round(peak_attack_alloc_mb, 4),
        "peak_attack_reserved_mb": round(peak_attack_reserved_mb, 4),
        "effective_trainable": effective,
        "total_params": total,
        "effective_percent": round(pct, 6),
        "selected_conv_weight_params": mask_stats["selected_conv_weight_params"],
        "trainable_conv_bias_params": mask_stats["trainable_conv_bias_params"],
        "final_head_params": mask_stats["final_head_params"],
        "selected_channels_total": mask_stats["selected_channels_total"],
        "candidate_channels_total": mask_stats["candidate_channels_total"],
        "num_conv_layers_scored": num_scored_layers,
        "num_conv_layers_trained": trained_conv_layers,
        "train_examples": n_train,
        "test_examples": n_test,
        "prediction_file": pred_path,
        "gpu_name": gpu_name(),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }, SEED_COLUMNS)

    logger.info(
        "Result | %s | seed=%d | budget=%.0f%% | epoch20=%.4f%% | best=%.4f%%@%d | "
        "probe=%.2fs | train=%.2fs | attack=%.2fs",
        info["name"], CURRENT_SEED, CHANNEL_RATIO * 100.0,
        final_acc, best_acc, best_epoch,
        probe_stats["probe_time_sec"], cumulative_train_time, total_attack_time,
    )

    train_hook.remove()


def run_one(info, seed, budget, logger):
    global CURRENT_SEED, CHANNEL_RATIO

    CURRENT_SEED = seed
    CHANNEL_RATIO = budget

    # Run each checkpoint, budget, and seed independently.
    set_seed(CURRENT_SEED)
    train_loader, test_loader, n_train, n_test = get_loaders(info["dataset"])

    model, missing, unexpected = build_model(info)
    model = model.to(DEVICE)

    ae_bank, channel_scores, num_scored_layers, probe_stats = train_ae_damage(
        model, train_loader, logger, info
    )

    append_csv(PROBE_CSV, {
        "method": info["method"],
        "checkpoint_name": info["name"],
        "dataset": info["dataset"],
        "task": info["task"],
        "arch": info["arch"],
        "checkpoint": info["path"],
        "seed": seed,
        "budget": budget,
        "budget_percent": 100.0 * budget,
        "ae_epochs": AE_EPOCHS,
        "ae_lr": AE_LR,
        "ae_max_batches": AE_MAX_BATCHES,
        "score_batches": SCORE_BATCHES,
        "ae_batches_used": probe_stats["ae_batches_used"],
        "ae_last_loss": probe_stats["ae_last_loss"],
        "num_conv_layers": num_scored_layers,
        "ae_train_time_sec": round(probe_stats["ae_train_time_sec"], 4),
        "score_time_sec": round(probe_stats["score_time_sec"], 4),
        "probe_time_sec": round(probe_stats["probe_time_sec"], 4),
        "probe_peak_alloc_mb": round(probe_stats["probe_peak_alloc_mb"], 4),
        "probe_peak_reserved_mb": round(probe_stats["probe_peak_reserved_mb"], 4),
        "gpu_name": gpu_name(),
    }, PROBE_COLUMNS)

    # Reload the hardened checkpoint before recovery without reseeding.
    model, missing, unexpected = build_model(info)
    model = model.to(DEVICE)

    train_recovery(
        info,
        model,
        missing,
        unexpected,
        channel_scores,
        num_scored_layers,
        train_loader,
        test_loader,
        ae_bank,
        probe_stats,
        n_train,
        n_test,
        logger,
    )

    del model, ae_bank, channel_scores
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def load_seed_rows():
    if not os.path.exists(FINAL_SEED_CSV):
        return []
    return pd.read_csv(FINAL_SEED_CSV).to_dict("records")


def done_keys():
    return {
        (
            str(r["checkpoint_name"]),
            round(float(r["budget"]), 6),
            int(r["seed"]),
        )
        for r in load_seed_rows()
    }


def rebuild_summary():
    rows = load_seed_rows()
    groups = {}

    for r in rows:
        key = (
            str(r["checkpoint_name"]),
            round(float(r["budget"]), 6),
        )
        groups.setdefault(key, []).append(r)

    out = []

    for (name, budget), vals in groups.items():
        latest = {}
        for v in vals:
            seed = int(v["seed"])
            if seed in FINAL_SEEDS:
                latest[seed] = v

        if set(latest) != set(FINAL_SEEDS):
            continue

        vals = [latest[s] for s in FINAL_SEEDS]
        acc = np.array([float(v["epoch20_acc"]) for v in vals], dtype=float)
        mean = float(acc.mean())
        sd = float(acc.std(ddof=1))
        half = CI_T_DF2 * sd / math.sqrt(len(FINAL_SEEDS))
        base = vals[0]

        def avg(key):
            return float(np.mean([float(v[key]) for v in vals]))

        out.append({
            "method": base["method"],
            "checkpoint_name": base["checkpoint_name"],
            "dataset": base["dataset"],
            "task": base["task"],
            "arch": base["arch"],
            "checkpoint": base["checkpoint"],
            "attack": "autosparse",
            "budget": budget,
            "budget_percent": 100.0 * budget,
            "lr": LR_STAGE1,
            "head_lr": LR_STAGE1 * HEAD_LR_MULT,
            "n_seeds": len(FINAL_SEEDS),
            "seeds": "|".join(str(s) for s in FINAL_SEEDS),
            "mean_test_acc": mean,
            "sample_sd": sd,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "mean_probe_time_sec": avg("probe_time_sec"),
            "mean_selection_time_sec": avg("selection_time_sec"),
            "mean_train_time_sec": avg("train_time_sec"),
            "mean_total_attack_time_sec": avg("total_attack_time_sec"),
            "mean_peak_attack_alloc_mb": avg("peak_attack_alloc_mb"),
            "effective_trainable": int(float(base["effective_trainable"])),
            "total_params": int(float(base["total_params"])),
            "effective_percent": float(base["effective_percent"]),
            "selected_conv_weight_params": int(float(base["selected_conv_weight_params"])),
            "trainable_conv_bias_params": int(float(base["trainable_conv_bias_params"])),
            "final_head_params": int(float(base["final_head_params"])),
            "selected_channels_total": int(float(base["selected_channels_total"])),
            "candidate_channels_total": int(float(base["candidate_channels_total"])),
            "num_conv_layers_scored": int(float(base["num_conv_layers_scored"])),
            "num_conv_layers_trained": int(float(base["num_conv_layers_trained"])),
            "gpu_name": base["gpu_name"],
        })

    pd.DataFrame(out).to_csv(FINAL_SUMMARY_CSV, index=False)


def setup_logger():
    os.makedirs(OUT_DIR, exist_ok=True)
    logger = logging.getLogger("autosparse")
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
    selected = [x for x in CHECKPOINTS if x["name"] == only]
    if not selected:
        valid = ", ".join(x["name"] for x in CHECKPOINTS)
        raise ValueError(f"Unknown checkpoint {only}. Valid: {valid}")
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

    logger.info("AutoSparse evaluation")
    logger.info("Budgets: %s", [int(b * 100) for b in BUDGETS])
    logger.info("Seeds: %s", FINAL_SEEDS)
    logger.info(
        "Preprocessing: RGB train Resize72 -> RandomCrop64 -> RandomHorizontalFlip -> "
        "ToTensor -> ImageNet Normalize; RGB test Resize64 -> ToTensor -> Normalize"
    )
    logger.info(
        "AE: epochs=%d | lr=%g | max_batches=%d | score_batches=%d",
        AE_EPOCHS, AE_LR, AE_MAX_BATCHES, SCORE_BATCHES,
    )
    logger.info(
        "Recovery: AdamW backbone_lr=%g | head_lr=%g | wd=%g | batch=%d | epochs=%d | "
        "cosine eta_min=%g",
        LR_STAGE1, LR_STAGE1 * HEAD_LR_MULT, WEIGHT_DECAY,
        BATCH_SIZE, EPOCHS, LR_STAGE1 * 0.05,
    )
    logger.info(
        "Recovery components: sample_beta=%g | AE residual lambda=%g",
        SAMPLE_BETA, AE_RESIDUAL_LAMBDA,
    )
    logger.info(
        "Runs are independent across checkpoints, budgets, and seeds; "
        "recovery continues with the same seeded training loader"
    )
    logger.info("The test set is evaluated after each epoch; epoch 20 is used for the final result")
    logger.info("GPU: %s", gpu_name())

    done = done_keys()

    for info in infos:
        for budget in BUDGETS:
            for seed in FINAL_SEEDS:
                key = (info["name"], round(float(budget), 6), seed)

                if key in done:
                    logger.info(
                        "Skip | %s | budget=%.0f%% | seed=%d",
                        info["name"], budget * 100.0, seed,
                    )
                    continue

                logger.info(
                    "Run | %s | budget=%.0f%% | seed=%d",
                    info["name"], budget * 100.0, seed,
                )

                run_one(info, seed, budget, logger)
                done.add(key)
                rebuild_summary()

    rebuild_summary()

    print("Done.")
    print("Epoch results :", FINAL_EPOCH_CSV)
    print("Seed results  :", FINAL_SEED_CSV)
    print("Summary       :", FINAL_SUMMARY_CSV)
    print("Probe results :", PROBE_CSV)
    print("Budget details:", BUDGET_DETAILS_CSV)
    print("Predictions   :", PRED_DIR)
    print("Log           :", LOG_PATH)


if __name__ == "__main__":
    main()
