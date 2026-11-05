import os
import gc
import csv
import math
import time
import argparse
import logging
import shutil
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import ablation as ab
import lora as lo
import dora as do


# AutoSparse-guided LoRA/DoRA hybrid
# Fixed full-checkpoint experiment: 10% AutoSparse channels + rank 16.

OUT_DIR = './results_clean/fusion10_r16'
LOG_PATH = os.path.join(OUT_DIR, 'fusion.log')
SEED_CSV = os.path.join(OUT_DIR, 'final_seed_results.csv')
EPOCH_CSV = os.path.join(OUT_DIR, 'final_epoch_results.csv')
SUMMARY_CSV = os.path.join(OUT_DIR, 'final_summary.csv')
SEEDS = [42, 43, 44]
ATTACKS = ['lora', 'dora']
EXPERIMENT_TAG = 'FULL13_B10_R16_VGGFIX'
BUDGET = 0.10
RANK = 16
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
T95_DF2 = 4.302652729911275
DEVICE = ab.DEVICE

TARGETS = [x['name'] for x in ab.CHECKPOINTS]

# Checkpoints used in the experiments.
# Paths are checked by suffix so CKPT_ROOT may be relative or absolute.
EXPECTED_CHECKPOINTS = {
    'sophon_il_cifar10': ('SOPHON-IL', 'sophon', 'resnet18', 'CIFAR10', 'checkpoints/sophon_il/sophon_il_cifar10.pth'),
    'sophon_il_cinic': ('SOPHON-IL', 'sophon', 'resnet18', 'CINIC', 'checkpoints/sophon_il/sophon_il_cinic.pth'),
    'sophon_il_mnist': ('SOPHON-IL', 'sophon', 'resnet18', 'MNIST', 'checkpoints/sophon_il/sophon_il_mnist.pth'),
    'sophon_il_stl10': ('SOPHON-IL', 'sophon', 'resnet18', 'STL10', 'checkpoints/sophon_il/sophon_il_stl10.pth'),
    'sophon_il_svhn': ('SOPHON-IL', 'sophon', 'resnet18', 'SVHN', 'checkpoints/sophon_il/sophon_il_svhn.pth'),
    'sophon_kl_cifar10': ('SOPHON-KL', 'sophon', 'resnet18', 'CIFAR10', 'checkpoints/sophon_kl/sophon_kl_cifar10.pth'),
    'sophon_kl_cinic': ('SOPHON-KL', 'sophon', 'resnet18', 'CINIC', 'checkpoints/sophon_kl/sophon_kl_cinic.pth'),
    'sophon_kl_mnist': ('SOPHON-KL', 'sophon', 'resnet18', 'MNIST', 'checkpoints/sophon_kl/sophon_kl_mnist.pth'),
    'sophon_kl_stl10': ('SOPHON-KL', 'sophon', 'resnet18', 'STL10', 'checkpoints/sophon_kl/sophon_kl_stl10.pth'),
    'sophon_kl_svhn': ('SOPHON-KL', 'sophon', 'resnet18', 'SVHN', 'checkpoints/sophon_kl/sophon_kl_svhn.pth'),
    'hntl_cifar_stl_vgg13': ('HNTL', 'vgg', 'vgg13', 'STL10', 'checkpoints/hntl_cifar_stl_vgg13.pth'),
    'cupi_cifar_stl_vgg13': ('CUPI', 'vgg', 'vgg13', 'STL10', 'checkpoints/cupi_cifar_stl_vgg13.pth'),
    'cuti_cifar_stl_vgg19': ('CUTI', 'vgg', 'vgg19', 'STL10', 'checkpoints/cuti_cifar_stl_vgg19_ntlbench.pth'),
}


def gpu_name():
    return torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'


def setup_logger():
    os.makedirs(OUT_DIR, exist_ok=True)
    logger = logging.getLogger('fusion')
    logger.handlers = []
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(message)s')
    fh = logging.FileHandler(LOG_PATH)
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def append_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed):
    ab.set_seed(seed)
    ab.CURRENT_SEED = seed
    ab.CHANNEL_RATIO = BUDGET
    ab.CURRENT_SELECTOR = 'fusion'


def reset_peak():
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_mb():
    if DEVICE != 'cuda':
        return 0.0, 0.0
    torch.cuda.synchronize()
    return (
        torch.cuda.max_memory_allocated() / 1024**2,
        torch.cuda.max_memory_reserved() / 1024**2,
    )


def alpha_for(info, attack):
    if attack == 'lora':
        return lo.LORA_ALPHA_SOPHON if info['type'] == 'sophon' else lo.LORA_ALPHA_VGG
    if attack == 'dora':
        return do.DORA_ALPHA_SOPHON if info['type'] == 'sophon' else do.DORA_ALPHA_VGG
    raise ValueError(attack)


def selected_rows_from_score(score, out_channels):
    score = score.detach().float().cpu()
    if score.numel() != out_channels:
        raise RuntimeError(f'AE score mismatch: {score.numel()} vs {out_channels}')
    k = max(1, int(out_channels * BUDGET))
    k = min(k, out_channels)
    idx = torch.topk(score, k=k, largest=True).indices
    mask = torch.zeros(out_channels, dtype=torch.float32)
    mask[idx] = 1.0
    return mask, k


# Masked LoRA

class MaskedLoRAConv2d(lo.LoRAConv2d):
    def __init__(self, conv, rank, alpha, selected_rows):
        if conv.groups != 1:
            raise RuntimeError('Hybrid currently supports groups=1 Conv2d only.')
        super().__init__(conv, rank, alpha)
        m = selected_rows.detach().float().view(-1, 1)
        if m.shape[0] != conv.out_channels:
            raise RuntimeError('Selected-row mask does not match Conv2d out_channels.')
        self.register_buffer('selected_rows', m)

    def forward(self, x):
        B = self.B * self.selected_rows
        delta = (B @ self.A).view(
            self.conv.out_channels,
            self.conv.in_channels,
            self.conv.kernel_size[0],
            self.conv.kernel_size[1],
        )
        weight = self.conv.weight + self.scale * delta
        return F.conv2d(
            x,
            weight,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )


# Masked DoRA
# Unselected Conv rows are forced to the original hardened base weight.

class MaskedDoRAConv2d(do.DoRAConv2d):
    def __init__(self, conv, rank, alpha, selected_rows):
        if conv.groups != 1:
            raise RuntimeError('Hybrid currently supports groups=1 Conv2d only.')
        super().__init__(conv, rank, alpha)
        m = selected_rows.detach().float().view(-1, 1)
        if m.shape[0] != conv.out_channels:
            raise RuntimeError('Selected-row mask does not match Conv2d out_channels.')
        self.register_buffer('selected_rows', m)

    def forward(self, x):
        out_ch = self.conv.out_channels
        base = self.conv.weight.view(out_ch, -1)

        B = self.B * self.selected_rows
        delta = B @ self.A
        adapted = base + self.scale * delta
        direction = adapted / (adapted.norm(dim=1, keepdim=True) + 1e-6)
        dora_weight = self.magnitude.unsqueeze(1) * direction

        # Only selected output rows use DoRA. Others are exactly the base rows.
        weight_flat = (
            base * (1.0 - self.selected_rows)
            + dora_weight * self.selected_rows
        )
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


def inject_guided_lora(module, scores, rank, alpha, prefix=''):
    conv_layers = linear_layers = selected_total = candidate_total = 0
    effective_adapter = physical_adapter = 0
    targeted = []

    for name, child in list(module.named_children()):
        full = f'{prefix}.{name}' if prefix else name

        if isinstance(child, nn.Conv2d):
            if full not in scores:
                raise RuntimeError(f'Missing AutoSparse AE score for {full}')
            mask, k = selected_rows_from_score(scores[full], child.out_channels)
            wrapped = MaskedLoRAConv2d(child, rank, alpha, mask)
            setattr(module, name, wrapped)

            conv_layers += 1
            selected_total += k
            candidate_total += child.out_channels
            physical_adapter += wrapped.A.numel() + wrapped.B.numel()
            effective_adapter += wrapped.A.numel() + k * rank
            targeted.append(f'{full}[{k}/{child.out_channels}]')

        elif isinstance(child, nn.Linear):
            wrapped = lo.LoRALinear(child, rank, alpha)
            setattr(module, name, wrapped)

            linear_layers += 1
            n = wrapped.A.numel() + wrapped.B.numel()
            physical_adapter += n
            effective_adapter += n
            targeted.append(full)

        else:
            c, l, s, t, e, p, names = inject_guided_lora(
                child, scores, rank, alpha, full
            )
            conv_layers += c
            linear_layers += l
            selected_total += s
            candidate_total += t
            effective_adapter += e
            physical_adapter += p
            targeted.extend(names)

    return (
        conv_layers,
        linear_layers,
        selected_total,
        candidate_total,
        effective_adapter,
        physical_adapter,
        targeted,
    )


def inject_guided_dora(module, scores, rank, alpha, prefix=''):
    conv_layers = linear_layers = selected_total = candidate_total = 0
    effective_adapter = physical_adapter = 0
    targeted = []

    for name, child in list(module.named_children()):
        full = f'{prefix}.{name}' if prefix else name

        if isinstance(child, nn.Conv2d):
            if full not in scores:
                raise RuntimeError(f'Missing AutoSparse AE score for {full}')
            mask, k = selected_rows_from_score(scores[full], child.out_channels)
            wrapped = MaskedDoRAConv2d(child, rank, alpha, mask)
            setattr(module, name, wrapped)

            conv_layers += 1
            selected_total += k
            candidate_total += child.out_channels

            physical_adapter += (
                wrapped.A.numel()
                + wrapped.B.numel()
                + wrapped.magnitude.numel()
            )
            # Shared A plus selected B rows and selected magnitudes.
            effective_adapter += wrapped.A.numel() + k * rank + k
            targeted.append(f'{full}[{k}/{child.out_channels}]')

        elif isinstance(child, nn.Linear):
            wrapped = do.DoRALinear(child, rank, alpha)
            setattr(module, name, wrapped)

            linear_layers += 1
            n = (
                wrapped.A.numel()
                + wrapped.B.numel()
                + wrapped.magnitude.numel()
            )
            physical_adapter += n
            effective_adapter += n
            targeted.append(full)

        else:
            c, l, s, t, e, p, names = inject_guided_dora(
                child, scores, rank, alpha, full
            )
            conv_layers += c
            linear_layers += l
            selected_total += s
            candidate_total += t
            effective_adapter += e
            physical_adapter += p
            targeted.extend(names)

    return (
        conv_layers,
        linear_layers,
        selected_total,
        candidate_total,
        effective_adapter,
        physical_adapter,
        targeted,
    )


@torch.no_grad()
def evaluate(model, loader, predictions=False):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    correct = total = 0
    loss_sum = 0.0
    rows = []
    sample_index = 0

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        out = model(x)
        loss_sum += ce(out, y).item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        if predictions:
            for yt, yp in zip(y.cpu().tolist(), pred.cpu().tolist()):
                rows.append({
                    'sample_index': sample_index,
                    'y_true': int(yt),
                    'y_pred': int(yp),
                })
                sample_index += 1

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    acc = 100.0 * correct / max(total, 1)
    loss = loss_sum / max(total, 1)
    return acc, loss, elapsed, pd.DataFrame(rows)


def build_attack_loaders(info, seed, attack):
    """Use the LoRA/DoRA data pipeline for adaptation.

    AutoSparse's own loader is intentionally used only for the AE probing stage.
    This uses AutoSparse for channel selection while keeping the actual
    PEFT adaptation directly comparable with the standalone LoRA/DoRA runs.
    """
    module = lo if attack == 'lora' else do
    train_set, test_set = module.get_dataset_pair(info['dataset'])
    train_loader = module.make_train_loader(train_set, seed)
    test_loader = module.make_eval_loader(test_set)
    return train_loader, test_loader, len(train_set), len(test_set)


def get_infos(only):
    infos = [x.copy() for x in ab.CHECKPOINTS]
    if only != 'all':
        infos = [x for x in infos if x['name'] == only]

    expected = len(ab.CHECKPOINTS) if only == 'all' else 1
    if len(infos) != expected:
        raise RuntimeError(
            f'Could not resolve requested fusion target(s): {only}; '
            f'expected={expected}, found={len(infos)}'
        )
    return infos


def normalize_vgg_state_to_model(state, model):
    """
    Normalize VGG classifier parameter indices to the classifier layout used by
    the instantiated model.

    Protected VGG checkpoints may store the three Linear layers at 0/3/6,
    while the attack model represents the same three Linear layers at 0/2/4.
    The three Linear layers are mapped positionally after verifying shapes.
    Tensor values are not changed.
    """
    target = model.state_dict()

    src_weights = []
    for key, value in state.items():
        m = re.fullmatch(r'classifier\.(\d+)\.weight', key)
        if m:
            src_weights.append((int(m.group(1)), key, value))
    src_weights.sort()

    tgt_weights = []
    for key, value in target.items():
        m = re.fullmatch(r'classifier\.(\d+)\.weight', key)
        if m:
            tgt_weights.append((int(m.group(1)), key, value))
    tgt_weights.sort()

    if len(src_weights) != len(tgt_weights):
        raise RuntimeError(
            f'VGG classifier Linear-count mismatch: '
            f'checkpoint={len(src_weights)}, model={len(tgt_weights)}'
        )

    remapped = {}
    classifier_source_keys = set()

    for (src_idx, src_w_key, src_w), (tgt_idx, tgt_w_key, tgt_w) in zip(
        src_weights, tgt_weights
    ):
        src_b_key = f'classifier.{src_idx}.bias'
        tgt_b_key = f'classifier.{tgt_idx}.bias'

        if tuple(src_w.shape) != tuple(tgt_w.shape):
            raise RuntimeError(
                f'VGG classifier weight-shape mismatch: '
                f'{src_w_key}{tuple(src_w.shape)} -> '
                f'{tgt_w_key}{tuple(tgt_w.shape)}'
            )

        remapped[tgt_w_key] = src_w
        classifier_source_keys.add(src_w_key)

        if src_b_key in state:
            if tgt_b_key not in target:
                raise RuntimeError(f'Missing target bias key: {tgt_b_key}')
            if tuple(state[src_b_key].shape) != tuple(target[tgt_b_key].shape):
                raise RuntimeError(
                    f'VGG classifier bias-shape mismatch: '
                    f'{src_b_key}{tuple(state[src_b_key].shape)} -> '
                    f'{tgt_b_key}{tuple(target[tgt_b_key].shape)}'
                )
            remapped[tgt_b_key] = state[src_b_key]
            classifier_source_keys.add(src_b_key)

    fixed = {
        key: value
        for key, value in state.items()
        if key not in classifier_source_keys
    }
    fixed.update(remapped)
    return fixed


def build_reference_base(module, info):
    """Build/load an unwrapped LoRA/DoRA reference base for validation."""
    if info['type'] == 'sophon':
        model = module.build_resnet18()
        state = module.load_sophon_state(info['path'])
    elif info['type'] == 'vgg':
        model = module.CustomVGG(info['arch'], 10)
        state = module.load_vgg_state(info['path'])
        state = normalize_vgg_state_to_model(state, model)
    else:
        raise ValueError(info['type'])

    missing, unexpected = model.load_state_dict(state, strict=False)
    return model, list(missing), list(unexpected)


def module_signature(model):
    """Structural signature, including parameter-free modules such as Dropout."""
    sig = []
    for name, m in model.named_modules():
        if name == '':
            continue
        item = [name, type(m).__name__]
        if isinstance(m, nn.Conv2d):
            item += [m.in_channels, m.out_channels, tuple(m.kernel_size), tuple(m.stride), tuple(m.padding)]
        elif isinstance(m, nn.Linear):
            item += [m.in_features, m.out_features]
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            item += [str(m.output_size)]
        sig.append(tuple(item))
    return sig


def assert_same_base_model(reference_name, fusion_model, reference_model, checkpoint_name):
    """Fail if fusion base model differs from the standalone PEFT base model."""
    if module_signature(fusion_model) != module_signature(reference_model):
        raise RuntimeError(
            f'{checkpoint_name}: model structure differs between fusion/AutoSparse '
            f'loader and {reference_name} loader.'
        )

    fs = fusion_model.state_dict()
    rs = reference_model.state_dict()
    if list(fs.keys()) != list(rs.keys()):
        raise RuntimeError(
            f'{checkpoint_name}: state-dict keys differ between fusion and {reference_name}.'
        )
    for key in fs:
        if fs[key].shape != rs[key].shape:
            raise RuntimeError(
                f'{checkpoint_name}: shape mismatch for {key} between fusion and {reference_name}: '
                f'{tuple(fs[key].shape)} vs {tuple(rs[key].shape)}'
            )
        if not torch.equal(fs[key].cpu(), rs[key].cpu()):
            raise RuntimeError(
                f'{checkpoint_name}: loaded tensor differs for {key} between fusion and {reference_name}.'
            )

def validate_environment(infos):
    required_ab = [
        'train_ae_probe',
        'get_loaders',
        'build_model',
        'CHECKPOINTS',
        'FINAL_SEEDS',
        'AE_EPOCHS',
        'AE_LR',
        'AE_MAX_BATCHES',
        'SCORE_BATCHES',
        'MASK_MODE',
    ]
    required_lo = [
        'LoRAConv2d',
        'LoRALinear',
        'LORA_ALPHA_SOPHON',
        'LORA_ALPHA_VGG',
        'get_dataset_pair',
        'make_train_loader',
        'make_eval_loader',
        'RANKS',
        'EPOCHS',
        'BATCH_SIZE',
        'FINAL_SEEDS',
        'build_resnet18',
        'CustomVGG',
        'load_sophon_state',
        'load_vgg_state',
    ]
    required_do = [
        'DoRAConv2d',
        'DoRALinear',
        'DORA_ALPHA_SOPHON',
        'DORA_ALPHA_VGG',
        'get_dataset_pair',
        'make_train_loader',
        'make_eval_loader',
        'RANKS',
        'EPOCHS',
        'BATCH_SIZE',
        'FINAL_SEEDS',
        'build_resnet18',
        'CustomVGG',
        'load_sophon_state',
        'load_vgg_state',
    ]

    for x in required_ab:
        if not hasattr(ab, x):
            raise RuntimeError(f'ablat.py missing {x}')
    for x in required_lo:
        if not hasattr(lo, x):
            raise RuntimeError(f'lora.py missing {x}')
    for x in required_do:
        if not hasattr(do, x):
            raise RuntimeError(f'dora.py missing {x}')

    if len(ab.CHECKPOINTS) != 13:
        raise RuntimeError(
            f'Expected 13 hardened checkpoints from ablat.py, found {len(ab.CHECKPOINTS)}'
        )
    if set(TARGETS) != set(EXPECTED_CHECKPOINTS):
        raise RuntimeError(
            'ablat.py checkpoint names do not match the expected 13-checkpoint set.\n'
            f'Found: {sorted(TARGETS)}'
        )
    if int(getattr(ab, 'BATCH_SIZE', BATCH_SIZE)) != BATCH_SIZE:
        raise RuntimeError(f'ablat.py batch size changed: {ab.BATCH_SIZE}')
    if list(getattr(ab, 'FINAL_SEEDS', SEEDS)) != SEEDS:
        raise RuntimeError(f'ablat.py seeds changed: {ab.FINAL_SEEDS}')
    if int(ab.AE_EPOCHS) != 4:
        raise RuntimeError(f'AutoSparse AE_EPOCHS changed: {ab.AE_EPOCHS}')
    if not np.isclose(float(ab.AE_LR), 1e-3):
        raise RuntimeError(f'AutoSparse AE_LR changed: {ab.AE_LR}')
    if int(ab.AE_MAX_BATCHES) != 200 or int(ab.SCORE_BATCHES) != 200:
        raise RuntimeError(
            f'AutoSparse probe limits changed: AE_MAX_BATCHES={ab.AE_MAX_BATCHES}, '
            f'SCORE_BATCHES={ab.SCORE_BATCHES}'
        )
    if str(ab.MASK_MODE).lower() != 'channel':
        raise RuntimeError(f'AutoSparse MASK_MODE must be channel, got: {ab.MASK_MODE}')

    for label, module in [('LoRA', lo), ('DoRA', do)]:
        if int(module.BATCH_SIZE) != BATCH_SIZE:
            raise RuntimeError(f'{label} batch size changed: {module.BATCH_SIZE}')
        if int(module.EPOCHS) != EPOCHS:
            raise RuntimeError(f'{label} epoch protocol changed: {module.EPOCHS}')
        if list(module.FINAL_SEEDS) != SEEDS:
            raise RuntimeError(f'{label} seeds changed: {module.FINAL_SEEDS}')
        # Fusion uses its explicitly requested rank; it does not have to
        # belong to the standalone LoRA/DoRA sweep grid.
        if RANK <= 0:
            raise RuntimeError(f'{label}: invalid Fusion rank {RANK}')

    if int(lo.LORA_ALPHA_SOPHON) != 32 or int(lo.LORA_ALPHA_VGG) != 16:
        raise RuntimeError('LoRA alpha protocol changed.')
    if int(do.DORA_ALPHA_SOPHON) != 32 or int(do.DORA_ALPHA_VGG) != 16:
        raise RuntimeError('DoRA alpha protocol changed.')

    print('\nHYBRID VALIDATION | ' + EXPERIMENT_TAG)
    print(
        f'checkpoints={len(infos)} | attacks=LoRA+DoRA | budget=10% | rank=16 | '
        'lr=0.001 constant | epochs=20 | batch=64 | seeds=42,43,44'
    )

    for info in infos:
        name = info['name']
        if name not in EXPECTED_CHECKPOINTS:
            raise RuntimeError(f'Unexpected checkpoint in ablat.py: {name}')

        exp_method, exp_type, exp_arch, exp_dataset, exp_suffix = EXPECTED_CHECKPOINTS[name]
        actual_meta = (info['method'], info['type'], info['arch'], info['dataset'])
        expected_meta = (exp_method, exp_type, exp_arch, exp_dataset)
        if actual_meta != expected_meta:
            raise RuntimeError(
                f'{name}: metadata mismatch: actual={actual_meta}, expected={expected_meta}'
            )

        norm_path = os.path.normpath(info['path']).replace('\\', '/')
        norm_suffix = os.path.normpath(exp_suffix).replace('\\', '/')
        if not norm_path.endswith(norm_suffix):
            raise RuntimeError(
                f'{name}: wrong checkpoint path: {info["path"]} | '
                f'expected suffix: {exp_suffix}'
            )
        if not os.path.exists(info['path']):
            raise FileNotFoundError(info['path'])

        model, missing, unexpected = ab.build_model(info)
        if missing or unexpected:
            raise RuntimeError(
                f'{name}: fusion/AutoSparse load mismatch: missing={missing}, unexpected={unexpected}'
            )

        # Independently reload the same checkpoint through the standalone LoRA
        # and DoRA loading paths. This catches VGG/CUPI/HNTL/CUTI mapping or
        # architecture drift before any expensive training starts.
        lora_base, lora_missing, lora_unexpected = build_reference_base(lo, info)
        if lora_missing or lora_unexpected:
            raise RuntimeError(
                f'{name}: LoRA reference load mismatch: '
                f'missing={lora_missing}, unexpected={lora_unexpected}'
            )
        dora_base, dora_missing, dora_unexpected = build_reference_base(do, info)
        if dora_missing or dora_unexpected:
            raise RuntimeError(
                f'{name}: DoRA reference load mismatch: '
                f'missing={dora_missing}, unexpected={dora_unexpected}'
            )
        assert_same_base_model('LoRA', model, lora_base, name)
        assert_same_base_model('DoRA', model, dora_base, name)

        nconv = sum(isinstance(m, nn.Conv2d) for m in model.modules())
        expected_convs = 20 if info['arch'] == 'resnet18' else (10 if info['arch'] == 'vgg13' else 16)
        if nconv != expected_convs:
            raise RuntimeError(
                f'{name}: expected {expected_convs} Conv2d layers, got {nconv}'
            )

        # VGG implementations may include Dropout modules, so classifier indices
        # can differ. Validate the actual Linear structure instead of hard-coding
        # 0/2/4 versus 0/3/6. Strict checkpoint loading above must still be clean.
        if info['type'] == 'vgg':
            linear_layers = [
                (n, m.in_features, m.out_features)
                for n, m in model.named_modules()
                if isinstance(m, nn.Linear)
            ]
            expected_shapes = [(2048, 256), (256, 256), (256, 10)]
            actual_shapes = [(i, o) for _, i, o in linear_layers]
            if actual_shapes != expected_shapes:
                raise RuntimeError(
                    f'{name}: unexpected VGG classifier Linear structure: {linear_layers}'
                )

        print(
            f'OK | {name} | {info["method"]} | {info["dataset"]} | '
            f'{info["arch"]} | convs={nconv} | path={info["path"]}'
        )
        del model, lora_base, dora_base

    print("Validation passed.")


def already_done(name, attack, seed):
    if not os.path.exists(SEED_CSV):
        return False

    df = pd.read_csv(SEED_CSV)
    if df.empty:
        return False

    required = {'checkpoint_name', 'attack', 'seed', 'budget', 'rank'}
    if not required.issubset(df.columns):
        return False

    attack_name = f'autosparse_guided_{attack}'
    mask = (
        (df['checkpoint_name'].astype(str) == str(name))
        & (df['attack'].astype(str) == attack_name)
        & (df['seed'].astype(int) == int(seed))
        & np.isclose(df['budget'].astype(float), BUDGET)
        & (df['rank'].astype(int) == RANK)
    )
    return bool(mask.any())


def train_attack(
    info,
    seed,
    attack,
    ae_scores,
    n_scored,
    probe_stats,
    n_train,
    n_test,
    logger,
):
    # Restart from the same seed/checkpoint for each attack.
    # The AE probe uses the AutoSparse pipeline, while adaptation uses
    # the LoRA/DoRA input pipeline and seeded DataLoader.
    set_seed(seed)
    train_loader, test_loader, n_train2, n_test2 = build_attack_loaders(
        info, seed, attack
    )
    if n_train != n_train2 or n_test != n_test2:
        raise RuntimeError(
            f'Dataset-size mismatch for {info["name"]}: '
            f'AutoSparse probe=({n_train},{n_test}), '
            f'{attack} adaptation=({n_train2},{n_test2})'
        )

    # Match standalone PEFT initialization exactly: reset the global RNG
    # immediately before constructing the base model and adapter parameters.
    set_seed(seed)
    model, missing, unexpected = ab.build_model(info)
    if missing or unexpected:
        raise RuntimeError(
            f'{info["name"]}: bad checkpoint load for hybrid: '
            f'{missing}, {unexpected}'
        )

    base_params = sum(p.numel() for p in model.parameters())
    for p in model.parameters():
        p.requires_grad = False

    alpha = alpha_for(info, attack)
    if attack == 'lora':
        stats = inject_guided_lora(model, ae_scores, RANK, alpha)
    elif attack == 'dora':
        stats = inject_guided_dora(model, ae_scores, RANK, alpha)
    else:
        raise ValueError(attack)

    (
        nconv,
        nlinear,
        selected_channels,
        candidate_channels,
        effective_adapter,
        physical_adapter,
        targeted,
    ) = stats

    if nconv != n_scored:
        raise RuntimeError(
            f'Conv score/injection mismatch: scored={n_scored}, injected={nconv}'
        )

    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != physical_adapter:
        raise RuntimeError(
            f'{attack}: trainable adapter mismatch: '
            f'requires_grad={trainable}, physical={physical_adapter}'
        )

    physical_pct = 100.0 * physical_adapter / base_params
    effective_pct = 100.0 * effective_adapter / base_params

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    ce = nn.CrossEntropyLoss()

    reset_peak()
    total_train = 0.0
    total_test = 0.0
    best_acc = -1.0
    best_epoch = -1
    epoch20_acc = None
    epoch20_loss = None

    logger.info(
        'RUN | %s | attack=%s | seed=%d | budget=10%% | r=16 | alpha=%d | '
        'conv=%d | linear=%d | selected_channels=%d/%d | '
        'effective_adapter=%d (%.6f%% base) | physical_adapter=%d (%.6f%% base) | '
        'adaptation_pipeline=%s_baseline',
        info['name'],
        attack,
        seed,
        alpha,
        nconv,
        nlinear,
        selected_channels,
        candidate_channels,
        effective_adapter,
        effective_pct,
        physical_adapter,
        physical_pct,
        attack,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()

        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        loss_sum = 0.0
        seen = 0

        for x, y in tqdm(
            train_loader,
            desc=f'HYBRID-{attack.upper()} {info["name"]} S{seed} E{epoch}',
            leave=False,
        ):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = ce(out, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            loss_sum += loss.item() * bs
            seen += bs

        if DEVICE == 'cuda':
            torch.cuda.synchronize()

        train_time = time.perf_counter() - t0
        total_train += train_time

        acc, test_loss, test_time, _ = evaluate(
            model,
            test_loader,
            predictions=False,
        )
        total_test += test_time

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        if epoch == EPOCHS:
            epoch20_acc = acc
            epoch20_loss = test_loss

        alloc, reserved = peak_mb()
        train_loss = loss_sum / max(seen, 1)

        append_row(EPOCH_CSV, {
            'checkpoint_name': info['name'],
            'method': info['method'],
            'arch': info['arch'],
            'dataset': info['dataset'],
            'attack': f'autosparse_guided_{attack}',
            'seed': seed,
            'budget': BUDGET,
            'budget_percent': 100.0 * BUDGET,
            'rank': RANK,
            'alpha': alpha,
            'lr': LR,
            'lr_schedule': 'constant',
            'adaptation_data_pipeline': f'{attack}_baseline',
            'epoch': epoch,
            'epochs': EPOCHS,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_acc': acc,
            'best_acc_so_far': best_acc,
            'best_epoch_so_far': best_epoch,
            'epoch_train_time_sec': train_time,
            'epoch_test_time_sec': test_time,
            'effective_adapter_params': effective_adapter,
            'physical_adapter_params': physical_adapter,
            'effective_percent_of_base': effective_pct,
            'physical_percent_of_base': physical_pct,
            'selected_channels_total': selected_channels,
            'candidate_channels_total': candidate_channels,
            'selected_channel_percent': (
                100.0 * selected_channels / max(candidate_channels, 1)
            ),
            'gpu': gpu_name(),
        })

        logger.info(
            'HYBRID | %s | attack=%s | seed=%d | epoch=%02d/%d | '
            'acc=%.2f%% | best=%.2f%%@%d | train=%.2fs | test=%.2fs',
            info['name'],
            attack,
            seed,
            epoch,
            EPOCHS,
            acc,
            best_acc,
            best_epoch,
            train_time,
            test_time,
        )

    final_acc, final_loss, final_test_time, _ = evaluate(
        model,
        test_loader,
        predictions=False,
    )

    if abs(final_acc - epoch20_acc) > 1e-9:
        raise RuntimeError(
            f'Final evaluation mismatch: {final_acc} vs {epoch20_acc}'
        )

    alloc, reserved = peak_mb()
    total_attack = probe_stats['probe_time_sec'] + total_train

    append_row(SEED_CSV, {
        'method': info['method'],
        'checkpoint_name': info['name'],
        'dataset': info['dataset'],
        'task': info['task'],
        'arch': info['arch'],
        'checkpoint': info['path'],
        'attack': f'autosparse_guided_{attack}',
        'seed': seed,
        'budget': BUDGET,
        'budget_percent': 100.0 * BUDGET,
        'rank': RANK,
        'adapter_alpha': alpha,
        'lr': LR,
        'lr_schedule': 'constant',
        'adaptation_data_pipeline': f'{attack}_baseline',
        'batch_size': BATCH_SIZE,
        'weight_decay': WEIGHT_DECAY,
        'epochs': EPOCHS,
        'epoch20_acc': epoch20_acc,
        'epoch20_loss': epoch20_loss,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'ae_probe_time_sec': probe_stats['probe_time_sec'],
        'train_time_sec': total_train,
        'test_time_sec': total_test,
        'final_test_time_sec': final_test_time,
        'total_attack_time_sec': total_attack,
        'peak_allocated_mb': alloc,
        'peak_reserved_mb': reserved,
        'base_model_params': base_params,
        'physical_adapter_params': physical_adapter,
        'effective_adapter_params': effective_adapter,
        'physical_percent_of_base': physical_pct,
        'effective_percent_of_base': effective_pct,
        'selected_channels_total': selected_channels,
        'candidate_channels_total': candidate_channels,
        'selected_channel_percent': (
            100.0 * selected_channels / max(candidate_channels, 1)
        ),
        'num_conv_adapter_layers': nconv,
        'num_linear_adapter_layers': nlinear,
        'missing_keys': len(missing),
        'unexpected_keys': len(unexpected),
        'gpu': gpu_name(),
        'targeted_layers': ' | '.join(targeted),
    })

    logger.info(
        'RESULT | %s | attack=%s | seed=%d | epoch20=%.4f%% | '
        'best=%.4f%%@%d | AEprobe=%.2fs | train=%.2fs | attack_time=%.2fs | '
        'effective=%.6f%% base | physical=%.6f%% base',
        info['name'],
        attack,
        seed,
        epoch20_acc,
        best_acc,
        best_epoch,
        probe_stats['probe_time_sec'],
        total_train,
        total_attack,
        effective_pct,
        physical_pct,
    )

    del model, optimizer, train_loader, test_loader
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()


def rebuild_summary():
    if not os.path.exists(SEED_CSV):
        return

    df = pd.read_csv(SEED_CSV)
    if df.empty:
        return

    df = df[
        np.isclose(df['budget'].astype(float), BUDGET)
        & (df['rank'].astype(int) == RANK)
    ].copy()

    rows = []

    for (name, attack), g in df.groupby(['checkpoint_name', 'attack']):
        g = (
            g[g['seed'].isin(SEEDS)]
            .drop_duplicates('seed', keep='last')
        )

        if set(g['seed'].astype(int)) != set(SEEDS):
            continue

        acc = g['epoch20_acc'].astype(float).to_numpy()
        mean = float(acc.mean())
        sd = float(acc.std(ddof=1))
        half = T95_DF2 * sd / math.sqrt(len(SEEDS))
        r = g.iloc[0]

        rows.append({
            'checkpoint_name': name,
            'method': r['method'],
            'dataset': r['dataset'],
            'arch': r['arch'],
            'attack': attack,
            'budget_percent': 100.0 * BUDGET,
            'rank': RANK,
            'adapter_alpha': int(r['adapter_alpha']),
            'n_seeds': len(SEEDS),
            'seeds': '|'.join(str(s) for s in SEEDS),
            'mean_epoch20_acc': mean,
            'sample_sd': sd,
            'ci95_low': mean - half,
            'ci95_high': mean + half,
            'mean_ae_probe_time_sec': float(g['ae_probe_time_sec'].mean()),
            'mean_train_time_sec': float(g['train_time_sec'].mean()),
            'mean_total_attack_time_sec': float(
                g['total_attack_time_sec'].mean()
            ),
            'mean_peak_allocated_mb': float(
                g['peak_allocated_mb'].mean()
            ),
            'physical_adapter_params': int(r['physical_adapter_params']),
            'effective_adapter_params': int(r['effective_adapter_params']),
            'physical_percent_of_base': float(
                r['physical_percent_of_base']
            ),
            'effective_percent_of_base': float(
                r['effective_percent_of_base']
            ),
            'selected_channel_percent': float(
                r['selected_channel_percent']
            ),
            'gpu': r['gpu'],
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['checkpoint_name', 'attack'])
    out.to_csv(SUMMARY_CSV, index=False)


def run_checkpoint_seed(info, seed, logger):
    # Probe once, then use exactly the same channel ranking for LoRA and DoRA.
    pending = [
        attack
        for attack in ATTACKS
        if not already_done(info['name'], attack, seed)
    ]

    if not pending:
        logger.info(
            'SKIP | %s | seed=%d | LoRA and DoRA already complete',
            info['name'],
            seed,
        )
        return

    set_seed(seed)
    probe_train, _, n_train, n_test = ab.get_loaders(info['dataset'])

    probe_model, missing, unexpected = ab.build_model(info)
    if missing or unexpected:
        raise RuntimeError(
            f'{info["name"]}: bad checkpoint load before probe: '
            f'{missing}, {unexpected}'
        )

    probe_model = probe_model.to(DEVICE)
    ae_bank, ae_scores, n_scored, probe_stats = ab.train_ae_probe(
        probe_model,
        probe_train,
        logger,
        info,
    )

    del probe_model, ae_bank, probe_train
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    for attack in pending:
        train_attack(
            info=info,
            seed=seed,
            attack=attack,
            ae_scores=ae_scores,
            n_scored=n_scored,
            probe_stats=probe_stats,
            n_train=n_train,
            n_test=n_test,
            logger=logger,
        )
        rebuild_summary()

    del ae_scores
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()


def print_progress():
    total = len(ab.CHECKPOINTS) * len(ATTACKS) * len(SEEDS)
    done = 0

    if os.path.exists(SEED_CSV):
        df = pd.read_csv(SEED_CSV)
        if not df.empty:
            df = df[
                np.isclose(df['budget'].astype(float), BUDGET)
                & (df['rank'].astype(int) == RANK)
                & df['attack'].isin(
                    ['autosparse_guided_lora', 'autosparse_guided_dora']
                )
                & df['seed'].isin(SEEDS)
            ]
            done = len(
                df.drop_duplicates(
                    ['checkpoint_name', 'attack', 'seed'],
                    keep='last',
                )
            )

    print(f'PROGRESS: {done}/{total} seed-attack runs complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--only',
        default='all',
        choices=['all'] + TARGETS,
    )
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    infos = get_infos(args.only)
    validate_environment(infos)

    if args.validate_only:
        return

    if args.overwrite and os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    logger = setup_logger()

    logger.info('=' * 100)
    logger.info(
        'AUTOSPARSE-GUIDED HYBRID | all checkpoints | LoRA+DoRA | '
        'budget=10%% | rank=16 | constant_lr=0.001 | seeds=42,43,44'
    )
    logger.info(
        'AE reconstruction-error probe is run once per checkpoint/seed '
        'and reused for LoRA and DoRA.'
    )
    logger.info('=' * 100)

    for info in infos:
        for seed in SEEDS:
            run_checkpoint_seed(info, seed, logger)
            print_progress()

    rebuild_summary()

    print("Done.")
    print_progress()
    print('Seed results :', SEED_CSV)
    print('Summary      :', SUMMARY_CSV)
    print('Log          :', LOG_PATH)


if __name__ == '__main__':
    main()
