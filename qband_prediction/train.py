import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, Callback, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from dmpnn import dmpnn
from data import MoleculeDataset, collate_fn
from features import (
    calculate_descriptors,
    descriptor_names,
    mol_from_smiles_relaxed,
    solute_descriptor_functions,
)
from training_utils import windows_high_qos


LR              = 2e-4
L2              = 1e-5
BATCH_SIZE      = 32
Dropout         = 0.05
Random_seed     = 42
EPOCHS          = 300
num_experts     = 3
moe_hid_dim     = 128
edge_output_dim = 64
node_output_dim = 64
patience        = 80
num_rounds      = 4
POOLING         = 'mean_max'
ACTIVATION_TYPE1 = 'leakyrelu'
ACTIVATION_TYPE2 = 'tanh'
CPU_THREADS     = 4

WARM_START_CHECKPOINT = Path(__file__).resolve().parent / 'initialization_models' / 'mean_qband_seed42.ckpt'
INITIALIZATION_METADATA = {'method': 'random'}

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
USE_LOG_TRANSFORM = False
NUM_WORKERS = 0
GRAPH_FEATURE_VERSION = 'periodic_v2'
AD_SIMILARITY_THRESHOLD = 0.75
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 40
SCHEDULER_MIN_LR = 1e-6
LOG_EVERY_N_STEPS = 5
GRADIENT_CLIP_VAL = 1.0




TAIL_WEIGHT = 2.0
LAMBDA_REAL = 0.01


def plot_parity(train_true, train_pred, test_true, test_pred, title, save_path):
    train_true = np.asarray(train_true).reshape(-1)
    train_pred = np.asarray(train_pred).reshape(-1)
    test_true = np.asarray(test_true).reshape(-1)
    test_pred = np.asarray(test_pred).reshape(-1)

    plt.figure(figsize=(8, 8), dpi=600)

    plt.scatter(
        train_true, train_pred,
        c='#1f77b4',
        label='Train Set',
        alpha=0.7,
        s=30,
        edgecolors='none'
    )

    plt.scatter(
        test_true, test_pred,
        c='#ff7f0e',
        label='Test Set',
        alpha=0.7,
        s=30,
        edgecolors='none'
    )

    all_true = np.concatenate([train_true, test_true])
    all_pred = np.concatenate([train_pred, test_pred])

    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))
    margin = (max_val - min_val) * 0.05 if max_val > min_val else 1.0

    plt.plot(
        [min_val - margin, max_val + margin],
        [min_val - margin, max_val + margin],
        linestyle='--',
        linewidth=2,
        color='red',
        label='Ideal Prediction'
    )

    r2_train = r2_score(train_true, train_pred)
    mae_train = mean_absolute_error(train_true, train_pred)
    r2_test = r2_score(test_true, test_pred)
    mae_test = mean_absolute_error(test_true, test_pred)

    text_str = (
        f'Train $\\mathbf{{R^2}}$ = {r2_train:.3f}\n'
        f'Train MAE = {mae_train:.3f}\n'
        f'Test  $\\mathbf{{R^2}}$ = {r2_test:.3f}\n'
        f'Test  MAE = {mae_test:.3f}'
    )

    plt.text(
        0.05, 0.95, text_str,
        transform=plt.gca().transAxes,
        fontsize=12,
        fontweight='bold',
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            edgecolor='black',
            linewidth=1.5,
            alpha=0.95
        )
    )

    plt.xlabel('Experimental Value', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Value', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')

    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    legend = plt.legend(fontsize=12, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor('black')

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    png_path = save_path.replace('.png', '_600dpi.png')
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Parity plot saved:\n- {png_path}")


def plot_single_parity(y_true, y_pred, title, save_path, point_color='#1f77b4', point_label='Dataset'):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    plt.figure(figsize=(8, 8), dpi=600)

    plt.scatter(
        y_true, y_pred,
        c=point_color,
        label=point_label,
        alpha=0.7,
        s=30,
        edgecolors='none'
    )

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    margin = (max_val - min_val) * 0.05 if max_val > min_val else 1.0

    plt.plot(
        [min_val - margin, max_val + margin],
        [min_val - margin, max_val + margin],
        linestyle='--',
        linewidth=2,
        color='red',
        label='Ideal Prediction'
    )

    r2_val = r2_score(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)

    text_str = (
        f'$\\mathbf{{R^2}}$ = {r2_val:.3f}\n'
        f'MAE = {mae_val:.3f}'
    )

    plt.text(
        0.05, 0.95, text_str,
        transform=plt.gca().transAxes,
        fontsize=12,
        fontweight='bold',
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            edgecolor='black',
            linewidth=1.5,
            alpha=0.95
        )
    )

    plt.xlabel('Experimental Value', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Value', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')

    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    legend = plt.legend(fontsize=12, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor('black')

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    png_path = save_path.replace('.png', '_600dpi.png')
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Single parity plot saved:\n- {png_path}")


def plot_loss_curve(history, save_dir):
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_r2 = history['train_r2']
    val_r2 = history['val_r2']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(14, 6), dpi=300)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs, val_loss, label='Val Loss', color='#ff7f0e', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training & Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_r2, label='Train R2', color='#2ca02c', linewidth=2)
    plt.plot(epochs, val_r2, label='Val R2', color='#d62728', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('R2 Score', fontsize=12)
    plt.title('Training & Validation R2 Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(-1, 1.1)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练过程曲线已保存: {save_path}")


def extract_history_from_metrics_csv(metrics_csv_path):
    if not os.path.exists(metrics_csv_path):
        raise FileNotFoundError(f"未找到 metrics.csv: {metrics_csv_path}")

    df = pd.read_csv(metrics_csv_path)
    if 'epoch' not in df.columns:
        raise ValueError("metrics.csv 中缺少 epoch 列，无法提取训练历史")

    wanted_cols = ['train_loss', 'val_loss', 'train_r2', 'val_r2']
    for col in wanted_cols:
        if col not in df.columns:
            df[col] = np.nan

    history_rows = []
    for epoch, g in df.groupby('epoch', sort=True):
        row = {'epoch': int(epoch)}
        for col in wanted_cols:
            valid = g[col].dropna()
            row[col] = float(valid.iloc[-1]) if len(valid) > 0 else np.nan
        history_rows.append(row)

    history_df = pd.DataFrame(history_rows).sort_values('epoch').reset_index(drop=True)
    history_df = history_df.dropna(subset=['train_loss', 'val_loss', 'train_r2', 'val_r2'], how='all')

    history = {
        'epoch': history_df['epoch'].tolist(),
        'train_loss': history_df['train_loss'].tolist(),
        'val_loss': history_df['val_loss'].tolist(),
        'train_r2': history_df['train_r2'].tolist(),
        'val_r2': history_df['val_r2'].tolist(),
    }
    return history, history_df


def build_output_dir(base_dir='results', test_r2=0.0):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{timestamp}_R2_{float(test_r2):.4f}"
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def collect_predictions(model, dataloader, device):
    model.eval()
    model.to(device)

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            batched_graph, extra_features, labels = batch
            batched_graph = batched_graph.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)

            preds = model(batched_graph, extra_features).squeeze(-1)
            preds_real, labels_real = model._to_original_scale(preds, labels)

            all_pred.append(preds_real.detach().cpu().numpy())
            all_true.append(labels_real.detach().cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    return all_true, all_pred


def calculate_applicability_domain(
    reference_smiles,
    query_smiles,
    similarity_threshold=AD_SIMILARITY_THRESHOLD,
):
    """Estimate whether queries are represented by the training chemistry."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=True,
    )
    reference_mols = [mol_from_smiles_relaxed(smiles) for smiles in reference_smiles]
    if any(mol is None for mol in reference_mols):
        raise ValueError("训练集中存在无法用于应用域计算的 SMILES")

    reference_fps = [generator.GetFingerprint(mol) for mol in reference_mols]
    reference_elements = {
        atom.GetAtomicNum()
        for mol in reference_mols
        for atom in mol.GetAtoms()
    }

    similarities = []
    unseen_elements = []
    for smiles in query_smiles:
        mol = mol_from_smiles_relaxed(smiles)
        if mol is None:
            similarities.append(0.0)
            unseen_elements.append(True)
            continue
        fingerprint = generator.GetFingerprint(mol)
        similarities.append(
            float(max(DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fps)))
        )
        unseen_elements.append(
            any(
                atom.GetAtomicNum() not in reference_elements
                for atom in mol.GetAtoms()
            )
        )

    similarities = np.asarray(similarities, dtype=np.float64)
    unseen_elements = np.asarray(unseen_elements, dtype=bool)
    in_domain = (similarities >= similarity_threshold) & (~unseen_elements)
    return {
        'max_train_tanimoto': similarities,
        'unseen_element': unseen_elements,
        'in_applicability_domain': in_domain,
    }


def save_scatter_csv(
    smiles_list,
    y_true,
    y_pred,
    save_path,
    extra_columns=None,
):
    data = {
        'smiles': smiles_list,
        'experimental_value': np.asarray(y_true).reshape(-1),
        'predicted_value': np.asarray(y_pred).reshape(-1),
        'error': np.asarray(y_pred).reshape(-1) - np.asarray(y_true).reshape(-1),
        'abs_error': np.abs(np.asarray(y_pred).reshape(-1) - np.asarray(y_true).reshape(-1)),
    }
    if extra_columns:
        for name, values in extra_columns.items():
            if len(values) != len(smiles_list):
                raise ValueError(f"附加列 {name} 的长度与样本数不一致")
            data[name] = values
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"散点数据 CSV 已保存: {save_path}")


def save_all_artifacts(
    best_model_path,
    scaler_params,
    train_dataset,
    val_dataset,
    test_dataset,
    logger_log_dir,
    final_test_r2,
    model_params,
    data_path,
):
    save_dir = build_output_dir(base_dir='results', test_r2=final_test_r2)

    hyperparameters = {
        'data': {
            'input_file': data_path,
            'input_file_absolute': os.path.abspath(data_path),
            'target_column': 'q_band',
            'batch_size': BATCH_SIZE,
            'train_ratio': TRAIN_RATIO,
            'val_ratio': VAL_RATIO,
            'test_ratio': 1.0 - TRAIN_RATIO - VAL_RATIO,
            'split_sizes': scaler_params['split_sizes'],
            'random_seed': Random_seed,
            'num_workers': NUM_WORKERS,
            'use_log_transform': USE_LOG_TRANSFORM,
            'graph_feature_version': GRAPH_FEATURE_VERSION,
        },
        'model': model_params,
        'loss': {
            'normalized_loss': 'tail_weighted_mse',
            'real_space_loss': 'smooth_l1',
            'tail_weight': TAIL_WEIGHT,
            'lambda_real': LAMBDA_REAL,
        },
        'optimizer': {
            'name': 'Adam',
            'learning_rate': LR,
            'weight_decay': L2,
        },
        'lr_scheduler': {
            'name': 'ReduceLROnPlateau',
            'monitor': 'val_r2',
            'mode': 'max',
            'factor': SCHEDULER_FACTOR,
            'patience': SCHEDULER_PATIENCE,
            'min_lr': SCHEDULER_MIN_LR,
        },
        'training': {
            'max_epochs': EPOCHS,
            'early_stopping_monitor': 'val_r2',
            'early_stopping_mode': 'max',
            'early_stopping_patience': patience,
            'gradient_clip_val': GRADIENT_CLIP_VAL,
            'log_every_n_steps': LOG_EVERY_N_STEPS,
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'cpu_threads': CPU_THREADS,
            'matmul_precision': torch.get_float32_matmul_precision(),
            'initialization': INITIALIZATION_METADATA,
        },
        'applicability_domain': {
            'tanimoto_similarity_threshold': AD_SIMILARITY_THRESHOLD,
        },
    }
    hyperparameters_path = os.path.join(save_dir, 'hyperparameters.json')
    with open(hyperparameters_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparameters, f, ensure_ascii=False, indent=2)
    print(f"本次训练超参数已保存: {hyperparameters_path}")

    scaler_path = os.path.join(save_dir, 'scaler_params.json')
    with open(scaler_path, 'w', encoding='utf-8') as f:
        json.dump(scaler_params, f, ensure_ascii=False, indent=2)

    if os.path.exists(best_model_path):
        ckpt_dst = os.path.join(save_dir, os.path.basename(best_model_path))
        shutil.copy2(best_model_path, ckpt_dst)
        print(f"最佳模型已复制到: {ckpt_dst}")
    else:
        raise FileNotFoundError(f"未找到最佳模型文件: {best_model_path}")

    metrics_csv_path = os.path.join(logger_log_dir, 'metrics.csv')
    if os.path.exists(metrics_csv_path):
        metrics_copy_path = os.path.join(save_dir, 'metrics.csv')
        shutil.copy2(metrics_csv_path, metrics_copy_path)
        print(f"训练日志已复制到: {metrics_copy_path}")

        history, history_df = extract_history_from_metrics_csv(metrics_csv_path)
        history_csv_path = os.path.join(save_dir, 'training_history.csv')
        history_df.to_csv(history_csv_path, index=False, encoding='utf-8-sig')
        print(f"训练历史 CSV 已保存: {history_csv_path}")

        if len(history['train_loss']) > 0:
            plot_loss_curve(history, save_dir)
    else:
        print(f"警告：未找到 metrics.csv，跳过训练曲线绘制。路径: {metrics_csv_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model = DMPNNLightningModule.load_from_checkpoint(
        best_model_path,
        map_location=device,
        weights_only=True,
    )
    best_model.eval()
    best_model.freeze()

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_eval_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_eval_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    train_true, train_pred = collect_predictions(best_model, train_eval_loader, device)
    val_true, val_pred = collect_predictions(best_model, val_eval_loader, device)
    test_true, test_pred = collect_predictions(best_model, test_eval_loader, device)
    val_domain = calculate_applicability_domain(
        train_dataset.smiles_list,
        val_dataset.smiles_list,
    )
    test_domain = calculate_applicability_domain(
        train_dataset.smiles_list,
        test_dataset.smiles_list,
    )

    train_csv_path = os.path.join(save_dir, 'train_scatter.csv')
    val_csv_path = os.path.join(save_dir, 'val_scatter.csv')
    test_csv_path = os.path.join(save_dir, 'test_scatter.csv')
    save_scatter_csv(train_dataset.smiles_list, train_true, train_pred, train_csv_path)
    save_scatter_csv(
        val_dataset.smiles_list,
        val_true,
        val_pred,
        val_csv_path,
        extra_columns=val_domain,
    )
    save_scatter_csv(
        test_dataset.smiles_list,
        test_true,
        test_pred,
        test_csv_path,
        extra_columns=test_domain,
    )

    plot_parity(
        train_true, train_pred, test_true, test_pred,
        title='Train/Test Parity Plot',
        save_path=os.path.join(save_dir, 'parity_plot.png')
    )

    plot_single_parity(
        train_true, train_pred,
        title='Train Set Parity Plot',
        save_path=os.path.join(save_dir, 'train_parity_plot.png'),
        point_color='#1f77b4',
        point_label='Train Set'
    )

    plot_single_parity(
        val_true, val_pred,
        title='Validation Set Parity Plot',
        save_path=os.path.join(save_dir, 'val_parity_plot.png'),
        point_color='#2ca02c',
        point_label='Validation Set'
    )

    plot_single_parity(
        test_true, test_pred,
        title='Test Set Parity Plot',
        save_path=os.path.join(save_dir, 'test_parity_plot.png'),
        point_color='#ff7f0e',
        point_label='Test Set'
    )

    def domain_metrics(y_true, y_pred, domain):
        mask = domain['in_applicability_domain']
        count = int(mask.sum())
        metrics = {
            'count': count,
            'coverage': float(mask.mean()),
            'similarity_threshold': AD_SIMILARITY_THRESHOLD,
        }
        if count >= 2:
            metrics.update({
                'r2': float(r2_score(y_true[mask], y_pred[mask])),
                'mae': float(mean_absolute_error(y_true[mask], y_pred[mask])),
                'rmse': float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))),
            })
        return metrics

    summary = {
        'best_model_path_original': best_model_path,
        'initialization': INITIALIZATION_METADATA,
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_r2': float(r2_score(test_true, test_pred)),
        'test_mae': float(mean_absolute_error(test_true, test_pred)),
        'test_rmse': float(np.sqrt(np.mean((test_true - test_pred) ** 2))),
        'train_r2': float(r2_score(train_true, train_pred)),
        'train_mae': float(mean_absolute_error(train_true, train_pred)),
        'train_rmse': float(np.sqrt(np.mean((train_true - train_pred) ** 2))),
        'val_r2': float(r2_score(val_true, val_pred)),
        'val_mae': float(mean_absolute_error(val_true, val_pred)),
        'val_rmse': float(np.sqrt(np.mean((val_true - val_pred) ** 2))),
        'val_in_domain': domain_metrics(val_true, val_pred, val_domain),
        'test_in_domain': domain_metrics(test_true, test_pred, test_domain),
    }
    summary_path = os.path.join(save_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"汇总信息已保存: {summary_path}")
    print(
        "应用域指标 "
        f"(Tanimoto >= {AD_SIMILARITY_THRESHOLD:.2f}): "
        f"Val R2={summary['val_in_domain'].get('r2', float('nan')):.4f}, "
        f"coverage={summary['val_in_domain']['coverage']:.1%}; "
        f"Test R2={summary['test_in_domain'].get('r2', float('nan')):.4f}, "
        f"coverage={summary['test_in_domain']['coverage']:.1%}"
    )

    print(f"\n所有结果已保存到文件夹: {save_dir}")
    return save_dir


class EpochLogCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        header = (
            f"{'Epoch':>6} | "
            f"{'Train Loss':>10} | {'Train R2':>8} | {'Train RMSE':>10} |"
            f"  {'Val Loss':>9} | {'Val R2':>8} | {'Val RMSE':>9} |"
            f"  {'LR':>10}"
        )
        sep = "-" * len(header)
        print(f"\n{sep}\n{header}\n{sep}")
        self._sep = sep

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        m = trainer.callback_metrics
        epoch = trainer.current_epoch
        train_loss = m.get("train_loss", float("nan"))
        train_r2 = m.get("train_r2", float("nan"))
        train_rmse = m.get("train_rmse", float("nan"))
        val_loss = m.get("val_loss", float("nan"))
        val_r2 = m.get("val_r2", float("nan"))
        val_rmse = m.get("val_rmse", float("nan"))
        lr = trainer.optimizers[0].param_groups[0]["lr"]

        def fmt(v):
            try:
                return f"{float(v):.4f}"
            except Exception:
                return "   N/A"

        print(
            f"{epoch:>6} | "
            f"{fmt(train_loss):>10} | {fmt(train_r2):>8} | {fmt(train_rmse):>10} |"
            f"  {fmt(val_loss):>9} | {fmt(val_r2):>8} | {fmt(val_rmse):>9} |"
            f"  {lr:.2e}"
        )

    def on_train_end(self, trainer, pl_module):
        print(self._sep + "\n")


class DMPNNLightningModule(LightningModule):
    def __init__(
        self,
        model_params,
        scaler_params,
        lr=5e-4,
        L2=5e-5,
        tail_weight=TAIL_WEIGHT,
        lambda_real=LAMBDA_REAL,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.L2 = L2
        self.model = dmpnn(**model_params)
        self.lr = lr

        self.loss_fn = nn.MSELoss()
        self.tail_weight = float(tail_weight)
        self.lambda_real = float(lambda_real)

        self.register_buffer("label_mean", torch.tensor(scaler_params['label_mean'], dtype=torch.float32))
        self.register_buffer("label_std", torch.tensor(scaler_params['label_std'], dtype=torch.float32))
        self.register_buffer("use_log", torch.tensor(scaler_params.get('use_log', False)))

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()

    def _to_original_scale(self, preds_norm, labels_norm):
        p = preds_norm * self.label_std + self.label_mean
        l = labels_norm * self.label_std + self.label_mean
        if self.use_log:
            p = torch.exp(p)
            l = torch.exp(l)
        return p, l

    def forward(self, batched_graph, extra_features):
        return self.model(batched_graph, extra_features)

    def training_step(self, batch, batch_idx):
        batched_graph, extra_features, labels = batch
        preds = self(batched_graph, extra_features).squeeze(-1)

        squared_error = (preds - labels).square()
        sample_weights = 1.0 + self.tail_weight * labels.abs()
        loss_norm = (sample_weights * squared_error).mean()

        preds_real, labels_real = self._to_original_scale(preds, labels)
        loss_real = F.smooth_l1_loss(preds_real, labels_real)

        loss = loss_norm + self.lambda_real * loss_real

        # Only accumulate metric state here. Calling a Metric instance invokes
        # forward(), which also computes the value for the current batch and
        # makes R2Score fail when the last batch contains a single sample.
        self.train_r2.update(preds_real.detach(), labels_real.detach())
        self.train_rmse.update(preds_real.detach(), labels_real.detach())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log('train_loss_norm', loss_norm, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log('train_loss_real', loss_real, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log('train_rmse', self.train_rmse, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log('train_r2', self.train_r2, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(labels))
        return loss

    def validation_step(self, batch, batch_idx):
        batched_graph, extra_features, labels = batch
        preds = self(batched_graph, extra_features).squeeze(-1)

        loss_norm = self.loss_fn(preds, labels)

        preds_real, labels_real = self._to_original_scale(preds, labels)
        loss_real = F.smooth_l1_loss(preds_real, labels_real)

        loss = loss_norm + self.lambda_real * loss_real

        self.val_r2.update(preds_real, labels_real)
        self.val_rmse.update(preds_real, labels_real)

        self.log('val_loss', loss, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log('val_loss_norm', loss_norm, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log('val_loss_real', loss_real, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log('val_rmse', self.val_rmse, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log('val_r2', self.val_r2, on_epoch=True, prog_bar=False, batch_size=len(labels))
        return loss

    def test_step(self, batch, batch_idx):
        batched_graph, extra_features, labels = batch
        preds = self(batched_graph, extra_features).squeeze(-1)

        preds_real, labels_real = self._to_original_scale(preds, labels)
        self.test_r2.update(preds_real, labels_real)
        self.test_rmse.update(preds_real, labels_real)
        self.test_mae.update(preds_real, labels_real)

        self.log('test_rmse', self.test_rmse, batch_size=len(labels))
        self.log('test_mae', self.test_mae, batch_size=len(labels))
        self.log('test_r2', self.test_r2, batch_size=len(labels))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.L2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=SCHEDULER_MIN_LR,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_r2",
                "interval": "epoch",
                "frequency": 1
            }
        }


def stratified_split(
    labels: np.ndarray,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    seed=Random_seed,
):
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 1 or len(labels) < 3:
        raise ValueError("labels 必须是一维数组且至少包含 3 个样本")
    if not np.isfinite(labels).all():
        raise ValueError("labels 中存在 NaN 或无穷值")
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("train_ratio/val_ratio 超出有效范围")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio 必须小于 1")

    rng = np.random.default_rng(seed)
    n_bins = 10
    # Quantile strata only depend on rank. Using raw labels avoids the old
    # unnecessary log(), which failed for zero/negative regression targets.
    bins = np.percentile(labels, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    bins = np.unique(bins)
    strata = np.digitize(labels, bins[1:-1], right=False)

    train_idx, val_idx, test_idx = [], [], []

    for s in np.unique(strata):
        idx_s = np.where(strata == s)[0]
        if len(idx_s) == 0:
            continue

        idx_s = rng.permutation(idx_s)
        n_s = len(idx_s)

        if n_s < 3:
            train_idx.extend(idx_s.tolist())
            continue

        n_te = max(1, int(np.round((1 - train_ratio - val_ratio) * n_s)))
        n_va = max(1, int(np.round(val_ratio * n_s)))
        n_tr = n_s - n_va - n_te

        if n_tr < 1:
            n_tr, n_va, n_te = n_s - 2, 1, 1

        train_idx.extend(idx_s[:n_tr].tolist())
        val_idx.extend(idx_s[n_tr:n_tr + n_va].tolist())
        test_idx.extend(idx_s[n_tr + n_va:].tolist())

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    assigned = np.concatenate([train_idx, val_idx, test_idx])
    if len(assigned) != len(labels) or len(np.unique(assigned)) != len(labels):
        raise RuntimeError("数据划分出现遗漏或交叉")
    return train_idx, val_idx, test_idx


def prepare_data(path, batch_size=BATCH_SIZE, scaler_output_path='scaler_params.json'):
    print(f"正在读取数据: {path}")
    df = pd.read_excel(path)

    required_columns = {'smiles', 'q_band'}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"缺少必要列: {sorted(missing_columns)}")

    df = df.copy()
    df['smiles'] = df['smiles'].astype('string').str.strip()
    df['q_band'] = pd.to_numeric(df['q_band'], errors='coerce')
    before_drop = len(df)
    df = df.dropna(subset=['smiles', 'q_band'])
    df = df[df['smiles'] != '']
    df = df[np.isfinite(df['q_band'].to_numpy(dtype=np.float64))]
    if len(df) != before_drop:
        print(f"警告：删除了 {before_drop - len(df)} 条空白或非数值样本")
    if USE_LOG_TRANSFORM and (df['q_band'] <= 0).any():
        raise ValueError("USE_LOG_TRANSFORM=True 时 q_band 必须全部大于 0")

    label_order = df['q_band'].to_numpy(dtype=np.float64)


    duplicate_smiles = df['smiles'].duplicated(keep=False)
    if duplicate_smiles.any():
        duplicate_count = int(duplicate_smiles.sum())
        raise ValueError(
            f"发现 {duplicate_count} 条重复 SMILES。请先合并重复测量，"
            "避免同一分子泄漏到训练集和测试集。"
        )

    features_list = []
    valid_smiles = []
    valid_canonical_smiles = []
    valid_labels = []

    for idx, (smiles, label) in enumerate(zip(df['smiles'].tolist(), df['q_band'].tolist())):
        mol = mol_from_smiles_relaxed(smiles)
        if mol is None:
            print(f"警告：样本 {idx} 的 SMILES 无效，跳过")
            continue
        desc = calculate_descriptors(smiles, solute_descriptor_functions)
        desc_vals = [desc[name] for name in descriptor_names]
        if np.any(np.isnan(desc_vals)) or np.any(np.isinf(desc_vals)):
            print(f"警告：样本 {idx} 的描述符含 nan/inf，跳过")
            continue
        features_list.append(desc_vals)
        valid_smiles.append(smiles)
        valid_canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
        valid_labels.append(label)

    canonical_series = pd.Series(valid_canonical_smiles)
    canonical_duplicates = canonical_series.duplicated(keep=False)
    if canonical_duplicates.any():
        raise ValueError(
            f"发现 {int(canonical_duplicates.sum())} 条规范化后重复的分子。"
            "请先按 canonical SMILES 合并重复测量，避免跨集合泄漏。"
        )

    extra_features = np.array(features_list, dtype=np.float32)
    extra_features = np.nan_to_num(extra_features, nan=0.0, posinf=0.0, neginf=0.0)

    labels = np.array(valid_labels, dtype=np.float32)
    smiles_list = valid_smiles
    total_len = len(labels)
    print(f"有效样本数: {total_len}，描述符维度: {extra_features.shape[1]}")

    if total_len == 0:
        raise ValueError("无有效样本，无法继续训练")

    train_idx, test_idx, val_idx = stratified_split(labels, seed=Random_seed)
    print(f"分层划分 : train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        sub = labels[idx]
        print(f"  {name}: min={sub.min():.1f}, max={sub.max():.1f}, mean={sub.mean():.1f}")

    feat_mean = extra_features[train_idx].mean(axis=0)
    feat_std = extra_features[train_idx].std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    extra_features = (extra_features - feat_mean) / feat_std

    X_train = extra_features[train_idx]
    X_val = extra_features[val_idx]
    X_test = extra_features[test_idx]

    work_labels = labels.copy()
    if USE_LOG_TRANSFORM:
        work_labels = np.log(work_labels)
        sk = pd.Series(work_labels).skew()
        ku = pd.Series(work_labels).kurtosis()
        print(f"Log变换后 - skewness: {sk:.4f}, kurtosis: {ku:.4f}")

    label_mean = work_labels[train_idx].mean()
    label_std = work_labels[train_idx].std()
    if label_std < 1e-6:
        label_std = 1.0

    y_train = np.nan_to_num((work_labels[train_idx] - label_mean) / label_std, nan=0.0)
    y_val = np.nan_to_num((work_labels[val_idx] - label_mean) / label_std, nan=0.0)
    y_test = np.nan_to_num((work_labels[test_idx] - label_mean) / label_std, nan=0.0)

    scaler_params = {
        'descriptor_names': descriptor_names,
        'feat_mean': feat_mean.tolist(),
        'feat_std': feat_std.tolist(),
        'label_mean': float(label_mean),
        'label_std': float(label_std),
        'raw_label_mean': float(labels[train_idx].mean()),
        'raw_label_std': float(labels[train_idx].std()),
        'use_log': USE_LOG_TRANSFORM,
        'use_extra_features': True,
        'graph_feature_version': GRAPH_FEATURE_VERSION,
        'node_feat_dim': 117 if GRAPH_FEATURE_VERSION == 'periodic_v2' else 111,
        'split_seed': Random_seed,
        'split_sizes': {
            'train': int(len(train_idx)),
            'val': int(len(val_idx)),
            'test': int(len(test_idx)),
        },
        'training_smiles': [smiles_list[i] for i in train_idx],
        'ad_similarity_threshold': AD_SIMILARITY_THRESHOLD,
    }

    with open(scaler_output_path, 'w', encoding='utf-8') as f:
        json.dump(scaler_params, f, ensure_ascii=False, indent=2)

    train_dataset = MoleculeDataset(
        [smiles_list[i] for i in train_idx],
        X_train,
        y_train,
        graph_feature_version=GRAPH_FEATURE_VERSION,
    )
    val_dataset = MoleculeDataset(
        [smiles_list[i] for i in val_idx],
        X_val,
        y_val,
        graph_feature_version=GRAPH_FEATURE_VERSION,
    )
    test_dataset = MoleculeDataset(
        [smiles_list[i] for i in test_idx],
        X_test,
        y_test,
        graph_feature_version=GRAPH_FEATURE_VERSION,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(Random_seed)
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=(NUM_WORKERS > 0),
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=(NUM_WORKERS > 0)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=(NUM_WORKERS > 0)
    )

    return train_loader, val_loader, test_loader, scaler_params, train_dataset, val_dataset, test_dataset


def initialize_from_mean_checkpoint(model, checkpoint_path, scaler_params):
    """Expand a compatible Mean readout and then fine-tune both pooling branches."""
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    source_params = payload['hyper_parameters']['model_params']
    target_params = model.hparams['model_params']
    if source_params.get('pooling', 'mean') != 'mean' or target_params.get('pooling') != 'mean_max':
        raise ValueError('Initialization requires a Mean source and a Mean+Max target.')
    for key in ('node_feat_dim', 'edge_feat_dim', 'edge_output_dim', 'node_output_dim',
                'extra_dim', 'num_rounds', 'moe_hid_dim', 'activation_type1', 'activation_type2'):
        if source_params[key] != target_params[key]:
            raise ValueError(f'Incompatible initialization architecture: {key}')
    source_scaler = payload['hyper_parameters']['scaler_params']
    for key in ('descriptor_names', 'feat_mean', 'feat_std', 'label_mean', 'label_std',
                'graph_feature_version', 'use_log'):
        if source_scaler.get(key) != scaler_params.get(key):
            raise ValueError(f'Incompatible initialization preprocessing: {key}')
    state, source_state = model.state_dict(), payload['state_dict']
    key = 'model.head.net.0.weight'
    if set(state) != set(source_state):
        raise ValueError('Initialization checkpoint has a different set of model tensors.')
    width, extra = target_params['node_output_dim'], target_params['extra_dim']
    original = source_state[key]
    if original.shape != (target_params['moe_hid_dim'], width + extra):
        raise ValueError('Unexpected Mean regression input dimensions.')
    for name in state:
        if name == key:
            expanded = torch.zeros_like(state[name])
            expanded[:, :width] = original[:, :width]
            expanded[:, 2 * width:] = original[:, width:]
            state[name] = expanded
        else:
            if state[name].shape != source_state[name].shape:
                raise ValueError(f'Initialization tensor shape differs: {name}')
            state[name] = source_state[name].clone()
    model.load_state_dict(state, strict=True)
    return {'method': 'Mean initialization followed by full Mean+Max fine-tuning',
            'checkpoint': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'max_branch_initial_weight_norm': 0.0, 'all_parameters_trainable': True}


def train():
    global INITIALIZATION_METADATA
    torch.set_num_threads(CPU_THREADS)
    windows_high_qos()
    torch.set_float32_matmul_precision('high')
    seed_everything(Random_seed, workers=True)
    np.random.seed(Random_seed)

    path = 'qband_data(531).xlsx'
    train_loader, val_loader, test_loader, scaler_params, train_dataset, val_dataset, test_dataset = prepare_data(path, BATCH_SIZE)
    EXTRA_DIM = len(scaler_params['feat_mean'])

    model_params = {
        'node_feat_dim': int(scaler_params['node_feat_dim']),
        'edge_feat_dim': 13,
        'edge_output_dim': edge_output_dim,
        'node_output_dim': node_output_dim,
        'extra_dim': EXTRA_DIM,
        'num_rounds': num_rounds,
        'dropout_rate': Dropout,
        'activation_type1': ACTIVATION_TYPE1,
        'activation_type2': ACTIVATION_TYPE2,
        'num_experts': num_experts,
        'moe_hid_dim': moe_hid_dim,
        'pooling': POOLING,
    }

    model = DMPNNLightningModule(
        model_params=model_params,
        scaler_params=scaler_params,
        lr=LR,
        L2=L2,
        tail_weight=TAIL_WEIGHT,
        lambda_real=LAMBDA_REAL,
    )

    INITIALIZATION_METADATA = {'method': 'random'}
    if WARM_START_CHECKPOINT:
        INITIALIZATION_METADATA = initialize_from_mean_checkpoint(model, WARM_START_CHECKPOINT, scaler_params)
        print(f'Initialization: {INITIALIZATION_METADATA}')

    epoch_log_cb = EpochLogCallback()
    lr_monitor = LearningRateMonitor(logging_interval='step')

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join('checkpoints', run_id),
        filename='best_model-{epoch:02d}-{val_r2:.4f}',
        save_top_k=1,
        monitor='val_r2',
        mode='max',
    )

    early_stop_cb = EarlyStopping(
        monitor='val_r2',
        patience=patience,
        mode='max'
    )

    csv_logger = CSVLogger("logs", name="dmpnn_qband", version=run_id)

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, epoch_log_cb],
        logger=csv_logger,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        enable_progress_bar=False,
    )

    print("开始训练 (监控 val_r2)...")
    trainer.fit(model, train_loader, val_loader)
    print(f"\n训练结束。最佳模型: {checkpoint_cb.best_model_path}")

    if checkpoint_cb.best_model_path:
        print("正在进行测试集评估...")
        test_results = trainer.test(model, dataloaders=test_loader, ckpt_path='best')
        final_r2 = test_results[0]['test_r2']
        final_rmse = test_results[0]['test_rmse']

        print("=" * 40)
        print("最终测试集结果 (原始 nm 空间):")
        print(f"  Test R2  : {final_r2:.4f}")
        print(f"  Test RMSE: {final_rmse:.4f} nm")
        print("=" * 40)

        save_dir = save_all_artifacts(
            best_model_path=checkpoint_cb.best_model_path,
            scaler_params=scaler_params,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            logger_log_dir=trainer.logger.log_dir,
            final_test_r2=final_r2,
            model_params=model_params,
            data_path=path,
        )
        print(f"结果归档目录: {save_dir}")


if __name__ == '__main__':
    train()
