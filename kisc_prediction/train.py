import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningModule, Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torchmetrics import Metric, R2Score
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from dmpnn import dmpnn
from data import MoleculeDataset, collate_fn
from features import calculate_descriptors, solute_descriptor_functions, descriptor_names, mol_from_smiles_relaxed

# Resolve local files relative to this script, independent of the working directory.
PROJECT_DIR = Path(__file__).resolve().parent

LR              = 5e-4
L2              = 1e-4
BATCH_SIZE      = 24
Dropout         = 0.25
Random_seed     = 42
EPOCHS          = 320
num_experts     = 1
moe_hid_dim     = 128
edge_output_dim = 96
node_output_dim = 96
patience        = 100
num_rounds      = 4
NUM_WORKERS     = 2
SPLIT_MODE      = 'random'  # 可选: 'random' / 'scaffold'
TRAIN_BIN_COUNT = 1



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
    save_dir = os.path.join(PROJECT_DIR, base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def build_training_hyperparameters(split_mode, model_params):
    return {
        'LR': LR,
        'L2': L2,
        'BATCH_SIZE': BATCH_SIZE,
        'Dropout': Dropout,
        'Random_seed': Random_seed,
        'EPOCHS': EPOCHS,
        'num_experts': num_experts,
        'moe_hid_dim': moe_hid_dim,
        'edge_output_dim': edge_output_dim,
        'node_output_dim': node_output_dim,
        'patience': patience,
        'num_rounds': num_rounds,
        'NUM_WORKERS': NUM_WORKERS,
        'SPLIT_MODE': str(split_mode),
        'TRAIN_BIN_COUNT': TRAIN_BIN_COUNT,
        'model_params': model_params,
    }


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


def save_scatter_csv(smiles_list, y_true, y_pred, save_path):
    df = pd.DataFrame({
        'smiles': smiles_list,
        'experimental_value': np.asarray(y_true).reshape(-1),
        'predicted_value': np.asarray(y_pred).reshape(-1),
        'error': np.asarray(y_pred).reshape(-1) - np.asarray(y_true).reshape(-1),
        'abs_error': np.abs(np.asarray(y_pred).reshape(-1) - np.asarray(y_true).reshape(-1)),
    })
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"散点数据 CSV 已保存: {save_path}")


def save_all_artifacts(
    best_model_path,
    scaler_params,
    train_dataset,
    test_dataset,
    logger_log_dir,
    final_test_r2,
    model_params,
    training_hyperparameters=None,
):
    save_dir = build_output_dir(base_dir='results', test_r2=final_test_r2)

    hyperparams_path = os.path.join(save_dir, 'training_hyperparameters.json')
    hyperparams_to_save = training_hyperparameters or {'model_params': model_params}
    with open(hyperparams_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams_to_save, f, ensure_ascii=False, indent=2)
    print(f"训练超参数已保存: {hyperparams_path}")

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
    best_model = DMPNNLightningModule.load_from_checkpoint(best_model_path, map_location=device)
    best_model.eval()
    best_model.freeze()

    train_eval_loader = DataLoader(
        train_dataset,
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
    test_true, test_pred = collect_predictions(best_model, test_eval_loader, device)

    train_csv_path = os.path.join(save_dir, 'train_scatter.csv')
    test_csv_path = os.path.join(save_dir, 'test_scatter.csv')
    save_scatter_csv(train_dataset.smiles_list, train_true, train_pred, train_csv_path)
    save_scatter_csv(test_dataset.smiles_list, test_true, test_pred, test_csv_path)

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
        test_true, test_pred,
        title='Test Set Parity Plot',
        save_path=os.path.join(save_dir, 'test_parity_plot.png'),
        point_color='#ff7f0e',
        point_label='Test Set'
    )

    summary = {
        'best_model_path_original': best_model_path,
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_r2': float(r2_score(test_true, test_pred)),
        'test_mae': float(mean_absolute_error(test_true, test_pred)),
        'test_rmse': float(np.sqrt(np.mean((test_true - test_pred) ** 2))),
        'train_r2': float(r2_score(train_true, train_pred)),
        'train_mae': float(mean_absolute_error(train_true, train_pred)),
        'train_rmse': float(np.sqrt(np.mean((train_true - train_pred) ** 2))),
    }
    summary_path = os.path.join(save_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"汇总信息已保存: {summary_path}")

    print(f"\n所有结果已保存到文件夹: {save_dir}")
    return save_dir


class EpochLogCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        header = (
            f"{'Epoch':>6} │ "
            f"{'Train Loss':>10} │ {'Train R²':>8} │ {'Train RMSE':>10} │"
            f"  {'Val Loss':>9} │ {'Val R²':>8} │ {'Val RMSE':>9} │"
            f"  {'LR':>10}"
        )
        sep = "─" * len(header)
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
            f"{epoch:>6} │ "
            f"{fmt(train_loss):>10} │ {fmt(train_r2):>8} │ {fmt(train_rmse):>10} │"
            f"  {fmt(val_loss):>9} │ {fmt(val_r2):>8} │ {fmt(val_rmse):>9} │"
            f"  {lr:.2e}"
        )

    def on_train_end(self, trainer, pl_module):
        print(self._sep + "\n")


class GlobalRMSE(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.detach().float().reshape(-1)
        target = target.detach().float().reshape(-1)
        self.sum_squared_error += torch.sum((preds - target) ** 2)
        self.total += target.numel()

    def compute(self):
        return torch.sqrt(self.sum_squared_error / self.total.clamp_min(1).float())



class DMPNNLightningModule(LightningModule):
    def __init__(self, model_params, scaler_params, lr=5e-4, L2=5e-5):
        super().__init__()
        self.save_hyperparameters()
        self.L2 = L2
        self.model = dmpnn(**model_params)
        self.lr = lr

        self.huber_loss = nn.SmoothL1Loss(beta=0.25)
        self.mse_loss = nn.MSELoss()

        self.register_buffer("label_mean", torch.tensor(scaler_params['label_mean'], dtype=torch.float32))
        self.register_buffer("label_std", torch.tensor(scaler_params['label_std'], dtype=torch.float32))
        self.register_buffer("use_log", torch.tensor(scaler_params.get('use_log', False)))

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()
        self.train_rmse = GlobalRMSE()
        self.val_rmse = GlobalRMSE()
        self.test_rmse = GlobalRMSE()

    def _to_original_scale(self, preds_norm, labels_norm):
        p = preds_norm * self.label_std + self.label_mean
        l = labels_norm * self.label_std + self.label_mean
        if bool(self.use_log):
            p = torch.exp(p)
            l = torch.exp(l)
        return p, l

    def _compute_loss(self, preds, labels):
        huber = self.huber_loss(preds, labels)
        mse = self.mse_loss(preds, labels)
        return 0.8 * huber + 0.2 * mse

    def forward(self, batched_graph, extra_features):
        return self.model(batched_graph, extra_features)

    def training_step(self, batch, batch_idx):
        batched_graph, extra_features, labels = batch
        preds = self(batched_graph, extra_features).squeeze(-1)

        loss = self._compute_loss(preds, labels)

        preds_real, labels_real = self._to_original_scale(preds.detach(), labels.detach())
        mae_real = F.l1_loss(preds_real, labels_real)
        self.train_r2(preds_real, labels_real)
        self.train_rmse(preds_real, labels_real)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log('train_rmse', self.train_rmse, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log('train_mae', mae_real, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log('train_r2', self.train_r2, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(labels))
        return loss

    def validation_step(self, batch, batch_idx):
        batched_graph, extra_features, labels = batch
        preds = self(batched_graph, extra_features).squeeze(-1)

        loss = self._compute_loss(preds, labels)
        preds_real, labels_real = self._to_original_scale(preds, labels)
        mae_real = F.l1_loss(preds_real, labels_real)
        self.val_r2(preds_real, labels_real)
        self.val_rmse(preds_real, labels_real)

        self.log('val_loss', loss, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log('val_rmse', self.val_rmse, on_epoch=True, prog_bar=True, batch_size=len(labels))
        self.log('val_mae', mae_real, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log('val_r2', self.val_r2, on_epoch=True, prog_bar=True, batch_size=len(labels))
        return loss

    def test_step(self, batch, batch_idx):
        batched_graph, extra_features, labels = batch
        preds = self(batched_graph, extra_features).squeeze(-1)

        preds_real, labels_real = self._to_original_scale(preds, labels)
        mae_real = F.l1_loss(preds_real, labels_real)
        self.test_r2(preds_real, labels_real)
        self.test_rmse(preds_real, labels_real)

        self.log('test_rmse', self.test_rmse, on_step=False, on_epoch=True, batch_size=len(labels))
        self.log('test_mae', mae_real, on_step=False, on_epoch=True, batch_size=len(labels))
        self.log('test_r2', self.test_r2, on_step=False, on_epoch=True, batch_size=len(labels))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.L2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=30,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_rmse",
                "interval": "epoch",
                "frequency": 1
            }
        }


def stratified_split(log_labels: np.ndarray, train_ratio=0.8, val_ratio=0.1, seed=Random_seed):
    rng = np.random.default_rng(seed)
    n_bins = 10
    bins = np.percentile(log_labels, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    strata = np.digitize(log_labels, bins) - 1

    train_idx, val_idx, test_idx = [], [], []

    for s in range(n_bins):
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

    return np.array(sorted(train_idx)), np.array(sorted(val_idx)), np.array(sorted(test_idx))


def _scaffold_key(smiles: str) -> str:
    mol = mol_from_smiles_relaxed(smiles)
    if mol is None:
        return f'INVALID::{smiles}'
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaffold = ''
    return scaffold if scaffold else f'NO_SCAFFOLD::{smiles}'


def scaffold_split(smiles_list, train_ratio=0.8, val_ratio=0.1, seed=42):
    total = len(smiles_list)
    if total < 3:
        idx = np.arange(total)
        return idx, np.array([], dtype=int), np.array([], dtype=int)

    groups = {}
    for idx, smiles in enumerate(smiles_list):
        key = _scaffold_key(smiles)
        groups.setdefault(key, []).append(idx)

    rng = np.random.default_rng(seed)
    grouped_indices = list(groups.values())
    rng.shuffle(grouped_indices)
    grouped_indices.sort(key=lambda g: len(g), reverse=True)

    target_train = int(round(total * train_ratio))
    target_val = int(round(total * val_ratio))
    target_test = total - target_train - target_val

    train_idx, val_idx, test_idx = [], [], []

    for group in grouped_indices:
        n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
        deficits = {
            'train': target_train - n_train,
            'val': target_val - n_val,
            'test': target_test - n_test,
        }
        split_name = max(deficits, key=lambda k: (deficits[k], {'train': 2, 'val': 1, 'test': 0}[k]))
        if split_name == 'train':
            train_idx.extend(group)
        elif split_name == 'val':
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    if len(val_idx) == 0 and len(train_idx) > 1:
        val_idx.append(train_idx.pop())
    if len(test_idx) == 0 and len(train_idx) > 1:
        test_idx.append(train_idx.pop())

    return (
        np.array(sorted(train_idx), dtype=int),
        np.array(sorted(val_idx), dtype=int),
        np.array(sorted(test_idx), dtype=int),
    )




def build_weighted_sampler(labels: np.ndarray, n_bins: int = TRAIN_BIN_COUNT, seed: int = Random_seed):
    if len(labels) <= 2:
        return None

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(labels, quantiles)
    edges = np.unique(edges)

    if len(edges) < 3:
        return None

    edges[0] -= 1e-6
    edges[-1] += 1e-6
    bin_ids = np.digitize(labels, edges[1:-1], right=False)

    counts = np.bincount(bin_ids, minlength=max(bin_ids) + 1).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts[bin_ids]
    weights = weights / weights.mean()

    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


def prepare_data(path, batch_size=BATCH_SIZE, split_mode=SPLIT_MODE):
    path = PROJECT_DIR / Path(path).expanduser()
    print(f"正在读取数据: {path}")
    df = pd.read_excel(path)

    if 'log_kisc' not in df.columns:
        raise ValueError("缺少 log_kisc 列")
    df = df.dropna(subset=['smiles', 'log_kisc']).copy()

    features_list = []
    valid_smiles = []
    valid_labels = []

    for idx, (smiles, label) in enumerate(zip(df['smiles'].tolist(), df['log_kisc'].tolist())):
        desc = calculate_descriptors(smiles, solute_descriptor_functions)
        desc_vals = [desc[name] for name in descriptor_names]
        if np.any(np.isnan(desc_vals)) or np.any(np.isinf(desc_vals)):
            print(f"警告：样本 {idx} 的描述符含 nan/inf，跳过")
            continue
        features_list.append(desc_vals)
        valid_smiles.append(smiles)
        valid_labels.append(label)

    extra_features = np.array(features_list, dtype=np.float32)
    extra_features = np.nan_to_num(extra_features, nan=0.0, posinf=0.0, neginf=0.0)

    labels = np.array(valid_labels, dtype=np.float32)
    smiles_list = valid_smiles
    total_len = len(labels)
    print(f"有效样本数: {total_len}，描述符维度: {extra_features.shape[1]}")

    if total_len == 0:
        raise ValueError("无有效样本，无法继续训练")

    split_mode = str(split_mode).lower().strip()
    if split_mode == 'random':
        train_idx, val_idx, test_idx = stratified_split(labels, seed=Random_seed)
        print(f"随机分层划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    elif split_mode == 'scaffold':
        train_idx, val_idx, test_idx = scaffold_split(smiles_list, seed=Random_seed)
        print(f"骨架划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    else:
        raise ValueError(f"未知 split_mode: {split_mode}，可选 random / scaffold")

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        sub = labels[idx]
        print(f"  {name}: min={sub.min():.4f}, max={sub.max():.4f}, mean={sub.mean():.4f}")

    feat_mean = extra_features[train_idx].mean(axis=0)
    feat_std = extra_features[train_idx].std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    extra_features = (extra_features - feat_mean) / feat_std

    X_train = extra_features[train_idx]
    X_val = extra_features[val_idx]
    X_test = extra_features[test_idx]

    work_labels = labels.copy()
    sk = pd.Series(work_labels).skew()
    ku = pd.Series(work_labels).kurtosis()
    print(f"log_kisc 分布 — skewness: {sk:.4f}, kurtosis: {ku:.4f}")

    label_mean = work_labels[train_idx].mean()
    label_std = work_labels[train_idx].std()
    if label_std < 1e-6:
        label_std = 1.0

    y_train = np.nan_to_num((work_labels[train_idx] - label_mean) / label_std, nan=0.0)
    y_val = np.nan_to_num((work_labels[val_idx] - label_mean) / label_std, nan=0.0)
    y_test = np.nan_to_num((work_labels[test_idx] - label_mean) / label_std, nan=0.0)

    scaler_params = {
        'feat_mean': feat_mean.tolist(),
        'feat_std': feat_std.tolist(),
        'label_mean': float(label_mean),
        'label_std': float(label_std),
        'raw_label_mean': float(labels[train_idx].mean()),
        'raw_label_std': float(labels[train_idx].std()),
        'use_log': False,
        'use_extra_features': True,
        'split_mode': split_mode,
        'descriptor_names': descriptor_names,
    }

    with open(PROJECT_DIR / 'scaler_params.json', 'w', encoding='utf-8') as f:
        json.dump(scaler_params, f, ensure_ascii=False, indent=2)

    train_dataset = MoleculeDataset([smiles_list[i] for i in train_idx], X_train, y_train)
    val_dataset = MoleculeDataset([smiles_list[i] for i in val_idx], X_val, y_val)
    test_dataset = MoleculeDataset([smiles_list[i] for i in test_idx], X_test, y_test)

    train_sampler = build_weighted_sampler(work_labels[train_idx], n_bins=TRAIN_BIN_COUNT, seed=Random_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0)
    )

    return train_loader, val_loader, test_loader, scaler_params, train_dataset, val_dataset, test_dataset


def train(split_mode=SPLIT_MODE):
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(Random_seed)
    np.random.seed(Random_seed)

    path = 'KISC_data.xlsx'
    train_loader, val_loader, test_loader, scaler_params, train_dataset, val_dataset, test_dataset = prepare_data(path, BATCH_SIZE, split_mode=split_mode)
    EXTRA_DIM = len(scaler_params['feat_mean'])

    # 自动读取图特征维度
    sample_graph = train_dataset[0][0]
    node_feat_dim = sample_graph.x.shape[1]
    edge_feat_dim = sample_graph.edge_attr.shape[1]
    model_params = {
        'node_feat_dim': node_feat_dim,
        'edge_feat_dim': edge_feat_dim,
        'edge_output_dim': edge_output_dim,
        'node_output_dim': node_output_dim,
        'extra_dim': EXTRA_DIM,
        'num_rounds': num_rounds,
        'dropout_rate': Dropout,
        'activation_type1': "leakyrelu",
        'activation_type2': "leakyrelu",
        'num_experts': num_experts,
        'moe_hid_dim': moe_hid_dim,
    }

    model = DMPNNLightningModule(
        model_params=model_params,
        scaler_params=scaler_params,
        lr=LR,
        L2=L2
    )

    epoch_log_cb = EpochLogCallback()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint_cb = ModelCheckpoint(
        dirpath=PROJECT_DIR / 'checkpoints',
        filename='best_model-{epoch:02d}-{val_rmse:.4f}',
        save_top_k=1,
        monitor='val_rmse',
        mode='min',
    )

    early_stop_cb = EarlyStopping(
        monitor='val_rmse',
        patience=patience,
        mode='min'
    )

    csv_logger = CSVLogger(PROJECT_DIR / 'logs', name="dmpnn_kisc_log")

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, epoch_log_cb],
        logger=csv_logger,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        enable_progress_bar=False,
        deterministic=True,
    )

    print(f"开始训练 (split_mode={split_mode}, 监控 val_rmse)...")
    trainer.fit(model, train_loader, val_loader)
    print(f"\n训练结束。最佳模型: {checkpoint_cb.best_model_path}")

    if checkpoint_cb.best_model_path:
        print("正在进行测试集评估...")
        test_results = trainer.test(model, dataloaders=test_loader, ckpt_path='best')
        final_r2 = test_results[0]['test_r2']
        final_rmse = test_results[0]['test_rmse']

        print("=" * 40)
        print("最终测试集结果 (log_kisc 空间):")
        print(f"  Test R²  : {final_r2:.4f}")
        print(f"  Test RMSE: {final_rmse:.4f}")
        print("=" * 40)

        save_dir = save_all_artifacts(
            best_model_path=checkpoint_cb.best_model_path,
            scaler_params=scaler_params,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            logger_log_dir=trainer.logger.log_dir,
            final_test_r2=final_r2,
            model_params=model_params,
            training_hyperparameters=build_training_hyperparameters(split_mode, model_params),
        )
        print(f"结果归档目录: {save_dir}")


if __name__ == '__main__':
    train()
