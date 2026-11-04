import argparse
import json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from data import MoleculeDataset, collate_fn
from features import calculate_descriptors, solute_descriptor_functions, validate_descriptor_names
from train import DMPNNLightningModule, calculate_applicability_domain


PROJECT_DIR = Path(__file__).resolve().parent


class InferenceModule(DMPNNLightningModule):
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batched_graph, extra_features = batch
        preds = self(batched_graph, extra_features).squeeze(-1)
        preds_real = preds * self.label_std + self.label_mean
        if bool(self.use_log):
            preds_real = torch.exp(preds_real)
        return preds_real.detach().cpu()


def find_latest_checkpoint(results_dir: str = str(PROJECT_DIR / 'results')) -> Path:
    candidates = list(Path(results_dir).glob('*/best_model-*.ckpt'))
    if not candidates:
        raise FileNotFoundError(
            f'在 {results_dir} 下未找到 best_model-*.ckpt，'
            '请先训练模型或显式指定 checkpoint。'
        )
    # copy2 preserves the original checkpoint mtime, so use the archive
    # directory mtime to identify the newest completed result.
    return max(candidates, key=lambda path: path.parent.stat().st_mtime)


def _load_scaler_params(checkpoint_path: str) -> dict:
    # weights_only=True avoids arbitrary pickle object construction while
    # reading metadata from an untrusted checkpoint.
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    hyper = ckpt.get('hyper_parameters', {})
    scaler_params = hyper.get('scaler_params')

    sidecar = Path(checkpoint_path).with_name('scaler_params.json')
    if sidecar.exists():
        sidecar_params = json.loads(sidecar.read_text(encoding='utf-8'))
        if scaler_params is None:
            return sidecar_params
        # A result archive may add non-model metadata (for example the
        # applicability-domain reference set) after training. Core checkpoint
        # values remain authoritative.
        for key in ('training_smiles', 'ad_similarity_threshold'):
            if key in sidecar_params:
                scaler_params[key] = sidecar_params[key]

    if scaler_params is not None:
        return scaler_params

    raise ValueError('无法在 checkpoint 或同目录 scaler_params.json 中找到缩放参数。')


def _prepare_input_dataframe(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if 'smiles' not in df.columns:
        raise ValueError('输入文件必须包含 smiles 列。')
    if df.empty:
        raise ValueError('输入文件不包含任何样本。')
    df = df.copy()
    df['smiles'] = df['smiles'].astype(str).str.strip()
    if (df['smiles'] == '').any():
        raise ValueError('输入文件中存在空白 smiles。')
    return df


def load_model(checkpoint_path: str) -> Tuple[InferenceModule, dict]:
    scaler_params = _load_scaler_params(checkpoint_path)
    model = InferenceModule.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu',
        weights_only=True,
    )
    model.eval()
    model.freeze()
    return model, scaler_params


def build_extra_features(smiles_list, scaler_params):
    descriptor_names_for_model = validate_descriptor_names(scaler_params)
    feat_mean = np.array(scaler_params['feat_mean'], dtype=np.float32)
    feat_std = np.array(scaler_params['feat_std'], dtype=np.float32)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

    descriptor_rows = []
    valid_mask = []
    repaired_rows = 0

    for smiles in smiles_list:
        desc = calculate_descriptors(smiles, solute_descriptor_functions)
        vals = np.array([desc[name] for name in descriptor_names_for_model], dtype=np.float32)
        is_valid = not (np.isnan(vals).any() or np.isinf(vals).any())
        if not is_valid:
            repaired_rows += 1
            # Training skips invalid descriptor rows. For inference, the least
            # distorting fallback is training-mean imputation, which maps to
            # zero after standardization. Raw-zero imputation created extreme
            # z-scores for descriptors whose mean is far from zero.
            vals = np.where(np.isfinite(vals), vals, feat_mean)
        descriptor_rows.append(vals)
        valid_mask.append(is_valid)

    extra = np.vstack(descriptor_rows).astype(np.float32)
    extra = (extra - feat_mean) / feat_std

    if repaired_rows > 0:
        print(
            f'提示：有 {repaired_rows} 条样本的描述符含 nan/inf，'
            '预测时已用训练集均值填补。'
        )
    return extra, np.array(valid_mask, dtype=bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(PROJECT_DIR / 'qband_data(531).xlsx'))
    parser.add_argument('--output', default=str(PROJECT_DIR / 'result.xlsx'))
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='模型 .ckpt 路径；省略时自动使用 results 下最新归档模型。',
    )
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    df = _prepare_input_dataframe(args.input)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else find_latest_checkpoint()
    )
    model, scaler_params = load_model(str(checkpoint_path))

    smiles_list = df['smiles'].tolist()
    extra_features, valid_mask = build_extra_features(smiles_list, scaler_params)

    dataset = MoleculeDataset(
        smiles_list,
        extra_features,
        graph_feature_version=scaler_params.get('graph_feature_version', 'legacy'),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    predictions = trainer.predict(model, dataloaders=dataloader)
    pred_array = torch.cat(predictions).numpy()

    df['pred_q_band'] = pred_array
    df['descriptor_valid'] = valid_mask
    training_smiles = scaler_params.get('training_smiles')
    if training_smiles:
        threshold = float(scaler_params.get('ad_similarity_threshold', 0.75))
        domain = calculate_applicability_domain(
            training_smiles,
            smiles_list,
            similarity_threshold=threshold,
        )
        for name, values in domain.items():
            df[name] = values
        out_of_domain = int((~domain['in_applicability_domain']).sum())
        print(
            f'应用域检查：{out_of_domain}/{len(df)} 条样本位于训练应用域之外 '
            f'(Tanimoto < {threshold:.2f} 或包含训练未见元素)。'
        )

    output_path = Path(args.output)
    if output_path.suffix.lower() in {'.xlsx', '.xls'}:
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    print(f'预测完成，结果已保存到: {output_path}')


if __name__ == '__main__':
    main()
