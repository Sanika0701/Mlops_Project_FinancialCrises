#!/usr/bin/env python3
"""
train_tcn_multi.py

Modifications / behavior:
- Auto-detects all columns starting with "target_" in train CSV and trains one TCN per target.
- Keeps percent-change (QoQ) preprocessing. First pct_change per company is filled with 0.0
  if KEEP_FIRST_AS_ZERO = True.
- Simplified TCN: fewer filters and dilations to reduce chance of training instability.
- Computes metrics in both scaled space and original pct-change units.
- Produces a final summary table (train/test metrics + averages), printed and saved to CSV.
- Minimal changes to your previous pipeline; model, scaler, outlier info, target norm saved per-target.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Conv1D,
                                     Dropout, GlobalAveragePooling1D, Dense)

# -----------------------
# CONFIG: pct-change behavior
# -----------------------
APPLY_PERCENT_CHANGE = True
KEEP_FIRST_AS_ZERO = True
# -----------------------

# ----------------------------
# Utilities: data -> sequences
# ----------------------------


def load_split_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'])
    # ensure sorted per company/time for pct_change grouping
    df = df.sort_values(['Company', 'Date']).reset_index(drop=True)
    return df


def pct_change_grouped(df: pd.DataFrame, by_col: str = "Company", numeric_only: bool = True) -> pd.DataFrame:
    """
    Compute period-over-period percent change per group (Company).
    Replaces numeric columns with pct_change. Non-numeric columns are left alone.
    If KEEP_FIRST_AS_ZERO is True, fills the first NaN (from pct_change) with 0.0.
    """
    out = df.copy()
    if numeric_only:
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = out.columns.tolist()

    out[numeric_cols] = out.groupby(by_col)[numeric_cols].pct_change()

    if KEEP_FIRST_AS_ZERO:
        out[numeric_cols] = out[numeric_cols].fillna(0.0)
    return out


def build_sequences(df: pd.DataFrame, feature_cols: List[str], target_col: str, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X, y sequences from panel DataFrame grouped by Company.
    For each company, for t >= seq_len, X[t] = features[t-seq_len:t], y[t] = target[t]
    """
    X_list, y_list = [], []
    for comp, g in df.groupby('Company'):
        g = g.sort_values('Date')
        missing = [c for c in feature_cols if c not in g.columns]
        if missing:
            # defensive: skip if group lacks feature columns
            continue
        features = g[feature_cols].values
        targets = g[target_col].values
        if len(features) <= seq_len:
            continue
        for end_idx in range(seq_len, len(features)):
            start_idx = end_idx - seq_len
            X_list.append(features[start_idx:end_idx])
            y_list.append(targets[end_idx])
    if not X_list:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,))
    X = np.stack(X_list).astype(np.float64)
    y = np.stack(y_list).astype(np.float32)
    return X, y


# ----------------------------
# TCN building blocks (simplified)
# ----------------------------


def residual_block(x, n_filters, kernel_size, dilation_rate, dropout_rate):
    prev_x = x
    conv1 = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='causal',
                   dilation_rate=dilation_rate)(x)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    do1 = Dropout(dropout_rate)(act1)

    conv2 = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='causal',
                   dilation_rate=dilation_rate)(do1)
    bn2 = BatchNormalization()(conv2)
    # match channels if needed
    if prev_x.shape[-1] != n_filters:
        prev_x = Conv1D(filters=n_filters, kernel_size=1,
                        padding='same')(prev_x)
    out = Add()([prev_x, bn2])
    out = Activation('relu')(out)
    return out


def build_tcn_model(seq_len: int, n_features: int, n_filters: int = 32, kernel_size: int = 2,
                    dilations: List[int] = [1, 2], dropout_rate: float = 0.1,
                    dense_units: int = 32) -> tf.keras.Model:
    """
    Simplified TCN: fewer filters/dilations to reduce overfitting and training instability.
    This structure is intentionally lighter than the original to make training more robust.
    """
    inp = Input(shape=(seq_len, n_features), name='input_seq')
    x = inp
    for d in dilations:
        x = residual_block(x, n_filters=n_filters, kernel_size=kernel_size,
                           dilation_rate=d, dropout_rate=dropout_rate)
    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, activation='linear', name='output')(x)
    model = Model(inputs=inp, outputs=out)
    return model


# ----------------------------
# Custom metrics
# ----------------------------


def r2_keras(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


# ----------------------------
# Preprocessing helpers
# ----------------------------
def flatten_seq(X: np.ndarray) -> np.ndarray:
    """Flatten shape (N, seq_len, n_feat) -> (N*seq_len, n_feat)"""
    if X.size == 0:
        return X.reshape(0, 0)
    return X.reshape(-1, X.shape[-1])


def reshape_back(X_flat: np.ndarray, orig_shape: tuple) -> np.ndarray:
    if X_flat.size == 0:
        return np.empty(orig_shape)
    return X_flat.reshape(orig_shape)


def compute_train_medians_and_clip(X_train_flat: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> Dict[str, np.ndarray]:
    """
    Compute per-feature medians and clip bounds from TRAIN flattened matrix.
    Returns dict with 'median', 'lower', 'upper' arrays.
    """
    X = X_train_flat.astype(np.float64)
    X[np.isposinf(X)] = np.nan
    X[np.isneginf(X)] = np.nan

    medians = np.nanmedian(X, axis=0)
    lower = np.nanpercentile(X, low_q, axis=0)
    upper = np.nanpercentile(X, high_q, axis=0)

    eps = 1e-6
    upper = np.where(upper == lower, lower + eps, upper)

    return {'median': medians.tolist(), 'lower': lower.tolist(), 'upper': upper.tolist()}


def apply_impute_and_clip(X_flat: np.ndarray, medians: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Replace inf with nan, impute NaN with median, then clip between lower/upper.
    """
    X = X_flat.astype(np.float64)
    X[np.isposinf(X)] = np.nan
    X[np.isneginf(X)] = np.nan
    inds = np.where(np.isnan(X))
    if inds[0].size > 0:
        X[inds] = np.take(medians, inds[1])
    X = np.maximum(X, lower)
    X = np.minimum(X, upper)
    return X


# ----------------------------
# Train function for a single target
# ----------------------------
def train_single_target(args, target: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Train one TCN for `target`. Returns a dict of metrics for summary table.
    """
    model_dir = Path(args.model_dir) / target
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Training target: {target} ===")
    # Detect numeric feature columns (from train_df)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_feature_cols = [
        c for c in numeric_cols if not c.startswith('target_')]
    for col_to_exclude in ['Year', 'Quarter_Num', 'Quarter']:
        if col_to_exclude in numeric_feature_cols:
            numeric_feature_cols.remove(col_to_exclude)
    numeric_feature_cols = sorted(numeric_feature_cols)
    feature_cols = numeric_feature_cols

    # Build sequences
    X_train, y_train_raw = build_sequences(
        train_df, feature_cols, target, args.seq_len)
    X_val, y_val_raw = build_sequences(
        val_df, feature_cols, target, args.seq_len)
    X_test, y_test_raw = build_sequences(
        test_df, feature_cols, target, args.seq_len)

    if X_train.shape[0] == 0:
        raise ValueError(
            f"No training sequences for target {target}. Consider lowering --seq_len or checking data.")

    print(
        f"Sequences -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Standardize target (pct-change units)
    y_mean = float(np.nanmean(y_train_raw.astype(np.float64)))
    y_std_raw = float(np.nanstd(y_train_raw.astype(np.float64)))
    y_std = y_std_raw if y_std_raw > 0 else 1.0

    y_train = ((y_train_raw.astype(np.float64) -
               y_mean) / y_std).astype(np.float32)
    y_val = ((y_val_raw.astype(np.float64) - y_mean) / y_std).astype(
        np.float32) if y_val_raw.size else np.array([], dtype=np.float32)
    y_test = ((y_test_raw.astype(np.float64) - y_mean) / y_std).astype(
        np.float32) if y_test_raw.size else np.array([], dtype=np.float32)

    target_norm_info = {
        "log_transform": False,
        "pct_change": True,
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "notes": "Invert: y_orig_pct = y_scaled * y_std + y_mean (e.g., 0.05 == 5% QoQ)"
    }
    print(f"Target mean={y_mean:.6f}, target std={y_std:.6f}")

    # Flatten for preprocessing
    X_train_flat = flatten_seq(X_train)
    X_val_flat = flatten_seq(X_val)
    X_test_flat = flatten_seq(X_test)

    # sanitize infs/nans
    X_train_flat[np.isposinf(X_train_flat)] = np.nan
    X_train_flat[np.isneginf(X_train_flat)] = np.nan
    X_val_flat[np.isposinf(X_val_flat)] = np.nan
    X_val_flat[np.isneginf(X_val_flat)] = np.nan
    X_test_flat[np.isposinf(X_test_flat)] = np.nan
    X_test_flat[np.isneginf(X_test_flat)] = np.nan

    outlier_info = compute_train_medians_and_clip(
        X_train_flat, low_q=1.0, high_q=99.0)
    medians = np.array(outlier_info['median'], dtype=np.float64)
    lower = np.array(outlier_info['lower'], dtype=np.float64)
    upper = np.array(outlier_info['upper'], dtype=np.float64)

    X_train_flat_proc = apply_impute_and_clip(
        X_train_flat, medians, lower, upper)
    X_val_flat_proc = apply_impute_and_clip(
        X_val_flat, medians, lower, upper) if X_val_flat.size else X_val_flat
    X_test_flat_proc = apply_impute_and_clip(
        X_test_flat, medians, lower, upper) if X_test_flat.size else X_test_flat

    scaler = StandardScaler()
    scaler.fit(X_train_flat_proc)

    X_train_flat_scaled = scaler.transform(X_train_flat_proc)
    X_val_flat_scaled = scaler.transform(
        X_val_flat_proc) if X_val_flat_proc.size else X_val_flat_proc
    X_test_flat_scaled = scaler.transform(
        X_test_flat_proc) if X_test_flat_proc.size else X_test_flat_proc

    X_train_seq = reshape_back(
        X_train_flat_scaled.astype(np.float32), X_train.shape)
    X_val_seq = reshape_back(X_val_flat_scaled.astype(
        np.float32), X_val.shape) if X_val_flat_scaled.size else X_val
    X_test_seq = reshape_back(X_test_flat_scaled.astype(
        np.float32), X_test.shape) if X_test_flat_scaled.size else X_test

    n_feat = X_train_seq.shape[-1]

    # Save outlier info
    outlier_file = model_dir / f'outlier_info_{target}.json'
    with open(outlier_file, 'w') as f:
        json.dump(outlier_info, f, indent=2)

    # Build simplified model
    tf.keras.backend.clear_session()
    model = build_tcn_model(seq_len=args.seq_len, n_features=n_feat,
                            n_filters=args.n_filters, kernel_size=args.kernel_size,
                            dilations=args.dilations, dropout_rate=args.dropout, dense_units=args.dense_units)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', r2_keras])
    model.summary()

    # callbacks
    model_file = model_dir / f"tcn_{target}.h5"
    ckpt = ModelCheckpoint(filepath=str(model_file),
                           save_best_only=True, monitor='val_loss', verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=args.patience,
                       verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # MLflow logging
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=f"tcn_{target}"):
        mlflow.log_params({
            'target': target,
            'seq_len': args.seq_len,
            'n_filters': args.n_filters,
            'kernel_size': args.kernel_size,
            'dilations': args.dilations,
            'dropout': args.dropout,
            'dense_units': args.dense_units,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'target_pct_change': True
        })

        history = model.fit(
            X_train_seq, y_train,
            validation_data=(X_val_seq, y_val) if y_val.size else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[ckpt, es, reduce_lr],
            verbose=2
        )

        # evaluate scaled metrics
        val_metrics_scaled = model.evaluate(
            X_val_seq, y_val, verbose=0) if y_val.size else []
        test_metrics_scaled = model.evaluate(
            X_test_seq, y_test, verbose=0) if y_test.size else []

        metric_names = ['loss', 'mae', 'r2_keras']
        if val_metrics_scaled:
            mlflow.log_metrics({f"val_{n}": float(v)
                               for n, v in zip(metric_names, val_metrics_scaled)})
        if test_metrics_scaled:
            mlflow.log_metrics({f"test_{n}": float(v)
                               for n, v in zip(metric_names, test_metrics_scaled)})

        # save scaler & feature list & target norm
        scaler_path = model_dir / f"scaler_{target}.pkl"
        joblib.dump(scaler, scaler_path)
        features_path = model_dir / f"features_{target}.json"
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f)
        target_norm_path = model_dir / f"target_norm_{target}.json"
        with open(target_norm_path, 'w') as f:
            json.dump(target_norm_info, f, indent=2)

        try:
            mlflow.log_artifact(str(model_file))
        except Exception:
            pass
        mlflow.log_artifact(str(scaler_path))
        mlflow.log_artifact(str(features_path))
        mlflow.log_artifact(str(outlier_file))
        mlflow.log_artifact(str(target_norm_path))

    # Post-train: load best model and compute original-unit metrics
    metrics_record = {
        'target': target,
        'train_mse_scaled': None,
        'train_mae_scaled': None,
        'train_r2_scaled': None,
        'test_mse_scaled': None,
        'test_mae_scaled': None,
        'test_r2_scaled': None,
        'train_mse_pct': None,
        'train_mae_pct': None,
        'train_r2_pct': None,
        'test_mse_pct': None,
        'test_mae_pct': None,
        'test_r2_pct': None,
        'count_train': int(y_train.size),
        'count_val': int(y_val.size),
        'count_test': int(y_test.size)
    }

    if model_file.exists():
        best_model = tf.keras.models.load_model(
            str(model_file), custom_objects={'r2_keras': r2_keras})

        def invert_target(y_scaled: np.ndarray) -> np.ndarray:
            return (y_scaled * y_std) + y_mean

        # helper to compute original-unit metrics
        def safe_metrics(y_true_scaled, y_pred_scaled):
            if y_pred_scaled is None or y_pred_scaled.size == 0 or y_true_scaled is None or y_true_scaled.size == 0:
                return None, None, None
            y_pred_scaled = np.asarray(y_pred_scaled).ravel()
            y_true_scaled = np.asarray(y_true_scaled).ravel()
            try:
                y_pred_orig = invert_target(y_pred_scaled)
            except Exception as e:
                print("Error inverting preds:", e)
                y_pred_orig = np.full_like(
                    y_pred_scaled, np.nan, dtype=np.float64)
            try:
                y_true_orig = invert_target(y_true_scaled)
            except Exception as e:
                print("Error inverting truth:", e)
                y_true_orig = np.full_like(
                    y_true_scaled, np.nan, dtype=np.float64)

            y_pred_orig[~np.isfinite(y_pred_orig)] = np.nan
            y_true_orig[~np.isfinite(y_true_orig)] = np.nan

            valid = (~np.isnan(y_true_orig)) & (~np.isnan(y_pred_orig)
                                                ) & np.isfinite(y_true_orig) & np.isfinite(y_pred_orig)
            if valid.sum() == 0:
                return None, None, None
            y_t = y_true_orig[valid]
            y_p = y_pred_orig[valid]
            mse = float(mean_squared_error(y_t, y_p))
            mae = float(mean_absolute_error(y_t, y_p))
            r2 = float(r2_score(y_t, y_p))
            return mse, mae, r2

        # TRAIN metrics (both scaled and pct units)
        y_train_pred_scaled = best_model.predict(
            X_train_seq, verbose=0).squeeze()
        # scaled metrics (train)
        if y_train.size:
            train_mse_scaled = float(
                mean_squared_error(y_train, y_train_pred_scaled))
            train_mae_scaled = float(
                mean_absolute_error(y_train, y_train_pred_scaled))
            train_r2_scaled = float(r2_score(y_train, y_train_pred_scaled))
            metrics_record['train_mse_scaled'] = train_mse_scaled
            metrics_record['train_mae_scaled'] = train_mae_scaled
            metrics_record['train_r2_scaled'] = train_r2_scaled
        # pct-change (original) train metrics
        train_mse_pct, train_mae_pct, train_r2_pct = safe_metrics(
            y_train, y_train_pred_scaled)
        metrics_record['train_mse_pct'] = train_mse_pct
        metrics_record['train_mae_pct'] = train_mae_pct
        metrics_record['train_r2_pct'] = train_r2_pct

        # TEST metrics
        if y_test.size:
            y_test_pred_scaled = best_model.predict(
                X_test_seq, verbose=0).squeeze()
            test_mse_scaled = float(
                mean_squared_error(y_test, y_test_pred_scaled))
            test_mae_scaled = float(
                mean_absolute_error(y_test, y_test_pred_scaled))
            test_r2_scaled = float(r2_score(y_test, y_test_pred_scaled))
            metrics_record['test_mse_scaled'] = test_mse_scaled
            metrics_record['test_mae_scaled'] = test_mae_scaled
            metrics_record['test_r2_scaled'] = test_r2_scaled

            test_mse_pct, test_mae_pct, test_r2_pct = safe_metrics(
                y_test, y_test_pred_scaled)
            metrics_record['test_mse_pct'] = test_mse_pct
            metrics_record['test_mae_pct'] = test_mae_pct
            metrics_record['test_r2_pct'] = test_r2_pct
        else:
            print("No test sequences to compute test metrics for", target)

    else:
        print("No saved model found for target:", target)

    # save metrics json
    metrics_path = model_dir / f"metrics_{target}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_record, f, indent=2, default=lambda x: None)

    return metrics_record


# ----------------------------
# Helper to auto-detect targets
# ----------------------------
def get_all_targets_from_csv(path: str) -> List[str]:
    df = pd.read_csv(path, nrows=5)  # read small slice just for columns
    return sorted([c for c in df.columns if c.startswith('target_')])


# ----------------------------
# CLI entrypoint and main loop
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/splits',
                        help='Directory with train/val/test CSVs')
    parser.add_argument('--target', type=str, default=None,
                        help='OPTIONAL: If provided, train only this target (e.g., target_revenue). If omitted, auto-train all target_* columns')
    parser.add_argument('--seq_len', type=int, default=4,
                        help='Sequence length (quarters)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--model_dir', type=str, default='models/tcn')
    parser.add_argument('--mlflow_experiment', type=str,
                        default='tcn_experiment')
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dense_units', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dilations', type=str, default='1,2',
                        help='Comma-separated dilation rates (e.g., "1,2")')
    args = parser.parse_args()
    args.dilations = [int(x) for x in args.dilations.split(',') if x.strip()]
    return args


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / 'train_data.csv'
    val_path = data_dir / 'val_data.csv'
    test_path = data_dir / 'test_data.csv'

    if not train_path.exists():
        raise FileNotFoundError(f"Train CSV not found at {train_path}")

    # find targets
    if args.target:
        target_list = [args.target]
    else:
        target_list = get_all_targets_from_csv(str(train_path))
        if not target_list:
            raise ValueError(
                "No columns starting with 'target_' detected in train CSV.")

    print("Targets to train:", target_list)

    # load full splits once
    train_df = load_split_csv(str(train_path))
    val_df = load_split_csv(str(val_path)) if val_path.exists(
    ) else pd.DataFrame(columns=train_df.columns)
    test_df = load_split_csv(str(test_path)) if test_path.exists(
    ) else pd.DataFrame(columns=train_df.columns)

    # optionally apply pct-change before training (applied to ALL numeric columns)
    if APPLY_PERCENT_CHANGE:
        print("Applying quarter-over-quarter percent-change to numeric columns (grouped by Company)...")
        train_df = pct_change_grouped(train_df, by_col='Company', numeric_only=True).sort_values(
            ['Company', 'Date']).reset_index(drop=True)
        val_df = pct_change_grouped(val_df, by_col='Company', numeric_only=True).sort_values(
            ['Company', 'Date']).reset_index(drop=True)
        test_df = pct_change_grouped(test_df, by_col='Company', numeric_only=True).sort_values(
            ['Company', 'Date']).reset_index(drop=True)
        print("Percent-change applied.")

    all_metrics = []
    for tgt in target_list:
        try:
            metrics = train_single_target(args, tgt, train_df, val_df, test_df)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error training target {tgt}: {e}")
            # continue to next target (partial completion is better than failing all)
            continue

    # Build summary DataFrame
    if not all_metrics:
        print("No metrics were generated (all targets failed). Exiting.")
        return

    df_metrics = pd.DataFrame(all_metrics)
    # Friendly column order
    cols_order = [
        'target', 'train_mse_pct', 'train_mae_pct', 'train_r2_pct',
        'test_mse_pct', 'test_mae_pct', 'test_r2_pct',
        'train_mse_scaled', 'train_mae_scaled', 'train_r2_scaled',
        'test_mse_scaled', 'test_mae_scaled', 'test_r2_scaled'
    ]
    # retain only columns present
    cols_order = [c for c in cols_order if c in df_metrics.columns]
    df_metrics = df_metrics[cols_order]

    # compute averages across targets (ignore None)
    avg = df_metrics.select_dtypes(include=[np.number]).mean(skipna=True)
    avg_row = pd.Series({k: (v if not np.isnan(v) else None)
                        for k, v in avg.items()})
    avg_row['target'] = 'AVERAGE'
    # put average at bottom
    df_summary = pd.concat([df_metrics, pd.DataFrame(
        [avg_row])], ignore_index=True, sort=False)

    # print neat table
    pd.set_option('display.float_format',
                  lambda x: f"{x:.6f}" if pd.notna(x) else "NaN")
    print("\n=== Per-target metrics ===")
    print(df_summary.to_string(index=False))

    # save CSV to model_dir root
    model_root = Path(args.model_dir)
    model_root.mkdir(parents=True, exist_ok=True)
    summary_csv = model_root / 'summary_metrics.csv'
    df_summary.to_csv(summary_csv, index=False)
    print("Saved summary metrics to:", summary_csv)


if __name__ == '__main__':
    main()
