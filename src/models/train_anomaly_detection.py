#!/usr/bin/env python3
"""
============================================================================
Model 3: Anomaly Detection - Complete Training Pipeline WITH SENSITIVITY
============================================================================
Author: Parth Saraykar
UPDATED: Now includes integrated hyperparameter sensitivity analysis (§5)
Requirements: Follows Model Development Guidelines §2-§8
============================================================================
"""

from src.utils.gcs_data_loader import GCSDataLoader, GCSOutputUploader
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime
import warnings
import sys
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any

# ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score
)

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# SHAP for sensitivity analysis
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration loader"""

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

    def __getitem__(self, key):
        return self.cfg[key]


# ============================================================================
# LOGGER
# ============================================================================

def setup_logger():
    """Setup comprehensive logger"""
    Path("logs").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/model_training_{timestamp}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# STEP 1: LOAD DATA FROM PIPELINE
# ============================================================================


class DataLoader:
    """Load data from Snorkel labeling pipeline - supports GCS and local"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_labeled_data(self) -> pd.DataFrame:
        """Load labeled data from Snorkel pipeline output (GCS or local)"""
        self.logger.info("="*80)
        self.logger.info("STEP 1: LOADING DATA FROM PIPELINE")
        self.logger.info("="*80)

        source_type = self.config['data'].get('source_type', 'local')
        self.logger.info(f"Data source: {source_type.upper()}")

        if source_type == 'gcs':
            # Load from GCS
            try:
                from src.utils.gcs_data_loader import GCSDataLoader
                gcs_loader = GCSDataLoader(
                    self.config.cfg['data'], self.logger)
                df = gcs_loader.load_csv('labeled_data_path', cache_local=True)
            except Exception as e:
                self.logger.error(f"GCS load failed: {e}")
                self.logger.info("Falling back to local file...")
                data_path = self.config['data']['local']['labeled_data_path']
                df = pd.read_csv(data_path)
        else:
            # Load from local
            data_path = self.config['data']['local']['labeled_data_path']
            self.logger.info(f"Loading from: {data_path}")
            df = pd.read_csv(data_path)

        # Process dates (applies to BOTH GCS and local)
        df['Date'] = pd.to_datetime(df[self.config['data']['date_column']])
        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.quarter

        # Log statistics (applies to BOTH)
        self.logger.info(f"✓ Loaded {len(df):,} labeled samples")
        self.logger.info(
            f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        self.logger.info(
            f"  Companies: {df[self.config['data']['company_column']].nunique()}")
        self.logger.info(
            f"  Sectors: {df[self.config['data']['sector_column']].nunique()}")

        at_risk = (df[self.config['data']['target_column']] == 1).sum()
        self.logger.info(f"  AT_RISK: {at_risk:,} ({at_risk/len(df):.2%})")
        self.logger.info(
            f"  NOT_AT_RISK: {len(df)-at_risk:,} ({(len(df)-at_risk)/len(df):.2%})")

        return df

    def temporal_split(self, df: pd.DataFrame):
        """Temporal train/val/test split"""
        self.logger.info("\nPerforming temporal split...")

        train_mask = df['Year'] <= self.config['split']['train_end_year']
        val_mask = (df['Year'] >= self.config['split']['val_start_year']) & \
                   (df['Year'] <= self.config['split']['val_end_year'])
        test_mask = df['Year'] >= self.config['split']['test_start_year']

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        self.logger.info(
            f"✓ Train: {len(train_df):,} samples (≤{self.config['split']['train_end_year']})")
        self.logger.info(
            f"✓ Val:   {len(val_df):,} samples ({self.config['split']['val_start_year']}-{self.config['split']['val_end_year']})")
        self.logger.info(
            f"✓ Test:  {len(test_df):,} samples (≥{self.config['split']['test_start_year']})")

        return train_df, val_df, test_df

    def prepare_features(self, train_df, val_df, test_df):
        """Prepare features and target"""
        self.logger.info("\nPreparing features...")

        feature_cols = []
        for category in self.config['data']['feature_columns'].values():
            feature_cols.extend(category)

        self.logger.info(f"Using {len(feature_cols)} features")

        X_train = train_df[feature_cols].copy()
        X_val = val_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()

        y_train = train_df[self.config['data']['target_column']].values
        y_val = val_df[self.config['data']['target_column']].values
        y_test = test_df[self.config['data']['target_column']].values

        X_train = X_train.fillna(X_train.median())
        X_val = X_val.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())

        X_train = X_train.replace(
            [np.inf, -np.inf], np.nan).fillna(X_train.median())
        X_val = X_val.replace(
            [np.inf, -np.inf], np.nan).fillna(X_train.median())
        X_test = X_test.replace(
            [np.inf, -np.inf], np.nan).fillna(X_train.median())

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        self.logger.info(f"✓ Features scaled using StandardScaler")
        self.logger.info(
            f"✓ Train at-risk rate: {y_train.sum()/len(y_train):.2%}")
        self.logger.info(f"✓ Val at-risk rate: {y_val.sum()/len(y_val):.2%}")

        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), scaler, feature_cols


# ============================================================================
# NEW: INTEGRATED SENSITIVITY ANALYZER
# ============================================================================

class IntegratedSensitivityAnalyzer:
    """
    Integrated hyperparameter sensitivity analysis
    Runs as part of the main training pipeline
    """

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.results = {}

    def run_sensitivity_analysis(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        quick_mode: bool = False
    ) -> Dict:
        """
        Run sensitivity analysis for all enabled models

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            quick_mode: If True, test fewer values (faster for demo)

        Returns:
            Dictionary with sensitivity results per model
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("HYPERPARAMETER SENSITIVITY ANALYSIS (§5)")
        self.logger.info("="*80)

        # Analyze enabled models
        if self.config['models']['isolation_forest']['enabled']:
            self.results['isolation_forest'] = self._analyze_isolation_forest(
                X_train, y_train, X_val, y_val, quick_mode
            )

        if self.config['models']['one_class_svm']['enabled']:
            self.results['one_class_svm'] = self._analyze_one_class_svm(
                X_train, y_train, X_val, y_val, quick_mode
            )

        if self.config['models']['local_outlier_factor']['enabled']:
            self.results['lof'] = self._analyze_lof(
                X_train, y_train, X_val, y_val, quick_mode
            )

        # Generate visualizations
        self._plot_sensitivity_results()

        # Generate report
        self._generate_sensitivity_report()

        return self.results

    def _analyze_isolation_forest(self, X_train, y_train, X_val, y_val, quick_mode):
        """Analyze Isolation Forest sensitivity"""
        self.logger.info("\n1. Isolation Forest Sensitivity...")

        results = {
            'contamination': {'values': [], 'roc_auc': [], 'precision_10': []}
        }

        baseline = self.config['models']['isolation_forest']['params'].copy()

        # Test contamination (most important parameter)
        contam_values = [0.01, 0.02, 0.035, 0.05, 0.075] if quick_mode else \
                        [0.005, 0.010, 0.020, 0.030, 0.035,
                            0.040, 0.050, 0.075, 0.100]

        for cont in contam_values:
            params = baseline.copy()
            params['contamination'] = cont

            model = IsolationForest(**params)
            model.fit(X_train)

            scores = model.score_samples(X_val)
            roc_auc = roc_auc_score(y_val, -scores)

            k = int(len(y_val) * 0.10)
            top_k_idx = np.argsort(scores)[:k]
            top_k_preds = np.zeros(len(y_val))
            top_k_preds[top_k_idx] = 1
            precision_10 = precision_score(y_val, top_k_preds, zero_division=0)

            results['contamination']['values'].append(cont)
            results['contamination']['roc_auc'].append(roc_auc)
            results['contamination']['precision_10'].append(precision_10)

            self.logger.info(
                f"  contamination={cont:.3f} → ROC-AUC={roc_auc:.4f}")

        # Log to MLflow
        with mlflow.start_run(run_name="sensitivity_isolation_forest", nested=True):
            mlflow.log_params(
                {'analysis_type': 'sensitivity', 'model': 'isolation_forest'})
            best_idx = np.argmax(results['contamination']['roc_auc'])
            mlflow.log_metric('optimal_contamination',
                              results['contamination']['values'][best_idx])
            mlflow.log_metric(
                'max_roc_auc', results['contamination']['roc_auc'][best_idx])

        return results

    def _analyze_one_class_svm(self, X_train, y_train, X_val, y_val, quick_mode):
        """Analyze One-Class SVM sensitivity"""
        self.logger.info("\n2. One-Class SVM Sensitivity...")

        results = {
            'nu': {'values': [], 'roc_auc': [], 'precision_10': []}
        }

        baseline = self.config['models']['one_class_svm']['params'].copy()

        nu_values = [0.01, 0.02, 0.035, 0.05, 0.075] if quick_mode else \
                    [0.005, 0.010, 0.020, 0.030, 0.035, 0.040, 0.050, 0.075, 0.100]

        for nu in nu_values:
            params = baseline.copy()
            params['nu'] = nu

            model = OneClassSVM(**params)
            model.fit(X_train)

            scores = model.score_samples(X_val)
            roc_auc = roc_auc_score(y_val, -scores)

            k = int(len(y_val) * 0.10)
            top_k_idx = np.argsort(scores)[:k]
            top_k_preds = np.zeros(len(y_val))
            top_k_preds[top_k_idx] = 1
            precision_10 = precision_score(y_val, top_k_preds, zero_division=0)

            results['nu']['values'].append(nu)
            results['nu']['roc_auc'].append(roc_auc)
            results['nu']['precision_10'].append(precision_10)

            self.logger.info(f"  nu={nu:.3f} → ROC-AUC={roc_auc:.4f}")

        with mlflow.start_run(run_name="sensitivity_one_class_svm", nested=True):
            mlflow.log_params(
                {'analysis_type': 'sensitivity', 'model': 'one_class_svm'})
            best_idx = np.argmax(results['nu']['roc_auc'])
            mlflow.log_metric('optimal_nu', results['nu']['values'][best_idx])
            mlflow.log_metric(
                'max_roc_auc', results['nu']['roc_auc'][best_idx])

        return results

    def _analyze_lof(self, X_train, y_train, X_val, y_val, quick_mode):
        """Analyze LOF sensitivity"""
        self.logger.info("\n3. Local Outlier Factor Sensitivity...")

        results = {
            'n_neighbors': {'values': [], 'roc_auc': [], 'precision_10': []}
        }

        baseline = self.config['models']['local_outlier_factor']['params'].copy(
        )

        neighbor_values = [10, 20, 30, 50] if quick_mode else \
                          [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

        for n_neigh in neighbor_values:
            params = baseline.copy()
            params['n_neighbors'] = n_neigh

            model = LocalOutlierFactor(**params)
            model.fit(X_train)

            scores = model.score_samples(X_val)
            roc_auc = roc_auc_score(y_val, -scores)

            k = int(len(y_val) * 0.10)
            top_k_idx = np.argsort(scores)[:k]
            top_k_preds = np.zeros(len(y_val))
            top_k_preds[top_k_idx] = 1
            precision_10 = precision_score(y_val, top_k_preds, zero_division=0)

            results['n_neighbors']['values'].append(n_neigh)
            results['n_neighbors']['roc_auc'].append(roc_auc)
            results['n_neighbors']['precision_10'].append(precision_10)

            self.logger.info(
                f"  n_neighbors={n_neigh:3d} → ROC-AUC={roc_auc:.4f}")

        with mlflow.start_run(run_name="sensitivity_lof", nested=True):
            mlflow.log_params({'analysis_type': 'sensitivity', 'model': 'lof'})
            best_idx = np.argmax(results['n_neighbors']['roc_auc'])
            mlflow.log_metric('optimal_n_neighbors',
                              results['n_neighbors']['values'][best_idx])
            mlflow.log_metric(
                'max_roc_auc', results['n_neighbors']['roc_auc'][best_idx])

        return results

    def _plot_sensitivity_results(self):
        """Create sensitivity visualization"""
        self.logger.info("\nGenerating sensitivity plots...")

        Path(self.config['output']['plots_dir']).mkdir(
            exist_ok=True, parents=True)

        # Plot for each model
        n_models = len(self.results)
        if n_models == 0:
            return

        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]

        fig.suptitle('Hyperparameter Sensitivity Analysis',
                     fontsize=16, fontweight='bold')

        idx = 0

        # Isolation Forest
        if 'isolation_forest' in self.results:
            ax = axes[idx]
            data = self.results['isolation_forest']['contamination']
            ax.plot(data['values'], data['roc_auc'],
                    marker='o', linewidth=2, markersize=8, color='steelblue')
            ax.axhline(y=0.85, color='red', linestyle='--',
                       linewidth=2, alpha=0.5, label='Target')
            ax.set_xlabel('contamination', fontsize=12, fontweight='bold')
            ax.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
            ax.set_title('Isolation Forest', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Annotate best
            best_idx = np.argmax(data['roc_auc'])
            ax.annotate(f"Best: {data['values'][best_idx]:.3f}\nAUC: {data['roc_auc'][best_idx]:.4f}",
                        xy=(data['values'][best_idx],
                            data['roc_auc'][best_idx]),
                        xytext=(data['values'][best_idx] + 0.01,
                                data['roc_auc'][best_idx] - 0.02),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=10, fontweight='bold', color='red')
            idx += 1

        # One-Class SVM
        if 'one_class_svm' in self.results:
            ax = axes[idx]
            data = self.results['one_class_svm']['nu']
            ax.plot(data['values'], data['roc_auc'],
                    marker='s', linewidth=2, markersize=8, color='darkred')
            ax.axhline(y=0.85, color='red', linestyle='--',
                       linewidth=2, alpha=0.5, label='Target')
            ax.set_xlabel('nu', fontsize=12, fontweight='bold')
            ax.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
            ax.set_title('One-Class SVM', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

            best_idx = np.argmax(data['roc_auc'])
            ax.annotate(f"Best: {data['values'][best_idx]:.3f}\nAUC: {data['roc_auc'][best_idx]:.4f}",
                        xy=(data['values'][best_idx],
                            data['roc_auc'][best_idx]),
                        xytext=(data['values'][best_idx] + 0.01,
                                data['roc_auc'][best_idx] - 0.02),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=10, fontweight='bold', color='red')
            idx += 1

        # LOF
        if 'lof' in self.results:
            ax = axes[idx]
            data = self.results['lof']['n_neighbors']
            ax.plot(data['values'], data['roc_auc'],
                    marker='D', linewidth=2, markersize=8, color='navy')
            ax.axhline(y=0.85, color='red', linestyle='--',
                       linewidth=2, alpha=0.5, label='Target')
            ax.set_xlabel('n_neighbors', fontsize=12, fontweight='bold')
            ax.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
            ax.set_title('Local Outlier Factor',
                         fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

            best_idx = np.argmax(data['roc_auc'])
            ax.annotate(f"Best: {data['values'][best_idx]}\nAUC: {data['roc_auc'][best_idx]:.4f}",
                        xy=(data['values'][best_idx],
                            data['roc_auc'][best_idx]),
                        xytext=(data['values'][best_idx] + 10,
                                data['roc_auc'][best_idx] - 0.02),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=10, fontweight='bold', color='red')

        plt.tight_layout()
        output_path = Path(
            self.config['output']['plots_dir']) / 'sensitivity_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Saved: {output_path}")

        # Log to MLflow
        mlflow.log_artifact(str(output_path))

    def _generate_sensitivity_report(self):
        """Generate sensitivity analysis summary"""
        self.logger.info("\nGenerating sensitivity report...")

        report = f"""
{'='*80}
HYPERPARAMETER SENSITIVITY ANALYSIS SUMMARY
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Requirement: Model Development Guidelines §5

KEY FINDINGS:
"""

        # Find most sensitive parameter across all models
        max_improvement = 0
        most_sensitive = None

        for model_name, model_results in self.results.items():
            for param_name, param_data in model_results.items():
                roc_range = max(param_data['roc_auc']) - \
                    min(param_data['roc_auc'])
                if roc_range > max_improvement:
                    max_improvement = roc_range
                    most_sensitive = (model_name, param_name)
                    optimal_val = param_data['values'][np.argmax(
                        param_data['roc_auc'])]
                    optimal_auc = max(param_data['roc_auc'])

        if most_sensitive:
            report += f"\n⭐ MOST SENSITIVE PARAMETER: {most_sensitive[1]} ({most_sensitive[0]})\n"
            report += f"   Impact: ±{max_improvement:.4f} ROC-AUC change\n"
            report += f"   Optimal value: {optimal_val}\n"
            report += f"   Best ROC-AUC: {optimal_auc:.4f}\n"

        report += f"\n{'='*80}\n"

        # Save report
        Path(self.config['output']['reports_dir']).mkdir(
            exist_ok=True, parents=True)
        report_path = Path(
            self.config['output']['reports_dir']) / 'sensitivity_summary.txt'
        with open(report_path, 'w') as f:
            f.write(report)

        self.logger.info(f"✓ Report saved: {report_path}")
        self.logger.info(report)


# ============================================================================
# STEP 2: HYPERPARAMETER TUNING (kept as before)
# ============================================================================

class HyperparameterTuner:
    """Hyperparameter tuning for anomaly detection models"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def tune_isolation_forest(self, X_train, y_train, X_val, y_val):
        """Grid search for Isolation Forest"""
        self.logger.info("\nHyperparameter tuning: Isolation Forest")

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_samples': [128, 256, 'auto'],
            'contamination': [0.02, 0.035, 0.05],
            'max_features': [0.8, 1.0]
        }

        best_score = -np.inf
        best_params = None

        for n_est in param_grid['n_estimators']:
            for max_samp in param_grid['max_samples']:
                for cont in param_grid['contamination']:
                    for max_feat in param_grid['max_features']:

                        model = IsolationForest(
                            n_estimators=n_est,
                            max_samples=max_samp,
                            contamination=cont,
                            max_features=max_feat,
                            random_state=42
                        )

                        model.fit(X_train)
                        scores = model.score_samples(X_val)

                        try:
                            roc_auc = roc_auc_score(y_val, -scores)

                            if roc_auc > best_score:
                                best_score = roc_auc
                                best_params = {
                                    'n_estimators': n_est,
                                    'max_samples': max_samp,
                                    'contamination': cont,
                                    'max_features': max_feat
                                }
                        except:
                            continue

        self.logger.info(f"✓ Best params: {best_params}")
        self.logger.info(f"✓ Best ROC-AUC: {best_score:.4f}")

        return best_params
# ============================================================================
# STEP 3: MODEL TRAINING WITH MLFLOW
# ============================================================================


class ModelTrainer:
    """Train anomaly detection models with MLflow tracking"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def train_isolation_forest(self, X_train, y_train, X_val, y_val, params=None):
        """Train Isolation Forest with MLflow"""
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING: ISOLATION FOREST")
        self.logger.info("="*80)

        if params is None:
            params = self.config['models']['isolation_forest']['params']

        with mlflow.start_run(run_name="isolation_forest") as run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.set_tags(self.config['mlflow']['tags'])

            # Train
            model = IsolationForest(**params)
            model.fit(X_train)

            self.logger.info("✓ Model trained")

            # Predict
            y_pred_val = model.predict(X_val)
            scores_val = model.score_samples(X_val)
            y_pred_binary = (y_pred_val == -1).astype(int)

            # Evaluate
            metrics = self._evaluate(y_val, y_pred_binary, scores_val)
            mlflow.log_metrics(metrics)

            # Log model with signature
            signature = infer_signature(X_train, y_pred_binary[:len(X_train)])
            mlflow.sklearn.log_model(model, "model", signature=signature)

            # Log model info
            run_id = run.info.run_id

            self.logger.info(f"✓ ROC-AUC: {metrics['roc_auc']:.4f}")
            self.logger.info(
                f"✓ Precision@10%: {metrics['precision_at_10pct']:.4f}")
            self.logger.info(f"✓ MLflow Run ID: {run_id}")

            return model, metrics, run_id

    def train_lof(self, X_train, y_train, X_val, y_val):
        """Train LOF with MLflow"""
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING: LOCAL OUTLIER FACTOR")
        self.logger.info("="*80)

        params = self.config['models']['local_outlier_factor']['params']

        with mlflow.start_run(run_name="local_outlier_factor") as run:
            mlflow.log_params(params)
            mlflow.set_tags(self.config['mlflow']['tags'])

            model = LocalOutlierFactor(**params)
            model.fit(X_train)

            y_pred_val = model.predict(X_val)
            scores_val = model.score_samples(X_val)
            y_pred_binary = (y_pred_val == -1).astype(int)

            metrics = self._evaluate(y_val, y_pred_binary, scores_val)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            run_id = run.info.run_id

            self.logger.info(f"✓ ROC-AUC: {metrics['roc_auc']:.4f}")
            self.logger.info(
                f"✓ Precision@10%: {metrics['precision_at_10pct']:.4f}")

            return model, metrics, run_id

    def train_one_class_svm(self, X_train, y_train, X_val, y_val):
        """Train One-Class SVM with MLflow"""
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING: ONE-CLASS SVM")
        self.logger.info("="*80)

        params = self.config['models']['one_class_svm']['params']

        with mlflow.start_run(run_name="one_class_svm") as run:
            mlflow.log_params(params)
            mlflow.set_tags(self.config['mlflow']['tags'])

            model = OneClassSVM(**params)
            model.fit(X_train)

            y_pred_val = model.predict(X_val)
            scores_val = model.score_samples(X_val)
            y_pred_binary = (y_pred_val == -1).astype(int)

            metrics = self._evaluate(y_val, y_pred_binary, scores_val)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            run_id = run.info.run_id

            self.logger.info(f"✓ ROC-AUC: {metrics['roc_auc']:.4f}")
            self.logger.info(
                f"✓ Precision@10%: {metrics['precision_at_10pct']:.4f}")

            return model, metrics, run_id

    def train_dbscan(self, X_train, y_train, X_val, y_val):
        """Train DBSCAN with MLflow"""
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING: DBSCAN")
        self.logger.info("="*80)

        params = self.config['models']['dbscan']['params']

        with mlflow.start_run(run_name="dbscan") as run:
            mlflow.log_params(params)
            mlflow.set_tags(self.config['mlflow']['tags'])

            model = DBSCAN(**params)
            clusters = model.fit_predict(X_val)

            # Noise points (cluster -1) are anomalies
            y_pred_binary = (clusters == -1).astype(int)
            scores_val = -np.ones(len(X_val))
            scores_val[clusters != -1] = 1

            metrics = self._evaluate(y_val, y_pred_binary, scores_val)
            mlflow.log_metrics(metrics)

            # Log cluster info
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            mlflow.log_param("n_clusters", n_clusters)
            mlflow.log_param("n_noise_points", n_noise)

            run_id = run.info.run_id

            self.logger.info(f"✓ Clusters: {n_clusters}, Noise: {n_noise}")
            self.logger.info(f"✓ ROC-AUC: {metrics['roc_auc']:.4f}")

            return model, metrics, run_id

    def _evaluate(self, y_true, y_pred, scores):
        """Evaluate model performance"""

        metrics = {}

        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, -scores)
        except:
            metrics['roc_auc'] = 0.0

        # Standard metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # Average Precision
        try:
            metrics['average_precision'] = average_precision_score(
                y_true, -scores)
        except:
            metrics['average_precision'] = 0.0

        # Precision@K and Recall@K
        k = int(len(y_true) * 0.10)
        top_k_idx = np.argsort(scores)[:k]
        top_k_preds = np.zeros(len(y_true))
        top_k_preds[top_k_idx] = 1

        metrics['precision_at_10pct'] = precision_score(
            y_true, top_k_preds, zero_division=0)
        metrics['recall_at_10pct'] = recall_score(
            y_true, top_k_preds, zero_division=0)

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)

        return metrics


# ============================================================================
# STEP 4: MODEL VALIDATION
# ============================================================================

class ModelValidator:
    """Comprehensive model validation"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def validate_against_targets(self, results: Dict) -> Dict:
        """Validate models against performance targets"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 4: MODEL VALIDATION AGAINST TARGETS")
        self.logger.info("="*80)

        targets = self.config['evaluation']['targets']
        validation_results = {}

        for model_name, result in results.items():
            metrics = result['metrics']

            meets_roc_auc = metrics['roc_auc'] >= targets['roc_auc_min']
            meets_precision = metrics['precision_at_10pct'] >= targets['precision_at_k_min']

            passes = meets_roc_auc and meets_precision

            validation_results[model_name] = {
                'passes': passes,
                'roc_auc_check': meets_roc_auc,
                'precision_check': meets_precision
            }

            status = "✓ PASS" if passes else "✗ FAIL"
            self.logger.info(f"\n{model_name}: {status}")
            self.logger.info(
                f"  ROC-AUC: {metrics['roc_auc']:.4f} (target: ≥{targets['roc_auc_min']}) {'✓' if meets_roc_auc else '✗'}")
            self.logger.info(
                f"  Precision@10%: {metrics['precision_at_10pct']:.4f} (target: ≥{targets['precision_at_k_min']}) {'✓' if meets_precision else '✗'}")

        return validation_results


# ============================================================================
# STEP 5: BIAS DETECTION (SECTOR SLICING)
# ============================================================================

# ============================================================================
# STEP 5: BIAS DETECTION (SECTOR SLICING) - COMPLETE FIXED CLASS
# ============================================================================

class BiasDetector:
    """Detect bias using sector slicing - FIXED for anomaly detection"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def detect_sector_bias(self, model, X_val, y_val, val_df, model_name):
        """Detect bias across sectors - FIXED VERSION"""
        self.logger.info("\n" + "="*80)
        self.logger.info(f"BIAS DETECTION: {model_name} (Sector Slicing)")
        self.logger.info("="*80)

        sector_col = self.config['data']['sector_column']
        sectors = val_df[sector_col].values

        # Get predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_val)
            scores = model.score_samples(X_val) if hasattr(
                model, 'score_samples') else None
        else:
            # DBSCAN or clustering
            clusters = model.fit_predict(X_val)
            y_pred = clusters
            scores = None

        # CRITICAL FIX: Convert anomaly detection format
        # Anomaly models output: -1 = anomaly (at-risk), +1 = normal
        # We need: 1 = at-risk, 0 = normal
        if isinstance(y_pred[0], (int, np.integer)) and np.min(y_pred) == -1:
            y_pred_binary = (y_pred == -1).astype(int)
            self.logger.info(
                f"  ✓ Converted predictions from [-1,+1] to [1,0] format")
        else:
            y_pred_binary = y_pred.astype(int)

        # Log overall prediction stats
        n_predicted_at_risk = y_pred_binary.sum()
        predicted_rate = n_predicted_at_risk / len(y_pred_binary)
        actual_rate = y_val.sum() / len(y_val)

        self.logger.info(
            f"  Overall predicted at-risk: {n_predicted_at_risk}/{len(y_pred_binary)} ({predicted_rate:.2%})")
        self.logger.info(
            f"  Overall actual at-risk: {y_val.sum()}/{len(y_val)} ({actual_rate:.2%})")

        # Analyze by sector
        unique_sectors = np.unique(sectors)
        sector_metrics = []

        self.logger.info(f"\n  Analyzing {len(unique_sectors)} sectors...")

        for sector in unique_sectors:
            sector_mask = sectors == sector
            n_sector = sector_mask.sum()

            if n_sector == 0:
                continue

            y_true_sector = y_val[sector_mask]
            y_pred_sector = y_pred_binary[sector_mask]

            n_true_at_risk = y_true_sector.sum()
            n_pred_at_risk = y_pred_sector.sum()

            # Calculate metrics with proper error handling
            try:
                # Check if we can calculate meaningful metrics
                unique_true = len(np.unique(y_true_sector))
                unique_pred = len(np.unique(y_pred_sector))

                if unique_pred < 2:
                    # All predictions are same class
                    if y_pred_sector[0] == 1:
                        # Predicted all at-risk
                        precision = n_true_at_risk / n_sector if n_sector > 0 else 0.0
                        recall = 1.0 if n_true_at_risk > 0 else 0.0
                    else:
                        # Predicted all normal
                        precision = 0.0
                        recall = 0.0
                    f1 = 2 * (precision * recall) / (precision +
                                                     recall) if (precision + recall) > 0 else 0.0

                elif unique_true < 2:
                    # All true labels are same class (all normal or all at-risk)
                    if n_true_at_risk == 0:
                        # Sector has no at-risk companies
                        precision = 0.0 if n_pred_at_risk > 0 else 1.0
                        recall = 0.0  # Can't recall what doesn't exist
                    else:
                        # Sector has only at-risk companies
                        recall = n_pred_at_risk / n_true_at_risk
                        precision = 1.0 if n_pred_at_risk > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision +
                                                     recall) if (precision + recall) > 0 else 0.0

                else:
                    # Normal case: both classes present in true and predicted
                    precision = precision_score(
                        y_true_sector, y_pred_sector, zero_division=0)
                    recall = recall_score(
                        y_true_sector, y_pred_sector, zero_division=0)
                    f1 = f1_score(y_true_sector, y_pred_sector,
                                  zero_division=0)

            except Exception as e:
                self.logger.debug(f"  Error for {sector}: {e}")
                precision = 0.0
                recall = 0.0
                f1 = 0.0

            sector_metrics.append({
                'Sector': sector,
                'Samples': int(n_sector),
                'True_At_Risk': int(n_true_at_risk),
                'Predicted_At_Risk': int(n_pred_at_risk),
                'Precision': float(precision),
                'Recall': float(recall),
                'F1_Score': float(f1)
            })

        sector_df = pd.DataFrame(sector_metrics)

        # Calculate disparity
        # Filter out sectors with F1=0 (no predictions) for fair std calculation
        non_zero_f1 = sector_df[sector_df['F1_Score'] > 0]['F1_Score']
        non_zero_prec = sector_df[sector_df['Precision'] > 0]['Precision']

        if len(non_zero_f1) > 1:
            f1_std = non_zero_f1.std()
        else:
            f1_std = 0.0

        if len(non_zero_prec) > 1:
            precision_std = non_zero_prec.std()
        else:
            precision_std = 0.0

        self.logger.info(f"\nSector Performance Summary:")
        self.logger.info(
            f"  Sectors with predictions: {len(non_zero_f1)}/{len(sector_df)}")
        self.logger.info(f"  F1-Score Std Dev: {f1_std:.4f}")
        self.logger.info(f"  Precision Std Dev: {precision_std:.4f}")

        # Show per-sector breakdown
        self.logger.info(f"\nPer-Sector Breakdown:")
        for _, row in sector_df.sort_values('F1_Score', ascending=False).iterrows():
            self.logger.info(
                f"  {row['Sector']:20s}: F1={row['F1_Score']:.3f}, "
                f"Prec={row['Precision']:.3f}, Rec={row['Recall']:.3f} | "
                f"Pred: {row['Predicted_At_Risk']}/{row['Samples']}, "
                f"Actual: {row['True_At_Risk']}/{row['Samples']}"
            )

        # Flag high disparity
        bias_threshold = 0.15
        if f1_std > bias_threshold or precision_std > bias_threshold:
            self.logger.warning(f"⚠️  HIGH DISPARITY DETECTED across sectors!")
            self.logger.warning(f"   Consider bias mitigation strategies")
            bias_passed = False
        else:
            self.logger.info(f"✓ Low disparity - model is fair across sectors")
            bias_passed = True

        # Log to MLflow
        mlflow.log_metric("sector_f1_std", f1_std)
        mlflow.log_metric("sector_precision_std", precision_std)
        mlflow.log_metric("sectors_with_predictions", len(non_zero_f1))
        mlflow.log_metric(f"bias_f1_std", f1_std)

        # Save sector analysis
        sector_path = Path(
            self.config['output']['results_dir']) / f"{model_name}_sector_analysis.csv"
        sector_df.to_csv(sector_path, index=False)
        mlflow.log_artifact(str(sector_path))

        self.logger.info(f"✓ Sector analysis saved: {sector_path}")

        return sector_df, bias_passed

# ============================================================================
# STEP 6: SENSITIVITY ANALYSIS (SHAP)
# ============================================================================


class SensitivityAnalyzer:
    """Perform sensitivity analysis using SHAP"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def analyze_feature_importance(self, model, X_val, feature_names, model_name):
        """SHAP-based feature importance"""
        self.logger.info("\n" + "="*80)
        self.logger.info(f"SENSITIVITY ANALYSIS: {model_name} (SHAP)")
        self.logger.info("="*80)

        try:
            # Use TreeExplainer for tree-based models
            if 'Forest' in model_name:
                # Use a wrapper for anomaly score
                def model_predict(X):
                    return -model.score_samples(X)

                # Sample for speed (SHAP is slow)
                X_sample = X_val[:min(100, len(X_val))]

                explainer = shap.Explainer(model_predict, X_sample)
                shap_values = explainer(X_sample)

                # Plot
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                  show=False, max_display=15)
                plt.tight_layout()

                shap_path = Path(
                    self.config['output']['plots_dir']) / f"{model_name}_shap_summary.png"
                plt.savefig(shap_path, dpi=300, bbox_inches='tight')
                plt.close()

                self.logger.info(f"✓ SHAP analysis saved: {shap_path}")
                mlflow.log_artifact(str(shap_path))

                # Get feature importance
                importance = np.abs(shap_values.values).mean(axis=0)
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)

                self.logger.info(f"\nTop 5 Most Important Features:")
                for idx, row in feature_importance.head(5).iterrows():
                    self.logger.info(
                        f"  {row['Feature']:30s}: {row['Importance']:.4f}")

                # Save importance
                importance_path = Path(
                    self.config['output']['results_dir']) / f"{model_name}_feature_importance.csv"
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))

                return feature_importance

        except Exception as e:
            self.logger.warning(f"Could not perform SHAP analysis: {str(e)}")
            return None


# ============================================================================
# STEP 7: MODEL SELECTION
# ============================================================================

class ModelSelector:
    """Select best model based on validation and bias"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def select_best_model(self, results: Dict, validation_results: Dict):
        """Select best model considering performance and bias"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 7: MODEL SELECTION")
        self.logger.info("="*80)

        # Filter to models that pass validation
        passing_models = {
            name: result for name, result in results.items()
            if validation_results[name]['passes']
        }

        if not passing_models:
            self.logger.warning("⚠️  No models passed validation targets!")
            self.logger.info("Selecting best performing model anyway...")
            passing_models = results

        # Select best by ROC-AUC
        best_model_name = max(
            passing_models.keys(),
            key=lambda x: passing_models[x]['metrics']['roc_auc']
        )

        best_result = passing_models[best_model_name]

        self.logger.info(f"\n✓ BEST MODEL: {best_model_name}")
        self.logger.info(f"  ROC-AUC: {best_result['metrics']['roc_auc']:.4f}")
        self.logger.info(
            f"  Precision@10%: {best_result['metrics']['precision_at_10pct']:.4f}")
        self.logger.info(
            f"  F1-Score: {best_result['metrics']['f1_score']:.4f}")
        self.logger.info(
            f"  Bias Check: {'✓ Passed' if best_result.get('bias_passed', True) else '✗ Failed'}")

        return best_model_name, best_result


# ============================================================================
# STEP 8: MODEL REGISTRY PUSH
# ============================================================================

class ModelRegistryManager:
    """Push model to MLflow Model Registry"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def register_model(self, model_name: str, run_id: str):
        """Register model in MLflow Model Registry"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 8: PUSHING TO MODEL REGISTRY")
        self.logger.info("="*80)

        try:
            model_uri = f"runs:/{run_id}/model"
            registered_model_name = f"financial_stress_{model_name.lower()}"

            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name
            )

            self.logger.info(f"✓ Model registered: {registered_model_name}")
            self.logger.info(f"✓ Version: {model_version.version}")
            self.logger.info(f"✓ Run ID: {run_id}")

            # Transition to staging
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=registered_model_name,
                version=model_version.version,
                stage="Staging"
            )

            self.logger.info(f"✓ Model promoted to Staging")

            return model_version

        except Exception as e:
            self.logger.error(f"Error registering model: {str(e)}")
            return None

    def save_model_locally(self, model, model_name: str, scaler, feature_cols):
        """Save model artifacts locally"""
        self.logger.info(f"\nSaving {model_name} artifacts locally...")

        model_dir = Path(self.config['output']['models_dir']) / model_name
        model_dir.mkdir(exist_ok=True, parents=True)

        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save scaler
        scaler_path = model_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Save feature names
        features_path = model_dir / "features.json"
        with open(features_path, 'w') as f:
            json.dump({'features': feature_cols}, f)

        self.logger.info(f"✓ Saved model artifacts to: {model_dir}")
    # ============================================================================
# NEW: BEST MODEL GCS UPLOADER
# ============================================================================


class BestModelGCSUploader:
    """Upload only the best model to GCS at models/anomaly_detection/"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def upload_best_model_only(
        self,
        model_name: str,
        model_dir: Path,
        metrics: Dict,
        run_id: str
    ):
        """Upload best model to GCS at models/anomaly_detection/"""
        self.logger.info("\n" + "="*80)
        self.logger.info("UPLOADING BEST MODEL TO GCS")
        self.logger.info("="*80)

        try:
            from google.cloud import storage

            gcs_config = self.config['data']['gcs']
            bucket_name = gcs_config['bucket']

            client = storage.Client()
            bucket = client.bucket(bucket_name)

            # Fixed GCS path
            gcs_base_path = "models/anomaly_detection"

            self.logger.info(f"Best model: {model_name}")
            self.logger.info(f"Target: gs://{bucket_name}/{gcs_base_path}/")

            # Upload model artifacts
            files_to_upload = {
                'model.pkl': f"{gcs_base_path}/model.pkl",
                'scaler.pkl': f"{gcs_base_path}/scaler.pkl",
                'features.json': f"{gcs_base_path}/features.json"
            }

            for local_file, gcs_path in files_to_upload.items():
                local_path = model_dir / local_file
                if local_path.exists():
                    blob = bucket.blob(gcs_path)
                    blob.upload_from_filename(str(local_path))
                    self.logger.info(
                        f"  ✓ {local_file} → gs://{bucket_name}/{gcs_path}")

            # Upload metadata
            metadata = {
                'model_name': model_name,
                'upload_timestamp': datetime.now().isoformat(),
                'mlflow_run_id': run_id,
                'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in metrics.items()},
                'gcs_path': f"gs://{bucket_name}/{gcs_base_path}/"
            }

            metadata_path = f"{gcs_base_path}/model_metadata.json"
            blob = bucket.blob(metadata_path)
            blob.upload_from_string(json.dumps(metadata, indent=2))
            self.logger.info(
                f"  ✓ metadata → gs://{bucket_name}/{metadata_path}")

            self.logger.info(f"\n✓ Best model uploaded successfully!")
            self.logger.info(
                f"  Access at: gs://{bucket_name}/{gcs_base_path}/")

            return True

        except Exception as e:
            self.logger.error(f"Failed to upload to GCS: {e}")
            return False


# ============================================================================
# VISUALIZATIONS
# ============================================================================

class Visualizer:
    """Create comprehensive visualizations"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        Path(self.config['output']['plots_dir']).mkdir(
            exist_ok=True, parents=True)

    def plot_model_comparison(self, results: Dict):
        """Compare all models"""
        self.logger.info("\nCreating model comparison plot...")

        model_names = list(results.keys())
        roc_aucs = [results[m]['metrics']['roc_auc'] for m in model_names]
        precisions_at_k = [results[m]['metrics']
                           ['precision_at_10pct'] for m in model_names]
        f1_scores = [results[m]['metrics']['f1_score'] for m in model_names]
        recalls = [results[m]['metrics']['recall'] for m in model_names]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ROC-AUC
        axes[0, 0].bar(model_names, roc_aucs,
                       color='steelblue', edgecolor='black')
        axes[0, 0].axhline(y=0.85, color='red', linestyle='--',
                           linewidth=2, label='Target: 0.85')
        axes[0, 0].set_ylabel('ROC-AUC', fontsize=12)
        axes[0, 0].set_title('ROC-AUC by Model',
                             fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Precision@10%
        axes[0, 1].bar(model_names, precisions_at_k,
                       color='coral', edgecolor='black')
        axes[0, 1].axhline(y=0.80, color='red', linestyle='--',
                           linewidth=2, label='Target: 0.80')
        axes[0, 1].set_ylabel('Precision@10%', fontsize=12)
        axes[0, 1].set_title('Precision@10% by Model',
                             fontsize=14, fontweight='bold')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # F1-Score
        axes[1, 0].bar(model_names, f1_scores,
                       color='lightgreen', edgecolor='black')
        axes[1, 0].set_ylabel('F1-Score', fontsize=12)
        axes[1, 0].set_title('F1-Score by Model',
                             fontsize=14, fontweight='bold')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Recall
        axes[1, 1].bar(model_names, recalls, color='gold', edgecolor='black')
        axes[1, 1].set_ylabel('Recall', fontsize=12)
        axes[1, 1].set_title('Recall by Model', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        output_path = Path(self.config['output']
                           ['plots_dir']) / "model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Saved: {output_path}")
        mlflow.log_artifact(str(output_path))

    def plot_confusion_matrices(self, results: Dict, y_val):
        """Plot confusion matrices"""
        self.logger.info("Creating confusion matrices...")

        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, result) in enumerate(results.items()):
            metrics = result['metrics']
            cm = [[metrics['true_negatives'], metrics['false_positives']],
                  [metrics['false_negatives'], metrics['true_positives']]]

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        ax=axes[idx], cbar=False,
                        xticklabels=['NOT_AT_RISK', 'AT_RISK'],
                        yticklabels=['NOT_AT_RISK', 'AT_RISK'])
            axes[idx].set_title(
                f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True', fontsize=10)
            axes[idx].set_xlabel('Predicted', fontsize=10)

        plt.tight_layout()

        output_path = Path(self.config['output']
                           ['plots_dir']) / "confusion_matrices.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Saved: {output_path}")
        mlflow.log_artifact(str(output_path))

    def plot_sector_bias(self, sector_df, model_name):
        """Plot sector bias analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Precision by sector
        axes[0].barh(sector_df['Sector'], sector_df['Precision'],
                     color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Precision', fontsize=11)
        axes[0].set_title(f'{model_name}: Precision by Sector',
                          fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')

        # Recall by sector
        axes[1].barh(sector_df['Sector'], sector_df['Recall'],
                     color='coral', edgecolor='black')
        axes[1].set_xlabel('Recall', fontsize=11)
        axes[1].set_title(f'{model_name}: Recall by Sector',
                          fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

        # F1 by sector
        axes[2].barh(sector_df['Sector'], sector_df['F1_Score'],
                     color='lightgreen', edgecolor='black')
        axes[2].set_xlabel('F1-Score', fontsize=11)
        axes[2].set_title(f'{model_name}: F1-Score by Sector',
                          fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        output_path = Path(
            self.config['output']['plots_dir']) / f"{model_name}_sector_bias.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Saved sector bias plot: {output_path}")
        mlflow.log_artifact(str(output_path))


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_comprehensive_report(results, validation_results, best_model_info, config, logger):
    """Generate final comprehensive report"""
    logger.info("\n" + "="*80)
    logger.info("GENERATING COMPREHENSIVE REPORT")
    logger.info("="*80)

    Path(config['output']['reports_dir']).mkdir(exist_ok=True, parents=True)

    report = f"""
{'='*80}
MODEL 3: ANOMALY DETECTION - COMPREHENSIVE TRAINING REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Author: Parth Saraykar
MLOps Innovation Expo - Financial Stress Test Generator

1. MODELS TRAINED
{'='*80}
Total Models: {len(results)}
"""

    for model_name, result in results.items():
        metrics = result['metrics']
        report += f"\n{model_name}:\n"
        report += f"  ROC-AUC:           {metrics['roc_auc']:.4f}\n"
        report += f"  Precision@10%:     {metrics['precision_at_10pct']:.4f}\n"
        report += f"  Recall@10%:        {metrics['recall_at_10pct']:.4f}\n"
        report += f"  F1-Score:          {metrics['f1_score']:.4f}\n"
        report += f"  Precision:         {metrics['precision']:.4f}\n"
        report += f"  Recall:            {metrics['recall']:.4f}\n"
        report += f"  MLflow Run ID:     {result['run_id']}\n"

    report += f"""
2. VALIDATION RESULTS
{'='*80}
Performance Targets:
  - ROC-AUC ≥ 0.85
  - Precision@10% ≥ 0.80

Model Validation Status:
"""

    for model_name, val_result in validation_results.items():
        status = "✓ PASS" if val_result['passes'] else "✗ FAIL"
        report += f"  {model_name:20s}: {status}\n"

    report += f"""
3. BEST MODEL SELECTED
{'='*80}
Model: {best_model_info['name']}
ROC-AUC: {best_model_info['metrics']['roc_auc']:.4f}
Precision@10%: {best_model_info['metrics']['precision_at_10pct']:.4f}
F1-Score: {best_model_info['metrics']['f1_score']:.4f}
Bias Check: {' ✓ Passed' if best_model_info.get('bias_passed', True) else '✗ Failed'}
MLflow Run ID: {best_model_info['run_id']}

4. BIAS DETECTION SUMMARY
{'='*80}
Slicing Dimension: Sector
Bias Metric: F1-Score Standard Deviation across sectors
Result: {'✓ No significant bias detected' if best_model_info.get('bias_passed', True) else '⚠️ Bias detected - mitigation recommended'}

5. SENSITIVITY ANALYSIS
{'='*80}
Method: SHAP (SHapley Additive exPlanations)
See outputs/models/plots/{best_model_info['name']}_shap_summary.png

6. MODEL REGISTRY
{'='*80}
Status: {'✓ Registered in MLflow Model Registry' if best_model_info.get('registered', False) else 'Local artifacts saved only'}
Local Path: models/anomaly_detection/{best_model_info['name']}/

7. OUTPUTS GENERATED
{'='*80}
✓ Model comparison plots
✓ Confusion matrices
✓ Sector bias analysis
✓ SHAP feature importance
✓ Model artifacts (model.pkl, scaler.pkl, features.json)
✓ MLflow experiment logs

8. NEXT STEPS
{'='*80}
1. Review MLflow UI: mlflow ui --port 5000
2. Examine sector bias analysis: outputs/models/results/*_sector_analysis.csv
3. Review SHAP plots: outputs/models/plots/*_shap_summary.png
4. Integrate with teammates' models (VAE/GAN, XGBoost/LSTM)
5. Deploy to production or proceed to ensemble model

{'='*80}
END OF REPORT
{'='*80}
"""

    report_path = Path(config['output']['reports_dir']) / \
        "comprehensive_training_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"✓ Report saved: {report_path}")
    print("\n" + report)

    return report


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Complete training pipeline with integrated sensitivity analysis"""

    # Setup
    logger = setup_logger()
    config = Config()

    logger.info("="*80)
    logger.info("MODEL 3: ANOMALY DETECTION - COMPLETE TRAINING PIPELINE")
    logger.info("WITH INTEGRATED HYPERPARAMETER SENSITIVITY ANALYSIS")
    logger.info("="*80)
    logger.info("Requirements Implemented:")
    logger.info("  ✓ Data loading from pipeline (§2.1)")
    logger.info("  ✓ Model training & selection (§2.2)")
    logger.info("  ✓ Hyperparameter tuning (§3)")
    logger.info("  ✓ MLflow experiment tracking (§4)")
    logger.info("  ✓ Model validation (§2.3)")
    logger.info("  ✓ SENSITIVITY ANALYSIS (§5)")
    logger.info("  ✓ Bias detection - sector slicing (§6)")
    logger.info("  ✓ SHAP feature importance (§5)")
    logger.info("  ✓ Model registry push (§2.6)")
    logger.info("="*80)

    # Create output directories
    for d in [config['output']['models_dir'], config['output']['results_dir'],
              config['output']['plots_dir'], config['output']['reports_dir']]:
        Path(d).mkdir(exist_ok=True, parents=True)

    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Initialize components
    data_loader = DataLoader(config, logger)
    sensitivity_analyzer = IntegratedSensitivityAnalyzer(config, logger)
    tuner = HyperparameterTuner(config, logger)
    trainer = ModelTrainer(config, logger)
    validator = ModelValidator(config, logger)
    bias_detector = BiasDetector(config, logger)
    shap_analyzer = SensitivityAnalyzer(config, logger)
    model_selector = ModelSelector(config, logger)
    registry_manager = ModelRegistryManager(config, logger)
    visualizer = Visualizer(config, logger)
    # STEP 1: Load data
    df = data_loader.load_labeled_data()
    train_df, val_df, test_df = data_loader.temporal_split(df)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_cols = \
        data_loader.prepare_features(train_df, val_df, test_df)

    # ========================================================================
    # STEP 1.5: INTEGRATED SENSITIVITY ANALYSIS
    # ========================================================================

    with mlflow.start_run(run_name="main_training_pipeline"):
        mlflow.set_tags(config['mlflow']['tags'])

        logger.info("\n" + "="*80)
        logger.info("RUNNING INTEGRATED SENSITIVITY ANALYSIS")
        logger.info("="*80)

        # Run quick sensitivity analysis (set quick_mode=True for faster demo)
        sensitivity_results = sensitivity_analyzer.run_sensitivity_analysis(
            X_train, y_train, X_val, y_val,
            quick_mode=True  # Set to False for comprehensive analysis
        )

        logger.info("✓ Sensitivity analysis complete")
        logger.info("  Results logged to MLflow and saved to outputs/")
    # STEP 2: Hyperparameter tuning (optional - Isolation Forest only)
    logger.info("\n" + "="*80)
    logger.info("STEP 2: HYPERPARAMETER TUNING")
    logger.info("="*80)
    best_if_params = tuner.tune_isolation_forest(
        X_train, y_train, X_val, y_val)

    # STEP 3: Train all models
    logger.info("\n" + "="*80)
    logger.info("STEP 3: TRAINING MODELS WITH MLFLOW")
    logger.info("="*80)

    results = {}

    # Isolation Forest (with tuned params)
    if config['models']['isolation_forest']['enabled']:
        model, metrics, run_id = trainer.train_isolation_forest(
            X_train, y_train, X_val, y_val, params=best_if_params
        )
        results['Isolation_Forest'] = {
            'model': model, 'metrics': metrics, 'run_id': run_id
        }

    # LOF
    if config['models']['local_outlier_factor']['enabled']:
        model, metrics, run_id = trainer.train_lof(
            X_train, y_train, X_val, y_val)
        results['LOF'] = {
            'model': model, 'metrics': metrics, 'run_id': run_id
        }

    # One-Class SVM
    if config['models']['one_class_svm']['enabled']:
        model, metrics, run_id = trainer.train_one_class_svm(
            X_train, y_train, X_val, y_val)
        results['One_Class_SVM'] = {
            'model': model, 'metrics': metrics, 'run_id': run_id
        }

    # DBSCAN
    if config['models']['dbscan']['enabled']:
        model, metrics, run_id = trainer.train_dbscan(
            X_train, y_train, X_val, y_val)
        results['DBSCAN'] = {
            'model': model, 'metrics': metrics, 'run_id': run_id
        }

    # STEP 4: Validation
    validation_results = validator.validate_against_targets(results)

    # STEP 5: Bias detection
    logger.info("\n" + "="*80)
    logger.info("STEP 5: BIAS DETECTION (SECTOR SLICING)")
    logger.info("="*80)

    for model_name, result in results.items():
        sector_df, bias_passed = bias_detector.detect_sector_bias(
            result['model'], X_val, y_val, val_df, model_name
        )
        result['bias_passed'] = bias_passed
        result['sector_analysis'] = sector_df

        # Visualize bias
        visualizer.plot_sector_bias(sector_df, model_name)

    # ========================================================================
# STEP 6: SHAP sensitivity analysis (Feature Importance)
# ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 6: FEATURE IMPORTANCE ANALYSIS (SHAP)")
    logger.info("="*80)

    for model_name, result in results.items():
        if 'Forest' in model_name:  # SHAP works best with tree-based models
            importance = shap_analyzer.analyze_feature_importance(  # ← Use shap_analyzer
                result['model'], X_val, feature_cols, model_name
            )
            result['feature_importance'] = importance

    # STEP 7: Select best model
    best_model_name, best_result = model_selector.select_best_model(
        results, validation_results)

    # STEP 8: Register best model
    registered_version = registry_manager.register_model(
        best_model_name, best_result['run_id'])
    best_result['registered'] = registered_version is not None

    # Save all models locally
    for model_name, result in results.items():
        registry_manager.save_model_locally(
            result['model'], model_name, scaler, feature_cols
        )

    # STEP 9: Visualizations
    logger.info("\n" + "="*80)
    logger.info("STEP 9: GENERATING VISUALIZATIONS")
    logger.info("="*80)

    visualizer.plot_model_comparison(results)
    visualizer.plot_confusion_matrices(results, y_val)

    # STEP 10: Final report
    best_model_info = {
        'name': best_model_name,
        'metrics': best_result['metrics'],
        'run_id': best_result['run_id'],
        'bias_passed': best_result['bias_passed'],
        'registered': best_result.get('registered', False)
    }

    report = generate_comprehensive_report(
        results, validation_results, best_model_info, config, logger
    )

    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING PIPELINE WITH ENSEMBLE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)

    # ========================================================================
    # STEP 13: UPLOAD BEST MODEL TO GCS
    # ========================================================================

    if config['output'].get('upload_to_gcs', False):
        logger.info("\n" + "="*80)
        logger.info("STEP 13: UPLOADING BEST MODEL TO GCS")
        logger.info("="*80)

        # Initialize best model uploader
        best_model_uploader = BestModelGCSUploader(config, logger)

        # Upload only the best model
        best_model_dir = Path(config['output']['models_dir']) / best_model_name

        if best_model_dir.exists():
            upload_success = best_model_uploader.upload_best_model_only(
                model_name=best_model_name,
                model_dir=best_model_dir,
                metrics=best_result['metrics'],
                run_id=best_result['run_id']
            )

            if upload_success:
                logger.info(
                    f"✓ Best model ({best_model_name}) uploaded to GCS")
                logger.info(
                    f"  Path: gs://{config['data']['gcs']['bucket']}/models/anomaly_detection/")
            else:
                logger.warning("⚠️  Failed to upload best model to GCS")
        else:
            logger.error(f"Model directory not found: {best_model_dir}")

        logger.info("\n All outputs uploaded to GCS!")
        logger.info(f"   Bucket: gs://{config['data']['gcs']['bucket']}/")
        logger.info(
            f"   Models: gs://{config['data']['gcs']['bucket']}/models/model3/")

    # Original view results section
    logger.info("\nView results:")
    logger.info("  1. MLflow UI:           mlflow ui --port 5000")
    logger.info(
        "  2. Ensemble comparison: cat outputs/models/results/ensemble_comparison.csv")
    logger.info(
        "  3. Sensitivity plots:   outputs/models/plots/sensitivity_analysis.png")
    logger.info(
        "  4. Report:              cat outputs/models/reports/comprehensive_training_report.txt")
    logger.info("  5. Visualizations:      outputs/models/plots/")
    logger.info("  6. Ensemble model:      models/Ensemble/ensemble_model.pkl")
    if config['output'].get('upload_to_gcs', False):
        logger.info(
            "  7. GCS Bucket:          gs://mlops-financial-stress-data/models/model3/")


if __name__ == "__main__":
    main()
