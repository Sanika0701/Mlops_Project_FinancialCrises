"""
src/monitoring/model_monitor.py

Comprehensive Model Monitoring System for Financial Crisis Prediction Models

Features:
1. Performance decay detection
2. Data drift detection
3. Feature distribution monitoring
4. Automatic retraining triggers
5. Alerting and notifications
6. Metrics logging for dashboard

Usage:
    # Run monitoring on new data
    python src/monitoring/model_monitor.py --new-data data/new_predictions.csv
    
    # Check all targets
    python src/monitoring/model_monitor.py --check-all
"""

import json
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# Setup paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


class ModelPerformanceMonitor:
    """Monitor model performance and detect decay"""
    
    def __init__(self, model_path: str, baseline_metrics: Dict):
        self.model_path = model_path
        self.baseline_metrics = baseline_metrics
        self.performance_threshold = 0.15  # 15% degradation triggers alert
        
    def evaluate_performance(self, X, y_true) -> Dict:
        """Evaluate current model performance"""
        model_data = joblib.load(self.model_path)
        model = model_data['model']
        
        y_pred = model.predict(X)
        
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def detect_performance_decay(self, current_metrics: Dict) -> Tuple[bool, Dict]:
        """
        Detect if model performance has degraded significantly
        
        Returns:
            (is_degraded, decay_info)
        """
        baseline_r2 = self.baseline_metrics['test']['r2']
        current_r2 = current_metrics['r2']
        
        # Calculate relative degradation
        degradation = (baseline_r2 - current_r2) / baseline_r2
        
        is_degraded = degradation > self.performance_threshold
        
        decay_info = {
            'baseline_r2': baseline_r2,
            'current_r2': current_r2,
            'degradation_pct': degradation * 100,
            'threshold_pct': self.performance_threshold * 100,
            'is_degraded': is_degraded,
            'severity': self._get_severity(degradation)
        }
        
        return is_degraded, decay_info
    
    def _get_severity(self, degradation: float) -> str:
        """Determine severity level of degradation"""
        if degradation < 0.10:
            return 'LOW'
        elif degradation < 0.20:
            return 'MEDIUM'
        elif degradation < 0.30:
            return 'HIGH'
        else:
            return 'CRITICAL'


class DataDriftDetector:
    """Detect data drift using statistical tests"""
    
    def __init__(self, reference_data: pd.DataFrame, alpha: float = 0.05):
        """
        Args:
            reference_data: Training data for reference
            alpha: Significance level for statistical tests
        """
        self.reference_data = reference_data
        self.alpha = alpha
        self.reference_stats = self._compute_statistics(reference_data)
    
    def _compute_statistics(self, data: pd.DataFrame) -> Dict:
        """Compute statistical properties of data"""
        stats_dict = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'median': data[col].median(),
                'q25': data[col].quantile(0.25),
                'q75': data[col].quantile(0.75),
                'min': data[col].min(),
                'max': data[col].max()
            }
        
        return stats_dict
    
    def detect_drift(self, new_data: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Detect data drift using multiple methods:
        1. Kolmogorov-Smirnov test (distribution change)
        2. Population Stability Index (PSI)
        3. Statistical distance measures
        
        Returns:
            (has_drift, drift_report)
        """
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'features_with_drift': [],
            'drift_scores': {},
            'ks_test_results': {},
            'psi_scores': {}
        }
        
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in new_data.columns:
                continue
            
            # 1. Kolmogorov-Smirnov Test
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[col].dropna(),
                new_data[col].dropna()
            )
            
            drift_results['ks_test_results'][col] = {
                'statistic': float(ks_stat),
                'p_value': float(p_value),
                'has_drift': p_value < self.alpha
            }
            
            # 2. Population Stability Index (PSI)
            psi = self._calculate_psi(
                self.reference_data[col].dropna(),
                new_data[col].dropna()
            )
            
            drift_results['psi_scores'][col] = float(psi)
            
            # Determine if feature has significant drift
            if p_value < self.alpha or psi > 0.2:  # PSI > 0.2 indicates significant drift
                drift_results['features_with_drift'].append(col)
                drift_results['drift_scores'][col] = {
                    'ks_statistic': float(ks_stat),
                    'psi': float(psi),
                    'severity': self._get_drift_severity(psi)
                }
        
        has_drift = len(drift_results['features_with_drift']) > 0
        drift_results['total_features_checked'] = len(numeric_cols)
        drift_results['num_features_with_drift'] = len(drift_results['features_with_drift'])
        drift_results['drift_percentage'] = (
            len(drift_results['features_with_drift']) / len(numeric_cols) * 100
        )
        
        return has_drift, drift_results
    
    def _calculate_psi(self, reference: pd.Series, new: pd.Series, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        PSI < 0.1: No significant change
        PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change (trigger alert)
        """
        # Create bins based on reference data
        try:
            breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
            breakpoints = np.unique(breakpoints)  # Remove duplicates
            
            if len(breakpoints) < 2:
                return 0.0
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=breakpoints)
            new_counts, _ = np.histogram(new, bins=breakpoints)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            ref_percents = (ref_counts + epsilon) / (len(reference) + epsilon * len(ref_counts))
            new_percents = (new_counts + epsilon) / (len(new) + epsilon * len(new_counts))
            
            # Calculate PSI
            psi = np.sum((new_percents - ref_percents) * np.log(new_percents / ref_percents))
            
            return psi
        except:
            return 0.0
    
    def _get_drift_severity(self, psi: float) -> str:
        """Determine severity of drift based on PSI"""
        if psi < 0.1:
            return 'LOW'
        elif psi < 0.2:
            return 'MEDIUM'
        elif psi < 0.3:
            return 'HIGH'
        else:
            return 'CRITICAL'


class MonitoringOrchestrator:
    """Main orchestrator for all monitoring tasks"""
    
    def __init__(
        self,
        models_dir: str = "models/best_models",
        reference_data_path: str = "data/splits/train_data.csv",
        monitoring_logs_dir: str = "logs/monitoring"
    ):
        self.models_dir = Path(models_dir)
        self.reference_data_path = Path(reference_data_path)
        self.monitoring_logs_dir = Path(monitoring_logs_dir)
        self.monitoring_logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.targets = ["revenue", "eps", "debt_equity", "profit_margin", "stock_return"]
        
    def run_comprehensive_monitoring(self, new_data_path: str = None) -> Dict:
        """
        Run complete monitoring pipeline:
        1. Performance decay check
        2. Data drift detection
        3. Generate alerts
        4. Log results
        
        Args:
            new_data_path: Path to new data for monitoring (if None, uses test set)
        
        Returns:
            Monitoring report with recommendations
        """
        print(f"\n{'='*80}")
        print(f"🔍 COMPREHENSIVE MODEL MONITORING")
        print(f"{'='*80}\n")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Monitoring {len(self.targets)} models...")
        
        # Load reference data
        reference_data = pd.read_csv(self.reference_data_path)
        
        # Load new data (or use test set as proxy)
        if new_data_path:
            new_data = pd.read_csv(new_data_path)
            print(f"New data: {new_data_path} ({len(new_data)} rows)")
        else:
            test_data_path = self.reference_data_path.parent / "test_data.csv"
            new_data = pd.read_csv(test_data_path)
            print(f"Using test set as proxy: {test_data_path}")
        
        monitoring_report = {
            'timestamp': datetime.now().isoformat(),
            'targets_monitored': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Monitor each target
        for target in self.targets:
            print(f"\n{'─'*80}")
            print(f"📊 Monitoring: {target.upper()}")
            print(f"{'─'*80}")
            
            target_report = self._monitor_single_target(
                target, reference_data, new_data
            )
            
            monitoring_report['targets_monitored'][target] = target_report
            
            # Generate alerts
            if target_report['performance_decay']['is_degraded']:
                alert = {
                    'target': target,
                    'type': 'PERFORMANCE_DECAY',
                    'severity': target_report['performance_decay']['severity'],
                    'message': f"Performance degraded by {target_report['performance_decay']['degradation_pct']:.1f}%",
                    'recommendation': 'RETRAIN_MODEL'
                }
                monitoring_report['alerts'].append(alert)
                print(f"   ⚠️  ALERT: {alert['message']}")
            
            if target_report['data_drift']['has_drift']:
                drift_pct = target_report['data_drift']['drift_percentage']
                alert = {
                    'target': target,
                    'type': 'DATA_DRIFT',
                    'severity': 'HIGH' if drift_pct > 30 else 'MEDIUM',
                    'message': f"Data drift detected in {target_report['data_drift']['num_features_with_drift']} features ({drift_pct:.1f}%)",
                    'recommendation': 'INVESTIGATE_AND_RETRAIN'
                }
                monitoring_report['alerts'].append(alert)
                print(f"   ⚠️  ALERT: {alert['message']}")
        
        # Generate recommendations
        monitoring_report['recommendations'] = self._generate_recommendations(
            monitoring_report['alerts']
        )
        
        # Save report
        self._save_monitoring_report(monitoring_report)
        
        # Print summary
        self._print_monitoring_summary(monitoring_report)
        
        return monitoring_report
    
    def _monitor_single_target(
        self,
        target: str,
        reference_data: pd.DataFrame,
        new_data: pd.DataFrame
    ) -> Dict:
        """Monitor a single target model"""
        
        model_path = self.models_dir / f"{target}_best.pkl"
        
        if not model_path.exists():
            return {
                'status': 'MODEL_NOT_FOUND',
                'error': f"Model not found: {model_path}"
            }
        
        # Load model and baseline metrics
        model_data = joblib.load(model_path)
        baseline_metrics = model_data['test_metrics']
        feature_names = model_data['feature_names']
        
        # Prepare data
        target_col = f"target_{target}"
        
        # Filter to features used by model
        X_new = new_data[feature_names] if all(f in new_data.columns for f in feature_names) else new_data
        y_new = new_data[target_col] if target_col in new_data.columns else None
        
        target_report = {
            'model_path': str(model_path),
            'baseline_metrics': baseline_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Performance Monitoring (if we have true labels)
        if y_new is not None:
            monitor = ModelPerformanceMonitor(model_path, model_data)
            
            # Drop NaN values
            valid_idx = ~y_new.isna()
            X_valid = X_new[valid_idx]
            y_valid = y_new[valid_idx]
            
            current_metrics = monitor.evaluate_performance(X_valid, y_valid)
            is_degraded, decay_info = monitor.detect_performance_decay(current_metrics)
            
            target_report['current_metrics'] = current_metrics
            target_report['performance_decay'] = decay_info
            
            print(f"   Performance: R² = {current_metrics['r2']:.4f} (baseline: {baseline_metrics['r2']:.4f})")
            if is_degraded:
                print(f"   ⚠️  Degradation: {decay_info['degradation_pct']:.1f}% ({decay_info['severity']})")
        else:
            target_report['performance_decay'] = {'status': 'NO_LABELS', 'is_degraded': False}
            print(f"   ⚠️  No labels available for performance monitoring")
        
        # 2. Data Drift Detection
        # Use only features that were in training
        ref_features = reference_data[feature_names] if all(f in reference_data.columns for f in feature_names) else reference_data
        new_features = X_new[feature_names] if all(f in X_new.columns for f in feature_names) else X_new
        
        drift_detector = DataDriftDetector(ref_features)
        has_drift, drift_report = drift_detector.detect_drift(new_features)
        
        target_report['data_drift'] = drift_report
        
        if has_drift:
            print(f"   ⚠️  Data Drift: {drift_report['num_features_with_drift']} features affected")
            print(f"      Features: {', '.join(drift_report['features_with_drift'][:5])}")
        else:
            print(f"   ✅ No significant data drift detected")
        
        return target_report
    
    def _generate_recommendations(self, alerts: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on alerts"""
        recommendations = []
        
        # Count alert types
        performance_alerts = [a for a in alerts if a['type'] == 'PERFORMANCE_DECAY']
        drift_alerts = [a for a in alerts if a['type'] == 'DATA_DRIFT']
        
        critical_alerts = [a for a in alerts if a['severity'] == 'CRITICAL']
        
        if critical_alerts:
            recommendations.append("🚨 URGENT: Critical alerts detected - immediate retraining recommended")
        
        if len(performance_alerts) >= 3:
            recommendations.append("⚠️  Multiple models showing performance decay - schedule full pipeline retraining")
        elif performance_alerts:
            targets = [a['target'] for a in performance_alerts]
            recommendations.append(f"⚠️  Retrain models: {', '.join(targets)}")
        
        if len(drift_alerts) >= 3:
            recommendations.append("⚠️  Widespread data drift detected - investigate data collection process")
        elif drift_alerts:
            recommendations.append("⚠️  Data drift detected - consider feature engineering review")
        
        if not alerts:
            recommendations.append("✅ All models performing well - no action needed")
        
        return recommendations
    
    def _save_monitoring_report(self, report: Dict):
        """Save monitoring report with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.monitoring_logs_dir / f"monitoring_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save as "latest"
        latest_path = self.monitoring_logs_dir / "monitoring_report_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Monitoring report saved:")
        print(f"   - {report_path}")
        print(f"   - {latest_path}")
    
    def _print_monitoring_summary(self, report: Dict):
        """Print monitoring summary"""
        print(f"\n{'='*80}")
        print(f"📊 MONITORING SUMMARY")
        print(f"{'='*80}\n")
        
        total_alerts = len(report['alerts'])
        
        if total_alerts == 0:
            print("✅ All systems healthy - no alerts")
        else:
            print(f"⚠️  Total Alerts: {total_alerts}")
            print(f"\nAlert Breakdown:")
            
            for alert in report['alerts']:
                icon = "🚨" if alert['severity'] == 'CRITICAL' else "⚠️"
                print(f"   {icon} [{alert['severity']}] {alert['target']}: {alert['message']}")
        
        if report['recommendations']:
            print(f"\n📋 Recommendations:")
            for rec in report['recommendations']:
                print(f"   {rec}")
        
        print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Model Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--new-data",
        type=str,
        help="Path to new data for monitoring (optional, uses test set if not provided)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/best_models",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--reference-data",
        type=str,
        default="data/splits/train_data.csv",
        help="Path to reference training data"
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs/monitoring",
        help="Directory to save monitoring logs"
    )
    
    args = parser.parse_args()
    
    # Run monitoring
    orchestrator = MonitoringOrchestrator(
        models_dir=args.models_dir,
        reference_data_path=args.reference_data,
        monitoring_logs_dir=args.logs_dir
    )
    
    report = orchestrator.run_comprehensive_monitoring(new_data_path=args.new_data)
    
    # Check if retraining should be triggered
    if report['alerts']:
        critical_alerts = [a for a in report['alerts'] if a['severity'] in ['CRITICAL', 'HIGH']]
        if critical_alerts:
            print(f"\n{'='*80}")
            print(f"🚨 RETRAINING TRIGGER")
            print(f"{'='*80}")
            print(f"{len(critical_alerts)} critical/high severity alerts detected")
            print(f"Recommendation: Trigger automated retraining pipeline")
            
            # Exit with code 1 to trigger CI/CD retraining
            sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()