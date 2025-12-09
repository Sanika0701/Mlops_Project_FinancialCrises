"""
src/monitoring/gcp_monitoring_setup.py

Set up Google Cloud Monitoring Dashboard for ML Models

This script:
1. Creates custom metrics in Cloud Monitoring
2. Pushes model performance metrics to GCP
3. Creates a monitoring dashboard
4. Sets up alerting policies

Usage:
    python src/monitoring/gcp_monitoring_setup.py --project-id ninth-iris-422916-f2
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from google.cloud import monitoring_v3
from google.api import metric_pb2 as ga_metric
from google.api import label_pb2 as ga_label
import time


class GCPMonitoringSetup:
    """Setup and manage GCP Cloud Monitoring for ML models"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.project_name = f"projects/{project_id}"
        self.client = monitoring_v3.MetricServiceClient()
        try:
            self.alert_client = monitoring_v3.AlertPolicyServiceClient()
        except:
            self.alert_client = None
            print("⚠️  Alert policy client not available")
        
    def create_custom_metric_descriptors(self):
        """Create custom metric descriptors for ML model monitoring"""
        
        print("📊 Creating custom metric descriptors...")
        
        metrics = [
            {
                "type": "custom.googleapis.com/ml/model/r2_score",
                "display_name": "Model R² Score",
                "description": "R-squared score for model performance",
                "metric_kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
                "value_type": ga_metric.MetricDescriptor.ValueType.DOUBLE,
                "labels": [
                    {"key": "model_name", "description": "Name of the ML model"},
                    {"key": "target", "description": "Target variable"}
                ]
            },
            {
                "type": "custom.googleapis.com/ml/model/rmse",
                "display_name": "Model RMSE",
                "description": "Root Mean Squared Error",
                "metric_kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
                "value_type": ga_metric.MetricDescriptor.ValueType.DOUBLE,
                "labels": [
                    {"key": "model_name", "description": "Name of the ML model"},
                    {"key": "target", "description": "Target variable"}
                ]
            },
            {
                "type": "custom.googleapis.com/ml/model/mae",
                "display_name": "Model MAE",
                "description": "Mean Absolute Error",
                "metric_kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
                "value_type": ga_metric.MetricDescriptor.ValueType.DOUBLE,
                "labels": [
                    {"key": "model_name", "description": "Name of the ML model"},
                    {"key": "target", "description": "Target variable"}
                ]
            },
            {
                "type": "custom.googleapis.com/ml/model/degradation_pct",
                "display_name": "Performance Degradation %",
                "description": "Percentage degradation from baseline",
                "metric_kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
                "value_type": ga_metric.MetricDescriptor.ValueType.DOUBLE,
                "labels": [
                    {"key": "model_name", "description": "Name of the ML model"},
                    {"key": "target", "description": "Target variable"}
                ]
            },
            {
                "type": "custom.googleapis.com/ml/model/drift_percentage",
                "display_name": "Data Drift Percentage",
                "description": "Percentage of features with significant drift",
                "metric_kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
                "value_type": ga_metric.MetricDescriptor.ValueType.DOUBLE,
                "labels": [
                    {"key": "model_name", "description": "Name of the ML model"},
                    {"key": "target", "description": "Target variable"}
                ]
            },
            {
                "type": "custom.googleapis.com/ml/model/alert_count",
                "display_name": "Active Alerts",
                "description": "Number of active alerts for the model",
                "metric_kind": ga_metric.MetricDescriptor.MetricKind.GAUGE,
                "value_type": ga_metric.MetricDescriptor.ValueType.INT64,
                "labels": [
                    {"key": "model_name", "description": "Name of the ML model"},
                    {"key": "target", "description": "Target variable"},
                    {"key": "severity", "description": "Alert severity"}
                ]
            }
        ]
        
        for metric_info in metrics:
            try:
                descriptor = ga_metric.MetricDescriptor()
                descriptor.type = metric_info["type"]
                descriptor.display_name = metric_info["display_name"]
                descriptor.description = metric_info["description"]
                descriptor.metric_kind = metric_info["metric_kind"]
                descriptor.value_type = metric_info["value_type"]
                
                for label in metric_info["labels"]:
                    label_descriptor = ga_label.LabelDescriptor()
                    label_descriptor.key = label["key"]
                    label_descriptor.description = label["description"]
                    descriptor.labels.append(label_descriptor)
                
                self.client.create_metric_descriptor(
                    name=self.project_name,
                    metric_descriptor=descriptor
                )
                print(f"   ✅ Created: {metric_info['display_name']}")
                
            except Exception as e:
                if "already exists" in str(e):
                    print(f"   ℹ️  Already exists: {metric_info['display_name']}")
                else:
                    print(f"   ❌ Error creating {metric_info['display_name']}: {e}")
    
    def push_metrics_from_monitoring_report(self, report_path: str):
        """Push metrics from monitoring report to Cloud Monitoring"""
        
        print(f"\n📤 Pushing metrics from: {report_path}")
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        current_time = time.time()
        
        for target, data in report.get('targets_monitored', {}).items():
            print(f"   Pushing metrics for: {target}")
            
            # Skip if error
            if data.get('status') == 'ERROR':
                continue
            
            # R² Score
            if 'current_metrics' in data:
                self._write_metric(
                    "custom.googleapis.com/ml/model/r2_score",
                    data['current_metrics']['r2'],
                    {"model_name": "best_model", "target": target},
                    current_time
                )
                
                self._write_metric(
                    "custom.googleapis.com/ml/model/rmse",
                    data['current_metrics']['rmse'],
                    {"model_name": "best_model", "target": target},
                    current_time
                )
                
                self._write_metric(
                    "custom.googleapis.com/ml/model/mae",
                    data['current_metrics']['mae'],
                    {"model_name": "best_model", "target": target},
                    current_time
                )
            
            # Degradation
            if 'performance_decay' in data:
                self._write_metric(
                    "custom.googleapis.com/ml/model/degradation_pct",
                    data['performance_decay'].get('degradation_pct', 0),
                    {"model_name": "best_model", "target": target},
                    current_time
                )
            
            # Drift
            if 'data_drift' in data:
                self._write_metric(
                    "custom.googleapis.com/ml/model/drift_percentage",
                    data['data_drift'].get('drift_percentage', 0),
                    {"model_name": "best_model", "target": target},
                    current_time
                )
        
        # Push alert counts
        alerts_by_target = {}
        for alert in report.get('alerts', []):
            target = alert['target']
            severity = alert['severity']
            key = f"{target}_{severity}"
            alerts_by_target[key] = alerts_by_target.get(key, 0) + 1
        
        for key, count in alerts_by_target.items():
            target, severity = key.rsplit('_', 1)
            self._write_metric(
                "custom.googleapis.com/ml/model/alert_count",
                count,
                {"model_name": "best_model", "target": target, "severity": severity},
                current_time,
                value_type='int64'
            )
        
        print("   ✅ Metrics pushed successfully")
    
    def _write_metric(self, metric_type: str, value: float, labels: dict, timestamp: float, value_type='double'):
        """Write a single metric to Cloud Monitoring"""
        
        series = monitoring_v3.TimeSeries()
        series.metric.type = metric_type
        
        for key, val in labels.items():
            series.metric.labels[key] = str(val)
        
        series.resource.type = "global"
        
        # Create interval and point
        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        
        interval = monitoring_v3.TimeInterval(
            {
                "end_time": {"seconds": seconds, "nanos": nanos}
            }
        )
        
        point = monitoring_v3.Point(
            {
                "interval": interval,
                "value": {"double_value": float(value)} if value_type == 'double' else {"int64_value": int(value)}
            }
        )
        
        series.points = [point]
        
        try:
            self.client.create_time_series(
                name=self.project_name,
                time_series=[series]
            )
        except Exception as e:
            print(f"      ⚠️  Error writing metric {metric_type}: {e}")
    
    def create_dashboard(self):
        """Create a monitoring dashboard using gcloud command"""
        
        print("\n📊 Creating Cloud Monitoring Dashboard...")
        print("   Note: Dashboard will be created via GCP Console")
        
        dashboard_config = {
            "displayName": "ML Model Monitoring Dashboard",
            "gridLayout": {
                "widgets": [
                    {
                        "title": "Model R² Score",
                        "xyChart": {
                            "dataSets": [{
                                "timeSeriesQuery": {
                                    "timeSeriesFilter": {
                                        "filter": 'metric.type="custom.googleapis.com/ml/model/r2_score"'
                                    }
                                }
                            }]
                        }
                    }
                ]
            }
        }
        
        # Save dashboard config to file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dashboard_config, f, indent=2)
            config_file = f.name
        
        print(f"\n   Dashboard configuration saved to: {config_file}")
        print("\n   To create dashboard, run:")
        print(f"   gcloud monitoring dashboards create --config-from-file={config_file}")
        print("\n   Or visit: https://console.cloud.google.com/monitoring/dashboards")
        print("   And create dashboard manually with these metrics:")
        print("   - custom.googleapis.com/ml/model/r2_score")
        print("   - custom.googleapis.com/ml/model/degradation_pct")
        print("   - custom.googleapis.com/ml/model/drift_percentage")
        
        return f"https://console.cloud.google.com/monitoring/dashboards?project={self.project_id}"
    
    def create_alert_policies(self):
        """Create alerting policies for model monitoring"""
        
        if not self.alert_client:
            print("\n⚠️  Alert policy client not available")
            print("   Create alerts manually in GCP Console:")
            print("   https://console.cloud.google.com/monitoring/alerting")
            return
        
        print("\n🚨 Creating alert policies...")
        
        policies = [
            {
                "display_name": "ML Model - High Performance Degradation",
                "conditions": [{
                    "display_name": "Performance degraded > 15%",
                    "condition_threshold": {
                        "filter": 'metric.type="custom.googleapis.com/ml/model/degradation_pct"',
                        "comparison": "COMPARISON_GT",
                        "threshold_value": 15.0,
                        "duration": {"seconds": 300},
                        "aggregations": [{
                            "alignment_period": {"seconds": 60},
                            "per_series_aligner": "ALIGN_MEAN"
                        }]
                    }
                }],
                "combiner": "OR",
                "enabled": True
            },
            {
                "display_name": "ML Model - Significant Data Drift",
                "conditions": [{
                    "display_name": "Data drift > 20%",
                    "condition_threshold": {
                        "filter": 'metric.type="custom.googleapis.com/ml/model/drift_percentage"',
                        "comparison": "COMPARISON_GT",
                        "threshold_value": 20.0,
                        "duration": {"seconds": 300},
                        "aggregations": [{
                            "alignment_period": {"seconds": 60},
                            "per_series_aligner": "ALIGN_MEAN"
                        }]
                    }
                }],
                "combiner": "OR",
                "enabled": True
            }
        ]
        
        for policy in policies:
            try:
                self.alert_client.create_alert_policy(
                    name=self.project_name,
                    alert_policy=policy
                )
                print(f"   ✅ Created: {policy['display_name']}")
            except Exception as e:
                if "already exists" in str(e):
                    print(f"   ℹ️  Already exists: {policy['display_name']}")
                else:
                    print(f"   ❌ Error creating {policy['display_name']}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup GCP Cloud Monitoring for ML models"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
        help="GCP Project ID"
    )
    parser.add_argument(
        "--monitoring-report",
        type=str,
        default="logs/monitoring/monitoring_report_latest.json",
        help="Path to monitoring report JSON"
    )
    parser.add_argument(
        "--create-dashboard",
        action="store_true",
        help="Create monitoring dashboard"
    )
    parser.add_argument(
        "--create-alerts",
        action="store_true",
        help="Create alert policies"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"GCP CLOUD MONITORING SETUP")
    print(f"{'='*80}\n")
    print(f"Project: {args.project_id}")
    
    setup = GCPMonitoringSetup(args.project_id)
    
    # Create metric descriptors
    setup.create_custom_metric_descriptors()
    
    # Push metrics from monitoring report
    if Path(args.monitoring_report).exists():
        setup.push_metrics_from_monitoring_report(args.monitoring_report)
    else:
        print(f"\n⚠️  Monitoring report not found: {args.monitoring_report}")
        print("   Run model monitoring first to generate the report")
    
    # Create dashboard
    if args.create_dashboard:
        setup.create_dashboard()
    
    # Create alerts
    if args.create_alerts:
        setup.create_alert_policies()
    
    print(f"\n{'='*80}")
    print("✅ GCP Monitoring setup complete!")
    print(f"{'='*80}\n")
    print("Next steps:")
    print("1. View dashboard: https://console.cloud.google.com/monitoring/dashboards")
    print("2. Configure notifications: https://console.cloud.google.com/monitoring/alerting")
    print("3. Run monitoring regularly to update metrics")


if __name__ == "__main__":
    main()