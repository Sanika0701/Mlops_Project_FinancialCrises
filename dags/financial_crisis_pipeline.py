"""
Financial Crisis Detection Pipeline - Clean & Modular DAG with Alerting + GCS
==============================================================================
Includes all validation checkpoints + GCS upload integration
 
Author: MLOps Group11 Team
"""
 
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import sys
 
# Add project to path for imports
PROJECT_DIR = '/opt/airflow/project'
sys.path.insert(0, PROJECT_DIR)
 
# Import your existing alerting system
try:
    from src.monitoring.alerting import AlertManager
    ALERTING_AVAILABLE = True
except ImportError:
    print("WARNING: AlertManager not available. Alerts disabled.")
    ALERTING_AVAILABLE = False

# GCS Configuration
GCS_BUCKET = 'mlops-financial-stress-data'
GCS_CREDENTIALS_PATH = '/opt/airflow/gcs-key.json'  # Mounted from docker-compose
 
# ==============================================================================
# CONFIGURATION
# ==============================================================================
 
# Pipeline steps configuration
PIPELINE_STEPS = [
    ('step0_collect_data', 'src/data/step0_data_collection.py', 'Collect data from APIs', 90, True),
    ('validate_checkpoint_1', 'src/validation/validate_checkpoint_1_raw.py', 'Validate raw data', 10, True),
    ('step1_data_cleaning_and_merging', 'src/data/step1_data_cleaning_and_merging.py', 'Clean & merge data', 20, True),
    ('validate_checkpoint_2', 'src/validation/validate_checkpoint_2_clean.py', 'Validate clean data', 10, True),
    ('step2_feature_engineering', 'src/data/step2_feature_engineering.py', 'Engineer features', 20, False),
    ('step3_bias_detection_with_explicit_slicing', 'src/data/step3_bias_detection_with_explicit_slicing.py', 'Detect bias', 10, False),
    ('step4_anomaly_detection', 'src/data/step4_anomaly_detection.py', 'Detect anomalies', 10, False),
    ('step5_drift_detection', 'src/data/step5_drift_detection.py', 'Detect drift', 10, False),
]
 
# ==============================================================================
# GCS CREDENTIAL VERIFICATION FUNCTION
# ==============================================================================

def verify_gcs_setup(**context):
    """
    Verify GCS credentials and bucket access before running pipeline.
    """
    import json
    from google.cloud import storage
    
    try:
        print("=" * 80)
        print("VERIFYING GCS SETUP")
        print("=" * 80)
        
        # Check if credentials file exists
        if not os.path.exists(GCS_CREDENTIALS_PATH):
            raise FileNotFoundError(
                f"GCS credentials not found at: {GCS_CREDENTIALS_PATH}\n"
                f"Please ensure the service account key is mounted in docker-compose.yml"
            )
        
        print(f"✓ Credentials file found: {GCS_CREDENTIALS_PATH}")
        
        # Validate JSON format
        with open(GCS_CREDENTIALS_PATH, 'r') as f:
            creds_data = json.load(f)
        
        if not isinstance(creds_data, dict):
            raise ValueError("Credentials file is not valid JSON")
        
        # Check for required fields
        required_fields = ['type', 'project_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if field not in creds_data]
        
        if missing_fields:
            raise ValueError(f"Credentials missing required fields: {missing_fields}")
        
        print(f"✓ Credentials format validated")
        print(f"  Service Account: {creds_data.get('client_email')}")
        print(f"  Project ID: {creds_data.get('project_id')}")
        
        # Test GCS connection
        client = storage.Client.from_service_account_json(GCS_CREDENTIALS_PATH)
        
        # Try to access bucket
        bucket = client.bucket(GCS_BUCKET)
        
        if not bucket.exists():
            raise ValueError(
                f"Bucket '{GCS_BUCKET}' does not exist or service account lacks access.\n"
                f"Please create the bucket or grant 'Storage Admin' role to:\n"
                f"  {creds_data.get('client_email')}"
            )
        
        print(f"✓ Successfully connected to bucket: gs://{GCS_BUCKET}/")
        
        # List existing files (optional - shows what's already there)
        blobs = list(bucket.list_blobs(prefix='data/raw/', max_results=5))
        if blobs:
            print(f"✓ Found {len(blobs)} existing files in data/raw/:")
            for blob in blobs[:3]:
                print(f"    - {blob.name}")
            if len(blobs) > 3:
                print(f"    ... and {len(blobs) - 3} more")
        else:
            print(f"  Note: No existing files in data/raw/ (this is fine for first run)")
        
        print("=" * 80)
        print("✓ GCS SETUP VERIFICATION PASSED")
        print("=" * 80)
        
        # Push success to XCom
        context['ti'].xcom_push(key='gcs_verified', value='success')
        context['ti'].xcom_push(key='project_id', value=creds_data.get('project_id'))
        context['ti'].xcom_push(key='service_account', value=creds_data.get('client_email'))
        
        return True
        
    except FileNotFoundError as e:
        print("=" * 80)
        print("✗ CREDENTIALS FILE NOT FOUND")
        print("=" * 80)
        print(str(e))
        print("\nFIX: Update docker-compose.yml to mount your service account key:")
        print(f"  - /Users/priyanka/Downloads/ninth-iris-422916-f2-9aec4e0969c6.json:/opt/airflow/gcs-key.json:ro")
        print("=" * 80)
        raise
        
    except json.JSONDecodeError as e:
        print("=" * 80)
        print("✗ INVALID JSON IN CREDENTIALS FILE")
        print("=" * 80)
        print(f"Error: {e}")
        print(f"\nThe file at {GCS_CREDENTIALS_PATH} is not valid JSON")
        print("=" * 80)
        raise
        
    except ValueError as e:
        print("=" * 80)
        print("✗ CREDENTIALS VALIDATION FAILED")
        print("=" * 80)
        print(str(e))
        print("=" * 80)
        raise
        
    except Exception as e:
        print("=" * 80)
        print("✗ GCS SETUP VERIFICATION FAILED")
        print("=" * 80)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        raise

# ==============================================================================
# ALERTING CALLBACKS
# ==============================================================================
 
def task_failure_alert(context):
    """Send alert on task failure using your AlertManager"""
    if not ALERTING_AVAILABLE:
        return
    try:
        task = context.get('task_instance')
        dag_run = context.get('dag_run')
        exception = context.get('exception')
        
        alert_manager = AlertManager()
        
        is_validation = 'validate' in task.task_id
        is_critical = any(step[0] == task.task_id and step[4] for step in PIPELINE_STEPS)
        severity = 'CRITICAL' if (is_validation or is_critical) else 'ERROR'
        
        message = f"""
        Pipeline Task Failed: {task.task_id}
        DAG: {task.dag_id}
        Execution Date: {dag_run.execution_date}
        Error: {str(exception) if exception else 'Check logs for details'}
        Log URL: {task.log_url}
        """
        
        if is_validation:
            message += "\nWARNING: Data Validation Failed - Pipeline stopped"
        
        alert_manager.send_alert(
            message=message,
            severity=severity,
            component=task.task_id,
            alert_type='PIPELINE_FAILURE'
        )
        print(f"Alert sent for {task.task_id} failure")
    except Exception as e:
        print(f"Failed to send alert: {str(e)}")
 
 
def pipeline_success_alert(**context):
    """Send alert on pipeline success"""
    if not ALERTING_AVAILABLE:
        return
    try:
        dag_run = context.get('dag_run')
        ti = context.get('ti')
        
        # Get info from verification task
        service_account = ti.xcom_pull(task_ids='verify_gcs_setup', key='service_account')
        
        duration = dag_run.end_date - dag_run.start_date if dag_run.end_date else "N/A"
        
        alert_manager = AlertManager()
        
        message = f"""
        SUCCESS: Financial Crisis Pipeline Completed
        Execution Date: {dag_run.execution_date}
        Duration: {duration}
        
        All data uploaded to: gs://{GCS_BUCKET}/
        Service Account: {service_account}
        
        Pipeline Summary:
        ✓ Data collected & uploaded to GCS
        ✓ Data validated (Checkpoint 1)
        ✓ Data cleaned & merged
        ✓ Data validated (Checkpoint 2)
        ✓ Features engineered
        ✓ Bias detection completed
        ✓ Anomaly detection completed
        ✓ Drift detection completed
        
        Data ready for model training!
        
        View data: https://console.cloud.google.com/storage/browser/{GCS_BUCKET}/data/raw
        """
        
        alert_manager.send_alert(
            message=message,
            severity='INFO',
            component='pipeline',
            alert_type='PIPELINE_SUCCESS'
        )
        print("Success alert sent")
    except Exception as e:
        print(f"Failed to send success alert: {str(e)}")
 
 
def validation_failure_alert(context):
    """Special alert for validation checkpoint failures"""
    if not ALERTING_AVAILABLE:
        return
    try:
        task = context.get('task_instance')
        alert_manager = AlertManager()
        
        checkpoint_num = '1' if 'checkpoint_1' in task.task_id else '2'
        
        message = f"""
        CRITICAL: Validation Checkpoint {checkpoint_num} Failed
        Task: {task.task_id}
        Execution Date: {context.get('dag_run').execution_date}
        
        Data quality issues detected. Pipeline stopped.
        
        Action Required:
        1. Check validation report: data/validation_reports/
        2. Review failed expectations
        3. Fix data quality issues
        4. Re-run pipeline
        
        Log URL: {task.log_url}
        """
        
        alert_manager.send_alert(
            message=message,
            severity='CRITICAL',
            component=f'validation_checkpoint_{checkpoint_num}',
            alert_type='VALIDATION_FAILURE'
        )
        print(f"CRITICAL validation alert sent for checkpoint {checkpoint_num}")
    except Exception as e:
        print(f"Failed to send validation alert: {str(e)}")
 
 
# ==============================================================================
# DAG DEFAULT ARGS
# ==============================================================================
 
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': task_failure_alert,
}
 
# ==============================================================================
# DAG DEFINITION
# ==============================================================================
 
with DAG(
    'financial_crisis_pipeline',
    default_args=default_args,
    description='Financial Crisis Pipeline with GCS Upload & Validation',
    schedule_interval=None,  # Manual trigger
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'financial', 'validation', 'gcs'],
    max_active_runs=1,
) as dag:
    
    # ==========================================================================
    # SETUP TASK - Verify GCS credentials and access
    # ==========================================================================
    
    verify_gcs = PythonOperator(
        task_id='verify_gcs_setup',
        python_callable=verify_gcs_setup,
        provide_context=True,
    )
    
    # ==========================================================================
    # CREATE PIPELINE TASKS DYNAMICALLY
    # ==========================================================================
    
    tasks = {}
    
    for task_id, script, desc, timeout, critical in PIPELINE_STEPS:
        # Validation checkpoints get special alerting
        callbacks = {}
        if 'validate' in task_id:
            callbacks['on_failure_callback'] = validation_failure_alert
        else:
            callbacks['on_failure_callback'] = task_failure_alert
        
        # Create task with proper GCS credentials
        tasks[task_id] = BashOperator(
            task_id=task_id,
            bash_command=f"""
            cd {PROJECT_DIR} && \
            echo "{'=' * 60}" && \
            echo "{'VALIDATING' if 'validate' in task_id else 'RUNNING'}: {desc}" && \
            echo "{'=' * 60}" && \
            export GOOGLE_APPLICATION_CREDENTIALS={GCS_CREDENTIALS_PATH} && \
            export GCS_BUCKET={GCS_BUCKET} && \
            python {script} && \
            echo "{'=' * 60}" && \
            echo "✓ {desc} completed successfully" && \
            echo "{'=' * 60}"
            """,
            execution_timeout=timedelta(minutes=timeout),
            env={
                'GOOGLE_APPLICATION_CREDENTIALS': GCS_CREDENTIALS_PATH,
                'GCS_BUCKET': GCS_BUCKET,
                'PYTHONUNBUFFERED': '1',  # Ensure real-time logging
            },
            **callbacks
        )
    
    # ==========================================================================
    # SUCCESS NOTIFICATION TASK
    # ==========================================================================
    
    pipeline_success = PythonOperator(
        task_id='pipeline_success',
        python_callable=pipeline_success_alert,
        provide_context=True,
        trigger_rule='all_success'
    )
    
    # ==========================================================================
    # DEFINE DEPENDENCIES
    # ==========================================================================
    
    # Verify GCS setup first, then start pipeline
    first_task = PIPELINE_STEPS[0][0]
    verify_gcs >> tasks[first_task]
    
    # Chain all pipeline steps sequentially
    for i in range(len(PIPELINE_STEPS) - 1):
        current_task = PIPELINE_STEPS[i][0]
        next_task = PIPELINE_STEPS[i + 1][0]
        tasks[current_task] >> tasks[next_task]
    
    # Add success notification at the end
    tasks[PIPELINE_STEPS[-1][0]] >> pipeline_success

