"""
Apache Airflow DAG for orchestrating multimodal model training pipeline.
Handles data preparation, model training, evaluation, and deployment.
"""
from datetime import datetime, timedelta
from typing import Dict, Any
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.models import Variable
from airflow.exceptions import AirflowException

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Default configuration
DEFAULT_ARGS = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1,
}

# DAG configuration
DAG_ID = 'multimodal_model_training'
SCHEDULE_INTERVAL = '@weekly'  # Run weekly
CATCHUP = False

# Get configuration from Airflow Variables
TRAINING_CONFIG = {
    'model_name': Variable.get('MODEL_NAME', default_var='clip-lora'),
    'batch_size': int(Variable.get('BATCH_SIZE', default_var='32')),
    'learning_rate': float(Variable.get('LEARNING_RATE', default_var='2e-5')),
    'num_epochs': int(Variable.get('NUM_EPOCHS', default_var='3')),
    'data_path': Variable.get('DATA_PATH', default_var='/data/coco'),
    'output_dir': Variable.get('OUTPUT_DIR', default_var='/outputs'),
    'use_distributed': Variable.get('USE_DISTRIBUTED', default_var='true').lower() == 'true',
    'num_gpus': int(Variable.get('NUM_GPUS', default_var='4')),
}

# MLflow configuration
MLFLOW_CONFIG = {
    'tracking_uri': Variable.get('MLFLOW_TRACKING_URI', default_var='http://mlflow:5000'),
    'experiment_name': Variable.get('MLFLOW_EXPERIMENT_NAME', default_var='multimodal-training'),
}

# Notification configuration
SLACK_WEBHOOK = Variable.get('SLACK_WEBHOOK', default_var=None)


def check_data_availability(**context) -> bool:
    """Check if training data is available and valid."""
    import os
    from pathlib import Path
    
    data_path = Path(TRAINING_CONFIG['data_path'])
    
    # Check if data directory exists
    if not data_path.exists():
        raise AirflowException(f"Data path {data_path} does not exist")
    
    # Check for required files/directories
    required_paths = [
        data_path / "train",
        data_path / "val",
    ]
    
    for path in required_paths:
        if not path.exists():
            raise AirflowException(f"Required path {path} not found")
    
    # Check minimum file count
    train_files = list((data_path / "train").rglob("*.jpg")) + list((data_path / "train").rglob("*.png"))
    if len(train_files) < 1000:
        raise AirflowException(f"Insufficient training data: only {len(train_files)} images found")
    
    print(f"Data validation passed. Found {len(train_files)} training images")
    return True


def prepare_training_environment(**context) -> Dict[str, Any]:
    """Prepare the training environment and configuration."""
    import json
    from pathlib import Path
    
    # Create output directory
    output_dir = Path(TRAINING_CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create run-specific directory
    run_id = context['run_id']
    run_dir = output_dir / f"run_{run_id}"
    run_dir.mkdir(exist_ok=True)
    
    # Save training configuration
    config_path = run_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(TRAINING_CONFIG, f, indent=2)
    
    # Set environment variables
    env_vars = {
        'CUDA_VISIBLE_DEVICES': ','.join(map(str, range(TRAINING_CONFIG['num_gpus']))),
        'MLFLOW_TRACKING_URI': MLFLOW_CONFIG['tracking_uri'],
        'TRAINING_OUTPUT_DIR': str(run_dir),
        'CONFIG_PATH': str(config_path),
    }
    
    # Push to XCom for downstream tasks
    context['task_instance'].xcom_push(key='env_vars', value=env_vars)
    context['task_instance'].xcom_push(key='run_dir', value=str(run_dir))
    
    print(f"Training environment prepared at {run_dir}")
    return env_vars


def start_mlflow_run(**context) -> str:
    """Start MLflow run for experiment tracking."""
    mlflow.set_tracking_uri(MLFLOW_CONFIG['tracking_uri'])
    
    # Create or get experiment
    experiment_name = MLFLOW_CONFIG['experiment_name']
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error with MLflow experiment: {e}")
        experiment_id = None
    
    # Start run
    run_name = f"airflow_run_{context['run_id']}"
    
    run = mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags={
            'airflow_dag_id': context['dag'].dag_id,
            'airflow_task_id': context['task'].task_id,
            'airflow_run_id': context['run_id'],
            'scheduled': str(context.get('dag_run', {}).get('external_trigger', False) == False)
        }
    )
    
    # Log configuration
    mlflow.log_params(TRAINING_CONFIG)
    
    # Push run ID to XCom
    context['task_instance'].xcom_push(key='mlflow_run_id', value=run.info.run_id)
    
    print(f"Started MLflow run: {run.info.run_id}")
    return run.info.run_id


def execute_training(**context) -> Dict[str, Any]:
    """Execute the model training process."""
    import subprocess
    import json
    
    # Get environment variables from XCom
    env_vars = context['task_instance'].xcom_pull(key='env_vars', task_ids='prepare_environment')
    run_dir = context['task_instance'].xcom_pull(key='run_dir', task_ids='prepare_environment')
    mlflow_run_id = context['task_instance'].xcom_pull(key='mlflow_run_id', task_ids='start_mlflow_run')
    
    # Update environment
    training_env = os.environ.copy()
    training_env.update(env_vars)
    training_env['MLFLOW_RUN_ID'] = mlflow_run_id
    
    # Prepare training command
    if TRAINING_CONFIG['use_distributed']:
        cmd = [
            'python', '-m', 'torch.distributed.launch',
            f'--nproc_per_node={TRAINING_CONFIG["num_gpus"]}',
            'scripts/train_distributed.py',
            f'--config={env_vars["CONFIG_PATH"]}',
            f'--output-dir={run_dir}'
        ]
    else:
        cmd = [
            'python', 'scripts/train_single.py',
            f'--config={env_vars["CONFIG_PATH"]}',
            f'--output-dir={run_dir}'
        ]
    
    # Execute training
    try:
        result = subprocess.run(
            cmd,
            env=training_env,
            capture_output=True,
            text=True,
            cwd='/opt/multimodal-foundation-model',
            timeout=3600 * 8  # 8 hour timeout
        )
        
        if result.returncode != 0:
            raise AirflowException(f"Training failed with return code {result.returncode}\nStderr: {result.stderr}")
        
        print("Training completed successfully")
        print(f"Stdout: {result.stdout}")
        
        # Parse training results
        results_file = Path(run_dir) / "training_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                training_results = json.load(f)
        else:
            training_results = {"status": "completed"}
        
        # Push results to XCom
        context['task_instance'].xcom_push(key='training_results', value=training_results)
        
        return training_results
        
    except subprocess.TimeoutExpired:
        raise AirflowException("Training timed out after 8 hours")
    except Exception as e:
        raise AirflowException(f"Training execution failed: {str(e)}")


def evaluate_model(**context) -> Dict[str, float]:
    """Evaluate the trained model."""
    import subprocess
    import json
    
    # Get training results
    run_dir = context['task_instance'].xcom_pull(key='run_dir', task_ids='prepare_environment')
    training_results = context['task_instance'].xcom_pull(key='training_results', task_ids='train_model')
    
    # Find best checkpoint
    checkpoint_dir = Path(run_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        raise AirflowException("No checkpoints found for evaluation")
    
    # Run evaluation
    eval_cmd = [
        'python', 'scripts/evaluate_model.py',
        f'--checkpoint-dir={checkpoint_dir}',
        f'--data-path={TRAINING_CONFIG["data_path"]}/val',
        f'--output-dir={run_dir}/evaluation'
    ]
    
    try:
        result = subprocess.run(
            eval_cmd,
            capture_output=True,
            text=True,
            cwd='/opt/multimodal-foundation-model',
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode != 0:
            raise AirflowException(f"Evaluation failed: {result.stderr}")
        
        # Parse evaluation results
        eval_results_file = Path(run_dir) / "evaluation" / "results.json"
        if eval_results_file.exists():
            with open(eval_results_file, 'r') as f:
                eval_results = json.load(f)
        else:
            eval_results = {"error": "No evaluation results found"}
        
        print(f"Evaluation results: {eval_results}")
        
        # Log to MLflow
        mlflow_run_id = context['task_instance'].xcom_pull(key='mlflow_run_id', task_ids='start_mlflow_run')
        
        with mlflow.start_run(run_id=mlflow_run_id):
            for metric, value in eval_results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"eval_{metric}", value)
        
        # Push results to XCom
        context['task_instance'].xcom_push(key='eval_results', value=eval_results)
        
        return eval_results
        
    except subprocess.TimeoutExpired:
        raise AirflowException("Evaluation timed out")
    except Exception as e:
        raise AirflowException(f"Evaluation failed: {str(e)}")


def check_model_quality(**context) -> bool:
    """Check if model quality meets deployment criteria."""
    eval_results = context['task_instance'].xcom_pull(key='eval_results', task_ids='evaluate_model')
    
    # Define quality thresholds
    quality_thresholds = {
        'bleu_4': 0.20,  # BLEU-4 score
        'cider': 0.80,   # CIDEr score
        'eval_loss': 2.0  # Maximum acceptable loss
    }
    
    quality_check_passed = True
    failed_metrics = []
    
    for metric, threshold in quality_thresholds.items():
        if metric in eval_results:
            value = eval_results[metric]
            
            if metric == 'eval_loss':
                # Lower is better for loss
                if value > threshold:
                    quality_check_passed = False
                    failed_metrics.append(f"{metric}: {value} > {threshold}")
            else:
                # Higher is better for other metrics
                if value < threshold:
                    quality_check_passed = False
                    failed_metrics.append(f"{metric}: {value} < {threshold}")
    
    if not quality_check_passed:
        print(f"Quality check failed. Failed metrics: {failed_metrics}")
        # Don't fail the task, but log the issue
        context['task_instance'].xcom_push(key='quality_check_passed', value=False)
        context['task_instance'].xcom_push(key='failed_metrics', value=failed_metrics)
    else:
        print("Quality check passed")
        context['task_instance'].xcom_push(key='quality_check_passed', value=True)
    
    return quality_check_passed


def register_model(**context) -> Dict[str, str]:
    """Register the model in MLflow model registry."""
    quality_passed = context['task_instance'].xcom_pull(key='quality_check_passed', task_ids='check_quality')
    mlflow_run_id = context['task_instance'].xcom_pull(key='mlflow_run_id', task_ids='start_mlflow_run')
    
    if not quality_passed:
        print("Skipping model registration due to quality check failure")
        return {"status": "skipped", "reason": "quality_check_failed"}
    
    mlflow.set_tracking_uri(MLFLOW_CONFIG['tracking_uri'])
    
    # Register model
    model_name = f"multimodal-{TRAINING_CONFIG['model_name']}"
    model_uri = f"runs:/{mlflow_run_id}/model"
    
    try:
        registered_model = mlflow.register_model(model_uri, model_name)
        
        # Transition to Staging
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage="Staging"
        )
        
        result = {
            "status": "registered",
            "model_name": model_name,
            "version": registered_model.version,
            "stage": "Staging"
        }
        
        print(f"Model registered: {result}")
        return result
        
    except Exception as e:
        print(f"Model registration failed: {str(e)}")
        return {"status": "failed", "error": str(e)}


def send_notification(**context) -> None:
    """Send notification about training completion."""
    if SLACK_WEBHOOK is None:
        print("No Slack webhook configured, skipping notification")
        return
    
    # Gather results
    training_results = context['task_instance'].xcom_pull(key='training_results', task_ids='train_model')
    eval_results = context['task_instance'].xcom_pull(key='eval_results', task_ids='evaluate_model')
    quality_passed = context['task_instance'].xcom_pull(key='quality_check_passed', task_ids='check_quality')
    registration_result = context['task_instance'].xcom_pull(key='return_value', task_ids='register_model')
    
    # Format message
    status_emoji = "✅" if quality_passed else "⚠️"
    
    message = f"{status_emoji} *Multimodal Model Training Completed*\n\n"
    message += f"*Run ID:* {context['run_id']}\n"
    message += f"*Model:* {TRAINING_CONFIG['model_name']}\n"
    message += f"*Quality Check:* {'Passed' if quality_passed else 'Failed'}\n\n"
    
    if eval_results:
        message += "*Evaluation Results:*\n"
        for metric, value in eval_results.items():
            if isinstance(value, (int, float)):
                message += f"• {metric}: {value:.4f}\n"
    
    if registration_result and registration_result.get('status') == 'registered':
        message += f"\n*Model Registry:* Version {registration_result['version']} in {registration_result['stage']}\n"
    
    # Send to Slack (this would require the actual SlackWebhookOperator configuration)
    print(f"Notification message: {message}")


def cleanup_resources(**context) -> None:
    """Clean up temporary resources and files."""
    import shutil
    
    run_dir = context['task_instance'].xcom_pull(key='run_dir', task_ids='prepare_environment')
    mlflow_run_id = context['task_instance'].xcom_pull(key='mlflow_run_id', task_ids='start_mlflow_run')
    
    # End MLflow run
    try:
        mlflow.set_tracking_uri(MLFLOW_CONFIG['tracking_uri'])
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.end_run()
        print(f"Ended MLflow run: {mlflow_run_id}")
    except Exception as e:
        print(f"Failed to end MLflow run: {e}")
    
    # Clean up old training runs (keep last 5)
    try:
        output_dir = Path(TRAINING_CONFIG['output_dir'])
        run_dirs = sorted([d for d in output_dir.glob('run_*') if d.is_dir()], key=lambda x: x.stat().st_mtime)
        
        if len(run_dirs) > 5:
            for old_dir in run_dirs[:-5]:
                shutil.rmtree(old_dir)
                print(f"Cleaned up old run directory: {old_dir}")
    except Exception as e:
        print(f"Failed to clean up old runs: {e}")
    
    print("Cleanup completed")


# Create DAG
dag = DAG(
    DAG_ID,
    default_args=DEFAULT_ARGS,
    description='Multimodal Model Training Pipeline',
    schedule_interval=SCHEDULE_INTERVAL,
    catchup=CATCHUP,
    tags=['ml', 'multimodal', 'training'],
)

# Define tasks
check_data_task = PythonOperator(
    task_id='check_data',
    python_callable=check_data_availability,
    dag=dag,
)

prepare_env_task = PythonOperator(
    task_id='prepare_environment',
    python_callable=prepare_training_environment,
    dag=dag,
)

start_mlflow_task = PythonOperator(
    task_id='start_mlflow_run',
    python_callable=start_mlflow_run,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=execute_training,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

quality_check_task = PythonOperator(
    task_id='check_quality',
    python_callable=check_model_quality,
    dag=dag,
)

register_model_task = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    dag=dag,
)

notify_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    trigger_rule='all_done',  # Run regardless of upstream task status
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup_resources,
    trigger_rule='all_done',
    dag=dag,
)

# Define task dependencies
check_data_task >> prepare_env_task
prepare_env_task >> start_mlflow_task
start_mlflow_task >> train_task
train_task >> evaluate_task
evaluate_task >> quality_check_task
quality_check_task >> register_model_task
register_model_task >> notify_task
notify_task >> cleanup_task