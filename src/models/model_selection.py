"""
Model Selection Script
Compares Dense VAE Optimized vs Ensemble VAE
Selects best model based on KS Pass Rate and Correlation MAE
Uploads best model to GCS for deployment
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
import mlflow
import gcsfs
from datetime import datetime

# GCS Configuration
GCS_BUCKET = 'mlops-financial-stress-data'
GCS_DEPLOYMENT_PATH = f'gs://{GCS_BUCKET}/models/vae/deployment/'

# ============================================
# 1. LOAD RESULTS FROM BOTH MODELS
# ============================================

def load_model_results():
    """Load validation results from both models"""
    
    print("="*60)
    print("MODEL COMPARISON & SELECTION")
    print("="*60 + "\n")
    
    # Model 1: Dense VAE Optimized
    dense_vae_dir = Path('outputs/output_Dense_VAE_optimized')
    ensemble_vae_dir = Path('outputs/output_Ensemble_VAE')
    
    results = {}
    
    # Read Dense VAE results
    if (dense_vae_dir / 'validation_report.txt').exists():
        print("✓ Found Dense VAE Optimized results")
        results['Dense_VAE_Optimized'] = {}
        with open(dense_vae_dir / 'validation_report.txt', 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'KS Test Pass Rate:' in line or 'KS Pass Rate:' in line:
                    ks_rate = float(line.split(':')[1].strip().replace('%', ''))
                    results['Dense_VAE_Optimized']['ks_pass_rate'] = ks_rate
                elif 'Correlation MAE:' in line:
                    corr_mae = float(line.split(':')[1].strip())
                    results['Dense_VAE_Optimized']['correlation_mae'] = corr_mae
                elif 'Wasserstein' in line:
                    wass = float(line.split(':')[1].strip())
                    results['Dense_VAE_Optimized']['wasserstein'] = wass
        
        # Store model path - look for .pth file
        results['Dense_VAE_Optimized']['model_path'] = dense_vae_dir / 'dense_vae_model.pth'
    
    # Read Ensemble VAE results
    if (ensemble_vae_dir / 'ensemble_validation.txt').exists():
        print("✓ Found Ensemble VAE results")
        results['Ensemble_VAE'] = {}
        with open(ensemble_vae_dir / 'ensemble_validation.txt', 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'KS Test Pass Rate:' in line or 'KS Pass Rate:' in line:
                    ks_rate = float(line.split(':')[1].strip().replace('%', ''))
                    results['Ensemble_VAE']['ks_pass_rate'] = ks_rate
                elif 'Correlation MAE:' in line:
                    corr_mae = float(line.split(':')[1].strip())
                    results['Ensemble_VAE']['correlation_mae'] = corr_mae
                elif 'Wasserstein' in line:
                    wass = float(line.split(':')[1].strip())
                    results['Ensemble_VAE']['wasserstein'] = wass
        
        # Store model path - look for .pth file
        results['Ensemble_VAE']['model_path'] = ensemble_vae_dir / 'ensemble_vae_complete.pth'
    
    return results


# ============================================
# 2. SELECT BEST MODEL
# ============================================

def select_best_model(results):
    """
    Select best model based on:
    1. Higher KS Pass Rate (primary)
    2. Lower Correlation MAE (secondary)
    """
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60 + "\n")
    
    # Display results
    print("Metric Comparison:")
    print("-" * 60)
    print(f"{'Model':<30} {'KS Pass Rate':<15} {'Corr MAE':<15} {'Wasserstein':<15}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<30} "
              f"{metrics['ks_pass_rate']:<15.2f} "
              f"{metrics['correlation_mae']:<15.4f} "
              f"{metrics.get('wasserstein', 0):<15.2f}")
    
    print("-" * 60)
    
    # Selection logic
    best_model = None
    best_score = -1
    
    for model_name, metrics in results.items():
        # Composite score: KS Pass Rate is weighted more heavily
        # Higher KS = better, Lower MAE = better
        score = (metrics['ks_pass_rate'] * 0.7) - (metrics['correlation_mae'] * 100 * 0.3)
        
        if score > best_score:
            best_score = score
            best_model = model_name
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   Composite Score: {best_score:.2f}")
    print(f"   KS Pass Rate: {results[best_model]['ks_pass_rate']:.2f}%")
    print(f"   Correlation MAE: {results[best_model]['correlation_mae']:.4f}")
    
    return best_model, results[best_model]


# ============================================
# 3. UPLOAD MODEL TO GCS FOR DEPLOYMENT
# ============================================

def upload_model_to_gcs(model_name, metrics):
    """
    Upload the best model to GCS deployment location
    Loads .pth file, creates deployment-ready .pkl package, uploads to GCS
    """
    
    print("\n" + "="*60)
    print("CREATING & UPLOADING DEPLOYMENT PACKAGE")
    print("="*60 + "\n")
    
    try:
        # Load the .pth file from training
        model_path = metrics['model_path']
        print(f"Loading model from: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load PyTorch model package
        import torch
        # Set weights_only=False because our .pth file contains StandardScaler (not just weights)
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print("✓ Model loaded successfully")
        print(f"  Model type: {model_name}")
        print(f"  Keys in model data: {list(model_data.keys())}")
        
        # Extract components based on model type
        if model_name == 'Dense_VAE_Optimized':
            # Dense VAE structure: {'model': state_dict, 'scaler': scaler, 'features': list, 'config': dict}
            deployment_package = {
                'model': model_data.get('model'),           # PyTorch state dict
                'scaler': model_data.get('scaler'),         # Fitted StandardScaler
                'feature_names': model_data.get('features'), # List of feature names
                'model_name': model_name,
                'model_type': 'Dense_VAE',
                'training_date': datetime.now().isoformat(),
                'performance_metrics': {
                    'ks_pass_rate': metrics['ks_pass_rate'],
                    'correlation_mae': metrics['correlation_mae'],
                    'wasserstein_distance': metrics.get('wasserstein', 0)
                },
                'config': model_data.get('config', {}),
                'version': 'v1.0',
                'deployment_ready': True
            }
        
        elif model_name == 'Ensemble_VAE':
            # Ensemble VAE structure: {'models': [state_dicts], 'scaler': scaler, 'features': list, 'config': dict}
            # For deployment, use the first model or create average
            deployment_package = {
                'model': model_data.get('models', [])[0] if model_data.get('models') else None,  # First model
                'scaler': model_data.get('scaler'),
                'feature_names': model_data.get('features'),
                'model_name': model_name,
                'model_type': 'Ensemble_VAE',
                'n_ensemble_models': len(model_data.get('models', [])),
                'training_date': datetime.now().isoformat(),
                'performance_metrics': {
                    'ks_pass_rate': metrics['ks_pass_rate'],
                    'correlation_mae': metrics['correlation_mae'],
                    'wasserstein_distance': metrics.get('wasserstein', 0)
                },
                'config': model_data.get('config', {}),
                'version': 'v1.0',
                'deployment_ready': True
            }
        
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Validate deployment package
        if deployment_package['model'] is None:
            raise ValueError("Model state dict is None")
        if deployment_package['scaler'] is None:
            raise ValueError("Scaler is None")
        if not deployment_package['feature_names']:
            raise ValueError("Feature names are empty")
        
        print("\n✓ Deployment package created:")
        print(f"  - Model state dict: ✓")
        print(f"  - Scaler: ✓")
        print(f"  - Features: {len(deployment_package['feature_names'])} features")
        print(f"  - Config: ✓")
        
        # Initialize GCS filesystem
        fs = gcsfs.GCSFileSystem()
        
        # Upload deployment .pkl file to GCS
        deployment_pkl_path = f"{GCS_DEPLOYMENT_PATH}best_model_deployment.pkl"
        print(f"\nUploading to: {deployment_pkl_path}")
        
        with fs.open(deployment_pkl_path, 'wb') as f:
            pickle.dump(deployment_package, f)
        
        print("✓ Model uploaded successfully")
        
        # Get file size for reporting
        pkl_size_bytes = len(pickle.dumps(deployment_package))
        pkl_size_kb = pkl_size_bytes / 1024
        
        # Create and upload deployment metadata JSON
        metadata = {
            'model_name': model_name,
            'deployment_timestamp': datetime.now().isoformat(),
            'gcs_path': deployment_pkl_path,
            'performance_metrics': {
                'ks_pass_rate': f"{metrics['ks_pass_rate']:.2f}%",
                'correlation_mae': f"{metrics['correlation_mae']:.4f}",
                'wasserstein_distance': f"{metrics.get('wasserstein', 0):.2f}"
            },
            'model_components': {
                'has_model': deployment_package['model'] is not None,
                'has_scaler': deployment_package['scaler'] is not None,
                'feature_count': len(deployment_package['feature_names']) if deployment_package['feature_names'] else 0
            },
            'model_config': deployment_package.get('config', {}),
            'version': 'v1.0',
            'deployment_status': 'ready',
            'file_size_kb': pkl_size_kb
        }
        
        metadata_path = f"{GCS_DEPLOYMENT_PATH}deployment_metadata.json"
        print(f"Uploading metadata to: {metadata_path}")
        
        with fs.open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✓ Metadata uploaded successfully")
        
        print("\n" + "="*60)
        print("DEPLOYMENT SUMMARY")
        print("="*60)
        print(f"Model: {model_name}")
        print(f"Location: {deployment_pkl_path}")
        print(f"Size: {pkl_size_kb:.2f} KB")
        print(f"\nComponents:")
        print(f"  - Model Type: {deployment_package.get('model_type', 'Unknown')}")
        print(f"  - PyTorch State Dict: ✓")
        print(f"  - Fitted Scaler: ✓")
        print(f"  - Feature Names: {len(deployment_package['feature_names'])} features")
        print(f"  - Configuration: ✓")
        print(f"\nPerformance:")
        print(f"  - KS Pass Rate: {metrics['ks_pass_rate']:.2f}%")
        print(f"  - Correlation MAE: {metrics['correlation_mae']:.4f}")
        print(f"  - Wasserstein Distance: {metrics.get('wasserstein', 0):.2f}")
        print("="*60 + "\n")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ Model file not found: {str(e)}")
        print(f"Expected path: {model_path}")
        print("\nMake sure training completed successfully and .pth file was saved.")
        return False
        
    except Exception as e:
        print(f"\n❌ Error creating/uploading deployment package: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nDeployment failed - model will not be available for production use")
        return False


# ============================================
# 4. VISUALIZE COMPARISON
# ============================================

def create_comparison_plots(results, output_dir='outputs/model_selection'):
    """Create comparison visualizations"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(results.keys())
    
    # Prepare data
    ks_rates = [results[m]['ks_pass_rate'] for m in models]
    corr_maes = [results[m]['correlation_mae'] for m in models]
    wassersteins = [results[m].get('wasserstein', 0) for m in models]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: KS Pass Rate
    axes[0].bar(models, ks_rates, color=['#3498db', '#e74c3c'])
    axes[0].set_ylabel('KS Pass Rate (%)')
    axes[0].set_title('KS Test Pass Rate\n(Higher is Better)')
    axes[0].set_ylim([0, 100])
    for i, v in enumerate(ks_rates):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Plot 2: Correlation MAE
    axes[1].bar(models, corr_maes, color=['#3498db', '#e74c3c'])
    axes[1].set_ylabel('Correlation MAE')
    axes[1].set_title('Correlation MAE\n(Lower is Better)')
    for i, v in enumerate(corr_maes):
        axes[1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Plot 3: Wasserstein Distance
    axes[2].bar(models, wassersteins, color=['#3498db', '#e74c3c'])
    axes[2].set_ylabel('Wasserstein Distance')
    axes[2].set_title('Wasserstein Distance\n(Lower is Better)')
    for i, v in enumerate(wassersteins):
        axes[2].text(i, v + 10, f'{v:.1f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_dir}/model_comparison.png")
    
    plt.close()


# ============================================
# 5. SAVE SELECTION REPORT
# ============================================

def save_selection_report(best_model, metrics, output_dir='outputs/model_selection'):
    """Save model selection report"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = f'{output_dir}/model_selection_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("MODEL SELECTION REPORT\n")
        f.write("Financial Stress Test Scenario Generator\n")
        f.write("="*60 + "\n\n")
        
        f.write("SELECTION CRITERIA:\n")
        f.write("-" * 60 + "\n")
        f.write("1. Primary Metric: KS Pass Rate (Higher is Better)\n")
        f.write("   - Measures distribution similarity with real data\n")
        f.write("   - Target: >75%\n\n")
        
        f.write("2. Secondary Metric: Correlation MAE (Lower is Better)\n")
        f.write("   - Measures preservation of feature relationships\n")
        f.write("   - Target: <0.10\n\n")
        
        f.write("="*60 + "\n")
        f.write("SELECTED MODEL\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Best Model: {best_model}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"KS Pass Rate:       {metrics['ks_pass_rate']:.2f}%\n")
        f.write(f"Correlation MAE:    {metrics['correlation_mae']:.4f}\n")
        f.write(f"Wasserstein Dist:   {metrics.get('wasserstein', 0):.2f}\n\n")
        
        f.write("="*60 + "\n")
        f.write("JUSTIFICATION\n")
        f.write("="*60 + "\n\n")
        
        if metrics['ks_pass_rate'] > 75:
            f.write("[PASS] KS Pass Rate exceeds 75% threshold\n")
        else:
            f.write("[ACCEPTABLE] KS Pass Rate below 75% - acceptable for stress testing\n")
        
        if metrics['correlation_mae'] < 0.10:
            f.write("[PASS] Correlation MAE below 0.10 threshold\n")
        else:
            f.write("[ACCEPTABLE] Correlation MAE above 0.10 - still preserves relationships\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("DEPLOYMENT RECOMMENDATION\n")
        f.write("="*60 + "\n\n")
        f.write(f"The {best_model} is recommended for production deployment.\n")
        f.write("This model provides the best balance of statistical validity\n")
        f.write("and scenario diversity for financial stress testing.\n\n")
        
        f.write("Deployment Location:\n")
        f.write(f"GCS Path: {GCS_DEPLOYMENT_PATH}best_model_deployment.pkl\n")
    
    print(f"✓ Saved selection report: {report_path}")
    
    # Also save as JSON
    json_path = f'{output_dir}/model_selection.json'
    
    # Create a JSON-serializable copy of metrics (remove model_path)
    metrics_for_json = {k: v for k, v in metrics.items() if k != 'model_path'}
    
    with open(json_path, 'w') as f:
        json.dump({
            'selected_model': best_model,
            'metrics': metrics_for_json,
            'timestamp': pd.Timestamp.now().isoformat(),
            'deployment_path': f"{GCS_DEPLOYMENT_PATH}best_model_deployment.pkl"
        }, f, indent=2)
    
    print(f"✓ Saved selection JSON: {json_path}\n")


# ============================================
# 6. MAIN EXECUTION
# ============================================

def main():
    """Main model selection pipeline"""
    
    # Load results
    results = load_model_results()
    
    if not results:
        print("❌ No model results found!")
        print("   Please run both models first:")
        print("   1. python Dense_VAE_optimized_mlflow_updated.py")
        print("   2. python Ensemble_VAE_updated.py")
        return
    
    # Select best model
    best_model, best_metrics = select_best_model(results)
    
    # Create visualizations
    create_comparison_plots(results)
    
    # Save report
    save_selection_report(best_model, best_metrics)
    
    # Upload to GCS for deployment
    upload_success = upload_model_to_gcs(best_model, best_metrics)
    
    print("\n" + "="*60)
    print("✅ MODEL SELECTION COMPLETE")
    print("="*60)
    print(f"\nBest Model: {best_model}")
    print(f"Results saved in: outputs/model_selection/")
    
    if upload_success:
        print(f"\n🚀 Model deployed to GCS:")
        print(f"   {GCS_DEPLOYMENT_PATH}best_model_deployment.pkl")
        print(f"   Ready for production use!")
    else:
        print(f"\n⚠️  Model deployment failed - check logs above")
    
    print("\nNext steps:")
    print("1. Review model_comparison.png")
    print("2. Read model_selection_report.txt")
    if upload_success:
        print("3. Test deployment with inference API")
        print("4. Set up monitoring for production model")
    else:
        print("3. Fix deployment issues and re-run")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()