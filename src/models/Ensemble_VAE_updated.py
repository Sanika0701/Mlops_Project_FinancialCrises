"""
Ensemble VAE for Macroeconomic Scenario Generation
GCS INTEGRATION VERSION
Reads from: gs://mlops-financial-stress-data/data/features/macro_features_clean.csv
Writes to: gs://mlops-financial-stress-data/models/vae/outputs/output_Ensemble_VAE/
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance, ks_2samp
import os
import sys
import warnings
import time
from datetime import datetime
import io
warnings.filterwarnings('ignore')

# GCS imports
import gcsfs

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# MLflow imports
import mlflow
import mlflow.pytorch
from src.utils.mlflow_config import MLflowConfig


# ============================================
# GCS SETUP
# ============================================

import gcsfs
import os

# Initialize GCS filesystem - let it auto-detect credentials
fs = gcsfs.GCSFileSystem(token='google_default')

# GCS paths
GCS_BUCKET = 'mlops-financial-stress-data'
GCS_DATA_PATH = f'gs://{GCS_BUCKET}/data/features/macro_features_clean.csv'
GCS_OUTPUT_BASE = f'gs://{GCS_BUCKET}/models/vae/outputs/output_Ensemble_VAE'


def cleanup_gcs_outputs(gcs_path):
    """Delete old outputs from GCS"""
    print(f"🗑️  Checking GCS: {gcs_path}")
    path_without_prefix = gcs_path.replace('gs://', '')
    
    try:
        if fs.exists(path_without_prefix):
            print(f"Deleting existing outputs...")
            fs.rm(path_without_prefix, recursive=True)
            print(f"✓ Cleaned up\n")
        else:
            print(f"Path doesn't exist, will create\n")
    except Exception as e:
        print(f"⚠ Could not delete: {e}\n")
    
    try:
        fs.makedirs(path_without_prefix, exist_ok=True)
        print(f"✓ Created GCS directory\n")
    except Exception as e:
        print(f"Directory: {e}\n")


# ============================================
# DATA LOADING (GCS)
# ============================================

def load_data_from_gcs(gcs_path, test_size=0.2, val_size=0.1, random_seed=42):
    """Load data from GCS"""
    
    print("="*60)
    print("LOADING DATA FROM GCS")
    print("="*60)
    print(f"GCS Path: {gcs_path}\n")
    
    # Set seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Read from GCS
    print("Reading CSV from GCS...")
    with fs.open(gcs_path.replace('gs://', ''), 'r') as f:
        df = pd.read_csv(f)
    print(f"✓ Loaded {len(df)} rows\n")
    
    if 'Date' in df.columns:
        df = df.drop('Date', axis=1)
    
    # Handle categorical
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if len(df[col].unique()) <= 10:
            df[col] = pd.Categorical(df[col]).codes
    
    # Numeric only
    df = df.select_dtypes(include=[np.number])
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    features = df.columns.tolist()
    print(f"✓ Dataset: {len(df)} rows, {len(features)} features")
    
    # Split
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_seed)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_seed)
    
    print(f"✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}\n")
    
    return train.values, val.values, test.values, features


def normalize_data(train, val, test):
    """Normalize data"""
    print("Normalizing...")
    
    for i in range(train.shape[1]):
        mean, std = train[:, i].mean(), train[:, i].std()
        lower, upper = mean - 5*std, mean + 5*std
        train[:, i] = np.clip(train[:, i], lower, upper)
        val[:, i] = np.clip(val[:, i], lower, upper)
        test[:, i] = np.clip(test[:, i], lower, upper)
    
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train)
    val_s = scaler.transform(val)
    test_s = scaler.transform(test)
    
    print("✓ Normalized\n")
    return train_s, val_s, test_s, scaler


# ============================================
# VAE ARCHITECTURE
# ============================================

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        curr = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(curr, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            curr = h
        self.layers = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(curr, latent_dim)
        self.fc_logvar = nn.Linear(curr, latent_dim)
    
    def forward(self, x):
        h = self.layers(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        curr = latent_dim
        for h in reversed(hidden_dims):
            layers.extend([
                nn.Linear(curr, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            curr = h
        layers.append(nn.Linear(curr, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.model(z)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=0.5):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


# ============================================
# ENSEMBLE VAE WRAPPER
# ============================================

class EnsembleVAEWrapper(mlflow.pyfunc.PythonModel):
    """Custom MLflow model wrapper for Ensemble VAE"""
    
    def load_context(self, context):
        import torch
        
        ensemble_path = context.artifacts["ensemble_package"]
        # Handle both local and GCS paths
        if ensemble_path.startswith('gs://'):
            with fs.open(ensemble_path.replace('gs://', ''), 'rb') as f:
                ensemble_data = torch.load(f, map_location='cpu')
        else:
            ensemble_data = torch.load(ensemble_path, map_location='cpu')
        
        self.models = []
        self.config = ensemble_data['config']
        
        for model_state in ensemble_data['models']:
            model = VAE(
                input_dim=ensemble_data['input_dim'],
                hidden_dims=self.config['hidden_dims'],
                latent_dim=self.config['latent_dim']
            )
            model.load_state_dict(model_state)
            model.eval()
            self.models.append(model)
        
        self.scaler = ensemble_data['scaler']
        self.features = ensemble_data['features']
        self.n_models = len(self.models)
    
    def predict(self, context, model_input):
        import torch
        import pandas as pd
        
        n_scenarios = model_input.get('n_scenarios', 100)
        severity_dist = model_input.get('severity_dist', {
            'baseline': 10, 'adverse': 20, 'severe': 50, 'extreme': 20
        })
        sigma_values = model_input.get('sigma_values', {
            'baseline': 0.5, 'adverse': 1.5, 'severe': 2.5, 'extreme': 3.5
        })
        
        all_scenarios = []
        all_severities = []
        per_model_dist = {k: v // self.n_models for k, v in severity_dist.items()}
        device = torch.device('cpu')
        
        for model in self.models:
            model.eval()
            latent_dim = model.decoder.model[0].in_features
            
            with torch.no_grad():
                for severity, count in per_model_dist.items():
                    if count > 0:
                        sigma = sigma_values[severity]
                        z = torch.randn(count, latent_dim, device=device) * sigma
                        scenarios = model.decoder(z).cpu().numpy()
                        all_scenarios.append(scenarios)
                        all_severities.extend([severity.capitalize()] * count)
        
        all_scenarios = np.vstack(all_scenarios)
        scenarios_denorm = self.scaler.inverse_transform(all_scenarios)
        
        df = pd.DataFrame(scenarios_denorm, columns=self.features)
        df.insert(0, 'Scenario', [f'Scenario_{i+1}' for i in range(len(df))])
        df.insert(1, 'Severity', all_severities)
        
        return df


# ============================================
# TRAIN SINGLE VAE
# ============================================

def train_single_vae(train_loader, val_loader, input_dim, hidden_dims, 
                    latent_dim, epochs, beta, device, model_id, random_seed, gcs_output_dir):
    """Train one VAE with nested MLflow run"""
    
    print(f"\n{'='*60}")
    print(f"TRAINING VAE #{model_id} (seed={random_seed})")
    print(f"{'='*60}")
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    with mlflow.start_run(run_name=f"VAE_Model_{model_id}", nested=True) as nested_run:
        
        mlflow.log_params({
            'model_id': model_id,
            'random_seed': random_seed,
            'hidden_dims': str(hidden_dims),
            'latent_dim': latent_dim,
            'beta': beta
        })
        
        model_start_time = time.time()
        
        model = VAE(input_dim, hidden_dims, latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        
        best_val = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_losses = []
            for batch in train_loader:
                if isinstance(batch, list):
                    batch = batch[0]
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss = vae_loss(recon, batch, mu, logvar, beta)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item() / len(batch))
            
            # Validate
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, list):
                        batch = batch[0]
                    batch = batch.to(device)
                    recon, mu, logvar = model(batch)
                    loss = vae_loss(recon, batch, mu, logvar, beta)
                    val_losses.append(loss.item() / len(batch))
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
                mlflow.log_metrics({
                    f'model_{model_id}_train_loss': train_loss,
                    f'model_{model_id}_val_loss': val_loss
                }, step=epoch+1)
            
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= 30:
                print(f"  Early stopping at epoch {epoch+1}")
                mlflow.log_metric(f'model_{model_id}_epochs_completed', epoch+1)
                break
        
        model.load_state_dict(best_state)
        
        model_training_time = time.time() - model_start_time
        mlflow.log_metrics({
            f'model_{model_id}_best_val_loss': best_val,
            f'model_{model_id}_training_time_seconds': model_training_time
        })
        
        # Save model to GCS
        gcs_path_prefix = gcs_output_dir.replace('gs://', '')
        model_data = {
            'model': model.state_dict(),
            'seed': random_seed,
            'model_id': model_id,
            'best_val_loss': best_val
        }
        
        with fs.open(f'{gcs_path_prefix}/vae_model_{model_id}.pth', 'wb') as f:
            torch.save(model_data, f)
        
        print(f"✓ VAE #{model_id} trained! Best Val Loss: {best_val:.4f}\n")
    
    return model


# ============================================
# ENSEMBLE GENERATION
# ============================================

def generate_ensemble_scenarios(models, scaler, features, scenarios_per_model=20, 
                                n_baseline=10, n_adverse=20, n_severe=50, n_extreme=20, device='cpu'):
    """Generate scenarios from ensemble"""
    
    print("\n" + "="*60)
    print("GENERATING ENSEMBLE SCENARIOS")
    print("="*60)
    print(f"Models: {len(models)}, Scenarios per model: {scenarios_per_model}\n")
    
    all_scenarios = []
    all_severities = []
    
    per_model_dist = {
        'baseline': n_baseline // len(models),
        'adverse': n_adverse // len(models),
        'severe': n_severe // len(models),
        'extreme': n_extreme // len(models)
    }
    
    for idx, model in enumerate(models):
        model.eval()
        latent_dim = model.decoder.model[0].in_features
        
        with torch.no_grad():
            if per_model_dist['baseline'] > 0:
                z = torch.randn(per_model_dist['baseline'], latent_dim, device=device) * 0.5
                s = model.decoder(z).cpu().numpy()
                all_scenarios.append(s)
                all_severities.extend(['Baseline'] * per_model_dist['baseline'])
            
            if per_model_dist['adverse'] > 0:
                z = torch.randn(per_model_dist['adverse'], latent_dim, device=device) * 1.5
                s = model.decoder(z).cpu().numpy()
                all_scenarios.append(s)
                all_severities.extend(['Adverse'] * per_model_dist['adverse'])
            
            if per_model_dist['severe'] > 0:
                z = torch.randn(per_model_dist['severe'], latent_dim, device=device) * 2.5
                s = model.decoder(z).cpu().numpy()
                all_scenarios.append(s)
                all_severities.extend(['Severe'] * per_model_dist['severe'])
            
            if per_model_dist['extreme'] > 0:
                z = torch.randn(per_model_dist['extreme'], latent_dim, device=device) * 3.5
                s = model.decoder(z).cpu().numpy()
                all_scenarios.append(s)
                all_severities.extend(['Extreme'] * per_model_dist['extreme'])
        
        print(f"✓ Model {idx+1}/{len(models)}")
    
    all_scenarios = np.vstack(all_scenarios)
    scenarios_denorm = scaler.inverse_transform(all_scenarios)
    
    df = pd.DataFrame(scenarios_denorm, columns=features)
    df.insert(0, 'Scenario', [f'Scenario_{i+1}' for i in range(len(df))])
    df.insert(1, 'Severity', all_severities)
    
    print(f"\n✓ Total scenarios: {len(df)}")
    print(f"  Baseline: {(df['Severity']=='Baseline').sum()}")
    print(f"  Adverse: {(df['Severity']=='Adverse').sum()}")
    print(f"  Severe: {(df['Severity']=='Severe').sum()}")
    print(f"  Extreme: {(df['Severity']=='Extreme').sum()}\n")
    
    mlflow.log_metrics({
        'total_scenarios': len(df),
        'n_baseline': int((df['Severity']=='Baseline').sum()),
        'n_adverse': int((df['Severity']=='Adverse').sum()),
        'n_severe': int((df['Severity']=='Severe').sum()),
        'n_extreme': int((df['Severity']=='Extreme').sum())
    })
    
    return df


def classify_crisis_type(scenarios_df, feature_names, reference_data=None):
    """Classify crisis types"""
    print("="*60)
    print("CLASSIFYING CRISIS TYPES")
    print("="*60 + "\n")
    
    if reference_data is not None:
        ref_df = pd.DataFrame(reference_data, columns=feature_names)
    else:
        ref_df = scenarios_df
    
    crisis_types = []
    for idx, row in scenarios_df.iterrows():
        gdp = row.get('GDP', 18000)
        vix = row.get('VIX', 20)
        unemployment = row.get('Unemployment_Rate', 5)
        
        if vix > 35:
            crisis_type = "Market Crash"
        elif unemployment > 9:
            crisis_type = "Unemployment Crisis"
        elif gdp < 14000:
            crisis_type = "Economic Recession"
        else:
            crisis_type = "Normal Economy"
        
        crisis_types.append(crisis_type)
    
    scenarios_df.insert(2, 'Crisis_Type', crisis_types)
    
    print("Crisis Type Distribution:")
    for crisis_type in scenarios_df['Crisis_Type'].unique():
        count = (scenarios_df['Crisis_Type'] == crisis_type).sum()
        pct = 100 * count / len(scenarios_df)
        print(f"  {crisis_type:25s}: {count:3d} ({pct:5.1f}%)")
    print("="*60 + "\n")
    
    return scenarios_df


# ============================================
# VALIDATION
# ============================================

def validate(real_data, gen_data, features, gcs_output_dir):
    """Validate and save to GCS"""
    
    print("="*60)
    print("VALIDATION")
    print("="*60 + "\n")
    
    if isinstance(gen_data, pd.DataFrame):
        gen_data = gen_data.drop(['Scenario', 'Severity', 'Crisis_Type'], 
                                 axis=1, errors='ignore').values
    
    # KS tests
    ks_results = []
    for i in range(len(features)):
        stat, pval = ks_2samp(real_data[:, i], gen_data[:, i])
        ks_results.append({'feature': features[i], 'p_value': pval, 'passed': pval > 0.05})
    
    ks_df = pd.DataFrame(ks_results)
    pass_rate = 100 * ks_df['passed'].sum() / len(ks_df)
    
    # Wasserstein
    w_dists = [wasserstein_distance(real_data[:, i], gen_data[:, i]) 
               for i in range(min(20, real_data.shape[1]))]
    w_mean = np.mean(w_dists)
    
    # Correlation
    real_corr = np.corrcoef(real_data[:, :20].T)
    gen_corr = np.corrcoef(gen_data[:, :20].T)
    corr_mae = np.mean(np.abs(real_corr - gen_corr))
    
    print(f"1. KS Pass Rate: {pass_rate:.1f}%")
    print(f"2. Correlation MAE: {corr_mae:.4f}")
    print(f"3. Wasserstein: {w_mean:.2f}\n")
    
    mlflow.log_metrics({
        'ks_pass_rate': pass_rate,
        'correlation_mae': corr_mae,
        'wasserstein_distance': w_mean
    })
    
    # Save to GCS
    gcs_path_prefix = gcs_output_dir.replace('gs://', '')
    
    report = f"""{'='*60}
ENSEMBLE VAE VALIDATION
{'='*60}

KS Pass Rate: {pass_rate:.1f}%
Correlation MAE: {corr_mae:.4f}
Wasserstein: {w_mean:.2f}
"""
    
    with fs.open(f"{gcs_path_prefix}/ensemble_validation.txt", 'w') as f:
        f.write(report)
    print(f"✓ Saved: {gcs_output_dir}/ensemble_validation.txt")
    
    with fs.open(f"{gcs_path_prefix}/ensemble_ks_results.csv", 'w') as f:
        ks_df.to_csv(f, index=False)
    print(f"✓ Saved: {gcs_output_dir}/ensemble_ks_results.csv\n")
    
    return {'pass_rate': pass_rate, 'wasserstein': w_mean, 'correlation': corr_mae}


# ============================================
# MAIN ENSEMBLE PIPELINE
# ============================================

def main(gcs_data_path, gcs_output_dir, n_models=5, hidden_dims=[256, 128, 64], latent_dim=32,
         batch_size=128, epochs=300, beta=0.5, base_seed=42,
         n_baseline=10, n_adverse=20, n_severe=50, n_extreme=20,
         mlflow_tracking_uri=None, experiment_name="Financial_Stress_Test_Scenarios"):
    """Train ensemble with GCS integration"""
    
    cleanup_gcs_outputs(gcs_output_dir)
    
    print("="*60)
    print("ENSEMBLE VAE - GCS VERSION")
    print("="*60)
    print(f"Training {n_models} VAEs")
    print(f"Input: {gcs_data_path}")
    print(f"Output: {gcs_output_dir}")
    print("="*60 + "\n")
    
    # Initialize MLflow
    mlflow_config = MLflowConfig(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name
    )
    
    run_name = f"Ensemble_VAE_GCS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as parent_run:
        print(f"✓ MLflow Parent Run: {parent_run.info.run_id}\n")
        
        mlflow_config.set_tags({
            'model_type': 'Ensemble_VAE_GCS',
            'n_models': n_models,
            'stage': 'development',
            'data_source': 'GCS',
            'output_location': 'GCS'
        })
        
        params = {
            'n_models': n_models,
            'hidden_dims': str(hidden_dims),
            'latent_dim': latent_dim,
            'batch_size': batch_size,
            'max_epochs': epochs,
            'beta': beta,
            'base_seed': base_seed
        }
        mlflow_config.log_params(params)
        
        start_time = time.time()
        
        # Load from GCS
        train, val, test, features = load_data_from_gcs(gcs_data_path, random_seed=base_seed)
        train_s, val_s, test_s, scaler = normalize_data(train, val, test)
        
        mlflow.log_metrics({
            'n_train_samples': len(train),
            'n_val_samples': len(val),
            'n_test_samples': len(test),
            'n_features': len(features)
        })
        
        # Create loaders
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_s)), 
                                  batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_s)), 
                                batch_size=batch_size, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = train_s.shape[1]
        
        # Train models
        print("="*60)
        print(f"TRAINING {n_models} VAE MODELS")
        print("="*60)
        
        models = []
        seeds = [base_seed + i * 111 for i in range(n_models)]
        
        for i, seed in enumerate(seeds):
            model = train_single_vae(
                train_loader, val_loader, input_dim, hidden_dims, 
                latent_dim, epochs, beta, device, i+1, seed, gcs_output_dir
            )
            models.append(model)
        
        print(f"\n✅ ALL {n_models} MODELS TRAINED!\n")
        
        # Generate scenarios
        scenarios_per_model = (n_baseline + n_adverse + n_severe + n_extreme) // n_models
        scenarios = generate_ensemble_scenarios(
            models, scaler, features, scenarios_per_model,
            n_baseline, n_adverse, n_severe, n_extreme, device
        )
        
        # Classify
        scenarios = classify_crisis_type(scenarios, features, reference_data=train)
        
        # Validate
        results = validate(test, scenarios, features, gcs_output_dir)
        
        training_time = time.time() - start_time
        mlflow.log_metric('training_time_seconds', training_time)
        
        # Save to GCS
        print("="*60)
        print("SAVING TO GCS")
        print("="*60 + "\n")
        
        gcs_path_prefix = gcs_output_dir.replace('gs://', '')
        
        # Save scenarios
        with fs.open(f"{gcs_path_prefix}/ensemble_vae_scenarios.csv", 'w') as f:
            scenarios.to_csv(f, index=False)
        print(f"✓ Saved: {gcs_output_dir}/ensemble_vae_scenarios.csv")
        
        # Save scaler
        import pickle
        with fs.open(f"{gcs_path_prefix}/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Saved: {gcs_output_dir}/scaler.pkl")
        
        # Create ensemble package
        ensemble_package = {
            'models': [model.state_dict() for model in models],
            'scaler': scaler,
            'features': features,
            'input_dim': input_dim,
            'config': {
                'n_models': n_models,
                'hidden_dims': hidden_dims,
                'latent_dim': latent_dim,
                'beta': beta,
                'seeds': seeds
            }
        }
        
        with fs.open(f"{gcs_path_prefix}/ensemble_vae_complete.pth", 'wb') as f:
            torch.save(ensemble_package, f)
        print(f"✓ Saved: {gcs_output_dir}/ensemble_vae_complete.pth")
        
        # Log custom MLflow model
        print("\nLogging MLflow custom model...")
        
        # Download ensemble package temporarily for MLflow
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            with fs.open(f"{gcs_path_prefix}/ensemble_vae_complete.pth", 'rb') as f:
                tmp.write(f.read())
            temp_model_path = tmp.name
        
        artifacts = {"ensemble_package": temp_model_path}
        
        conda_env = {
            'channels': ['defaults', 'conda-forge'],
            'dependencies': [
                'python=3.9',
                'pip',
                {'pip': ['mlflow', 'torch', 'numpy', 'pandas', 'scikit-learn', 'gcsfs']}
            ],
            'name': 'ensemble_vae_env'
        }
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=EnsembleVAEWrapper(),
            artifacts=artifacts,
            conda_env=conda_env
        )
        
        # Clean up temp file
        os.remove(temp_model_path)
        
        print("✅ Logged ensemble as MLflow model!\n")
        
        # Results
        print("="*60)
        print("ENSEMBLE VAE RESULTS")
        print("="*60)
        print(f"KS Pass Rate: {results['pass_rate']:.1f}%")
        print(f"Correlation MAE: {results['correlation']:.4f}")
        print(f"Wasserstein: {results['wasserstein']:.1f}")
        print(f"Training Time: {training_time:.1f}s")
        print(f"MLflow Run: {parent_run.info.run_id}")
        print(f"GCS Output: {gcs_output_dir}")
        print("="*60 + "\n")
        
        return models, scenarios, results, parent_run.info.run_id


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    
    from dotenv import load_dotenv
    load_dotenv()
    
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'Financial_Stress_Test_Scenarios')
    
    print("="*60)
    print("ENSEMBLE VAE - GCS INTEGRATION")
    print("="*60)
    print(f"Reading from: {GCS_DATA_PATH}")
    print(f"Writing to: {GCS_OUTPUT_BASE}")
    print(f"MLflow: {mlflow_uri}")
    print("="*60 + "\n")
    
    models, scenarios, results, run_id = main(
        gcs_data_path=GCS_DATA_PATH,
        gcs_output_dir=GCS_OUTPUT_BASE,
        n_models=5,
        hidden_dims=[256, 128, 64],
        latent_dim=32,
        batch_size=128,
        epochs=300,
        beta=0.5,
        base_seed=42,
        n_baseline=10,
        n_adverse=20,
        n_severe=50,
        n_extreme=20,
        mlflow_tracking_uri=mlflow_uri,
        experiment_name=experiment_name
    )
    
    print("="*60)
    print("🎉 ENSEMBLE VAE COMPLETE!")
    print("="*60)
    print(f"\nResults:")
    print(f"  KS Pass Rate: {results['pass_rate']:.1f}%")
    print(f"  Correlation MAE: {results['correlation']:.4f}")
    print(f"\nAll outputs in GCS:")
    print(f"  {GCS_OUTPUT_BASE}/")
    print(f"\nMLflow Run: {run_id}")
    print(f"View at: {mlflow_uri}")
    print("="*60)