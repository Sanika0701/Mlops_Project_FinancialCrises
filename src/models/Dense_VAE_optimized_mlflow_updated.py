"""
Dense VAE for Macroeconomic Scenario Generation
FINAL VERSION - GCS Integration
Reads from: gs://mlops-financial-stress-data/data/features/macro_features_clean.csv
Writes to: gs://mlops-financial-stress-data/models/vae/outputs/output_Dense_VAE_optimized/
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# MLflow imports
import mlflow
import mlflow.pytorch
from src.utils.mlflow_config import MLflowConfig


# ============================================
# 0. GCS SETUP
# ============================================
'''
# Initialize GCS filesystem
fs = gcsfs.GCSFileSystem(project='ninth-iris-422916-f2', token='google_default')  # Replace with your project ID

# GCS paths
GCS_BUCKET = 'mlops-financial-stress-data'
GCS_DATA_PATH = f'gs://{GCS_BUCKET}/data/features/macro_features_clean.csv'
GCS_OUTPUT_BASE = f'gs://{GCS_BUCKET}/models/vae/outputs/output_Dense_VAE_optimized'
'''

import gcsfs
import os

# Initialize GCS filesystem - let it auto-detect credentials
fs = gcsfs.GCSFileSystem(token='google_default')

# GCS paths
GCS_BUCKET = 'mlops-financial-stress-data'
GCS_DATA_PATH = f'gs://{GCS_BUCKET}/data/features/macro_features_clean.csv'
GCS_OUTPUT_BASE = f'gs://{GCS_BUCKET}/models/vae/outputs/output_Dense_VAE_optimized'

def cleanup_gcs_outputs(gcs_path):
    """Delete old outputs from GCS"""
    print(f"🗑️  Checking GCS path: {gcs_path}")
    
    # Remove gs:// prefix for gcsfs operations
    path_without_prefix = gcs_path.replace('gs://', '')
    
    try:
        if fs.exists(path_without_prefix):
            print(f"Deleting existing outputs at {gcs_path}...")
            fs.rm(path_without_prefix, recursive=True)
            print(f"✓ Cleaned up {gcs_path}\n")
        else:
            print(f"Path doesn't exist yet, will create new\n")
    except Exception as e:
        print(f"⚠ Could not delete: {e}")
        print(f"⚠ Will overwrite files instead\n")
    
    # Create directory
    try:
        fs.makedirs(path_without_prefix, exist_ok=True)
        print(f"✓ Created GCS directory: {gcs_path}\n")
    except Exception as e:
        print(f"Directory creation: {e}\n")


# ============================================
# 1. DATA LOADING (GCS)
# ============================================

def load_data_from_gcs(gcs_path, test_size=0.2, val_size=0.1, random_seed=42):
    """Load data from GCS"""
    
    print("="*60)
    print("LOADING DATA FROM GCS")
    print("="*60)
    print(f"GCS Path: {gcs_path}")
    print(f"Random Seed: {random_seed}\n")
    
    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    # Read from GCS
    print("Reading CSV from GCS...")
    with fs.open(gcs_path.replace('gs://', ''), 'r') as f:
        df = pd.read_csv(f)
    print(f"✓ Loaded {len(df)} rows from GCS\n")
    
    # Store date for crisis analysis
    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'])
        df_with_dates = df.copy()
        df = df.drop('Date', axis=1)
    else:
        dates = None
        df_with_dates = df.copy()
    
    # Handle categorical
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"Converting categorical columns: {categorical_cols}")
        for col in categorical_cols:
            if len(df[col].unique()) <= 10:
                df[col] = pd.Categorical(df[col]).codes
                print(f"  ✓ {col} encoded")
    
    # Numeric only
    df = df.select_dtypes(include=[np.number])
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    feature_names = df.columns.tolist()
    print(f"\n✓ Dataset: {len(df)} rows, {len(feature_names)} features")
    
    # ANALYZE CRISIS PERIODS
    if dates is not None:
        analyze_crisis_periods(df_with_dates, dates, feature_names)
    
    # Split
    train_val, test = train_test_split(df, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
    
    print(f"✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    return train.values, val.values, test.values, feature_names


def analyze_crisis_periods(df, dates, feature_names):
    """Analyze crisis periods in dataset"""
    print("\n" + "="*60)
    print("CRISIS PERIOD ANALYSIS")
    print("="*60)
    
    df['Date'] = dates
    df['Year'] = dates.dt.year
    
    if 'VIX' in feature_names and 'Unemployment_Rate' in feature_names:
        vix_crisis = df['VIX'].quantile(0.90)
        unemp_crisis = df['Unemployment_Rate'].quantile(0.85)
        
        df['is_high_stress'] = (
            (df['VIX'] > vix_crisis) | 
            (df['Unemployment_Rate'] > unemp_crisis)
        )
        
        crisis_years = df[df['is_high_stress']].groupby('Year').size()
        crisis_years = crisis_years[crisis_years > 5]
        
        print(f"\nIdentified Crisis Periods:")
        print("-" * 40)
        
        if len(crisis_years) > 0:
            for year, count in crisis_years.items():
                year_data = df[df['Year'] == year]
                avg_vix = year_data['VIX'].mean()
                avg_unemp = year_data['Unemployment_Rate'].mean()
                
                print(f"Year {year}: {count} stress months")
                print(f"  Avg VIX: {avg_vix:.1f}")
                print(f"  Avg Unemployment: {avg_unemp:.1f}%")
                
                if year in [1990, 1991]:
                    print(f"  → Early 1990s Recession")
                elif year in [1997, 1998]:
                    print(f"  → Asian Financial Crisis")
                elif year in [2000, 2001, 2002]:
                    print(f"  → Dot-com Bubble Burst")
                elif year in [2008, 2009]:
                    print(f"  → Financial Crisis")
                elif year in [2020]:
                    print(f"  → COVID-19 Pandemic")
                print()
        
        total_crisis_months = df['is_high_stress'].sum()
        crisis_pct = 100 * total_crisis_months / len(df)
        
        print("-" * 40)
        print(f"Total Crisis Months: {total_crisis_months} ({crisis_pct:.1f}%)")
        print(f"Normal Months: {len(df) - total_crisis_months} ({100-crisis_pct:.1f}%)")
    
    print("="*60 + "\n")


def normalize_data(train, val, test):
    """Normalize data"""
    print("Normalizing...")
    
    # Clip outliers
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
# 2. VAE MODEL (Same as before)
# ============================================

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        curr = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(curr, h),
                nn.LayerNorm(h),             
                nn.SiLU(),                   
                nn.Dropout(0.1)              
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
                nn.LayerNorm(h),           
                nn.SiLU(),                 
                nn.Dropout(0.1)
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


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ============================================
# 3. TRAINING (Same as before)
# ============================================

def train_vae(train_loader, val_loader, input_dim, hidden_dims, latent_dim, 
              epochs, beta, device):
    """Train VAE with MLflow logging"""
    
    print("="*60)
    print("TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Architecture: {hidden_dims} → {latent_dim}")
    print(f"Beta: {beta}\n")

    warmup_epochs = 30

    model = VAE(input_dim, hidden_dims, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    best_val = float('inf')
    patience_counter = 0
    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()
        curr_beta = min(beta, beta * epoch / warmup_epochs)

        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, logvar, curr_beta)
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
                loss, _, _ = vae_loss(recon, batch, mu, logvar, beta)
                val_losses.append(loss.item() / len(batch))

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch+1)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= 30:
            print(f"\nEarly stopping at epoch {epoch+1}")
            mlflow.log_metric('epochs_completed', epoch+1)
            break

    model.load_state_dict(best_state)
    
    avg_epoch_time = np.mean(epoch_times)
    mlflow.log_metrics({
        'best_val_loss': best_val,
        'avg_epoch_time_seconds': avg_epoch_time
    })
    
    print(f"\n✓ Training complete! Best Val Loss: {best_val:.4f}\n")
    return model


# ============================================
# 4. GENERATION
# ============================================

def generate_scenarios(model, scaler, features, n_baseline=10, n_adverse=20, 
                      n_severe=50, n_extreme=20, device='cpu'):
    """Generate crisis-focused scenarios"""
    
    print("="*60)
    print("GENERATING SCENARIOS")
    print("="*60)
    
    model.eval()
    latent_dim = model.decoder.model[0].in_features
    
    all_scenarios = []
    labels = []
    
    with torch.no_grad():
        # Baseline
        z = torch.randn(n_baseline, latent_dim, device=device) * 0.5
        s = model.decoder(z).cpu().numpy()
        all_scenarios.append(s)
        labels.extend(['Baseline'] * n_baseline)
        
        # Adverse
        z = torch.randn(n_adverse, latent_dim, device=device) * 1.5
        s = model.decoder(z).cpu().numpy()
        all_scenarios.append(s)
        labels.extend(['Adverse'] * n_adverse)
        
        # Severe
        z = torch.randn(n_severe, latent_dim, device=device) * 2.5
        s = model.decoder(z).cpu().numpy()
        all_scenarios.append(s)
        labels.extend(['Severe'] * n_severe)
        
        # Extreme
        z = torch.randn(n_extreme, latent_dim, device=device) * 3.5
        s = model.decoder(z).cpu().numpy()
        all_scenarios.append(s)
        labels.extend(['Extreme'] * n_extreme)
    
    all_scenarios = np.vstack(all_scenarios)
    scenarios_denorm = scaler.inverse_transform(all_scenarios)
    
    df = pd.DataFrame(scenarios_denorm, columns=features)
    df.insert(0, 'Scenario', [f'Scenario_{i+1}' for i in range(len(df))])
    df.insert(1, 'Severity', labels)
    
    print(f"\n✓ Generated {len(df)} scenarios")
    print(f"  - {n_baseline} Baseline, {n_adverse} Adverse, {n_severe} Severe, {n_extreme} Extreme")
    print(f"✓ Crisis scenarios: {n_adverse + n_severe + n_extreme} ({100*(n_adverse + n_severe + n_extreme)//100}%)")
    print("="*60 + "\n")
    
    mlflow.log_metrics({
        'n_baseline_scenarios': n_baseline,
        'n_adverse_scenarios': n_adverse,
        'n_severe_scenarios': n_severe,
        'n_extreme_scenarios': n_extreme,
        'total_scenarios': len(df),
        'crisis_scenario_percentage': (n_adverse + n_severe + n_extreme)
    })
    
    return df


def classify_stress_level(scenarios_df, feature_names):
    """Classify stress levels"""
    print("="*60)
    print("CLASSIFYING STRESS LEVELS")
    print("="*60 + "\n")
    
    stress_levels = []
    stress_scores = []
    
    for idx, row in scenarios_df.iterrows():
        score = 0
        
        if 'GDP' in feature_names:
            gdp = row['GDP']
            if gdp < 12000: score += 4
            elif gdp < 14000: score += 3
            elif gdp < 16000: score += 2
            elif gdp < 18000: score += 1
        
        if 'VIX' in feature_names:
            vix = row['VIX']
            if vix > 45: score += 4
            elif vix > 30: score += 3
            elif vix > 20: score += 2
            elif vix > 15: score += 1
        
        if 'Unemployment_Rate' in feature_names:
            unemp = row['Unemployment_Rate']
            if unemp > 10: score += 4
            elif unemp > 8: score += 3
            elif unemp > 6: score += 2
            elif unemp > 5: score += 1
        
        if score >= 12: stress_level = 'Extreme Crisis'
        elif score >= 8: stress_level = 'Severe Crisis'
        elif score >= 5: stress_level = 'Moderate Stress'
        elif score >= 2: stress_level = 'Mild Stress'
        else: stress_level = 'Normal'
        
        stress_levels.append(stress_level)
        stress_scores.append(score)
    
    scenarios_df.insert(2, 'Stress_Level', stress_levels)
    scenarios_df.insert(3, 'Stress_Score', stress_scores)
    
    print("Stress Level Distribution:")
    for level in ['Normal', 'Mild Stress', 'Moderate Stress', 'Severe Crisis', 'Extreme Crisis']:
        count = (scenarios_df['Stress_Level'] == level).sum()
        pct = 100 * count / len(scenarios_df)
        print(f"  {level:20s}: {count:3d} ({pct:5.1f}%)")
    print("="*60 + "\n")
    
    return scenarios_df


# ============================================
# 5. VALIDATION
# ============================================

def validate(real_data, gen_data, features, gcs_output_dir):
    """Validate and save to GCS"""
    
    print("="*60)
    print("VALIDATION")
    print("="*60 + "\n")
    
    if isinstance(gen_data, pd.DataFrame):
        gen_data = gen_data.drop(['Scenario', 'Severity', 'Stress_Level', 'Stress_Score'], 
                                 axis=1, errors='ignore').values
    
    # KS tests
    ks_results = []
    for i in range(len(features)):
        stat, pval = ks_2samp(real_data[:, i], gen_data[:, i])
        ks_results.append({
            'feature': features[i],
            'statistic': stat,
            'p_value': pval,
            'passed': pval > 0.05
        })
    
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
    
    print(f"1. KS Test Pass Rate: {pass_rate:.1f}%")
    print(f"2. Correlation MAE: {corr_mae:.4f}")
    print(f"3. Wasserstein Distance: {w_mean:.2f}\n")
    
    mlflow.log_metrics({
        'ks_pass_rate': pass_rate,
        'correlation_mae': corr_mae,
        'wasserstein_distance': w_mean,
        'n_features_passed_ks': int(ks_df['passed'].sum()),
        'n_features_total': len(ks_df)
    })
    
    # Save to GCS
    gcs_path_prefix = gcs_output_dir.replace('gs://', '')
    
    # Save validation report
    report_content = f"""{'='*60}
DENSE VAE VALIDATION REPORT
{'='*60}

KS Test Pass Rate: {pass_rate:.1f}%
Correlation MAE: {corr_mae:.4f}
Wasserstein Distance: {w_mean:.2f}

Failed Features (p < 0.05):
"""
    failed = ks_df[~ks_df['passed']]
    for _, row in failed.iterrows():
        report_content += f"  - {row['feature']}: p={row['p_value']:.4f}\n"
    
    with fs.open(f"{gcs_path_prefix}/validation_report.txt", 'w') as f:
        f.write(report_content)
    print(f"✓ Saved: {gcs_output_dir}/validation_report.txt")
    
    # Save KS results
    with fs.open(f"{gcs_path_prefix}/ks_test_results.csv", 'w') as f:
        ks_df.to_csv(f, index=False)
    print(f"✓ Saved: {gcs_output_dir}/ks_test_results.csv\n")
    
    return {'pass_rate': pass_rate, 'wasserstein': w_mean, 'correlation': corr_mae}


# ============================================
# 6. VISUALIZATION (Save to GCS)
# ============================================

def create_plots(real_data, gen_data, features, gcs_output_dir):
    """Create plots and save to GCS"""
    
    if isinstance(gen_data, pd.DataFrame):
        gen_data = gen_data.drop(['Scenario', 'Severity', 'Stress_Level', 'Stress_Score'], 
                                 axis=1, errors='ignore').values
    
    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    key_features = ['GDP', 'CPI', 'Unemployment_Rate', 'VIX', 'SP500_Close', 'Oil_Price']
    plot_idx = [features.index(f) for f in key_features if f in features][:6]
    
    for idx, feat_idx in enumerate(plot_idx):
        axes[idx].hist(real_data[:, feat_idx], bins=50, alpha=0.5, 
                      label='Real', density=True, color='blue')
        axes[idx].hist(gen_data[:, feat_idx], bins=50, alpha=0.5, 
                      label='Generated', density=True, color='orange')
        axes[idx].set_title(features[feat_idx])
        axes[idx].legend()
    
    plt.tight_layout()
    
    # Save to GCS
    gcs_path_prefix = gcs_output_dir.replace('gs://', '')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    with fs.open(f"{gcs_path_prefix}/distributions.png", 'wb') as f:
        f.write(buf.read())
    plt.close()
    
    # Correlation heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    real_corr = np.corrcoef(real_data[:, :20].T)
    gen_corr = np.corrcoef(gen_data[:, :20].T)
    
    sns.heatmap(real_corr, ax=axes[0], cmap='coolwarm', center=0, vmin=-1, vmax=1)
    axes[0].set_title('Real Correlation')
    
    sns.heatmap(gen_corr, ax=axes[1], cmap='coolwarm', center=0, vmin=-1, vmax=1)
    axes[1].set_title('Generated Correlation')
    
    sns.heatmap(real_corr - gen_corr, ax=axes[2], cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5)
    axes[2].set_title('Difference')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    with fs.open(f"{gcs_path_prefix}/correlations.png", 'wb') as f:
        f.write(buf.read())
    plt.close()
    
    print(f"✓ Saved plots to {gcs_output_dir}/\n")


# ============================================
# 7. MAIN PIPELINE
# ============================================

def main(gcs_data_path, gcs_output_dir, hidden_dims=[256, 128, 64], latent_dim=32, 
         batch_size=64, epochs=300, beta=0.5,
         n_baseline=10, n_adverse=20, n_severe=50, n_extreme=20,
         mlflow_tracking_uri=None, experiment_name="Financial_Stress_Test_Scenarios"):
    """Complete pipeline with GCS integration"""
    
    cleanup_gcs_outputs(gcs_output_dir)
    
    print("="*60)
    print("DENSE VAE OPTIMIZED - GCS VERSION")
    print("="*60)
    print(f"Input: {gcs_data_path}")
    print(f"Output: {gcs_output_dir}")
    print("="*60 + "\n")
    
    # Initialize MLflow
    mlflow_config = MLflowConfig(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name
    )
    
    run_name = f"Dense_VAE_GCS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n✓ MLflow Run: {run.info.run_id}\n")
        
        mlflow_config.set_tags({
            'model_type': 'Dense_VAE_Optimized_GCS',
            'stage': 'development',
            'data_source': 'GCS',
            'output_location': 'GCS'
        })
        
        params = {
            'hidden_dims': str(hidden_dims),
            'latent_dim': latent_dim,
            'batch_size': batch_size,
            'max_epochs': epochs,
            'beta': beta,
            'random_seed': 42,
            'gcs_data_path': gcs_data_path,
            'gcs_output_dir': gcs_output_dir
        }
        mlflow_config.log_params(params)
        
        start_time = time.time()
        
        # Load from GCS
        train, val, test, features = load_data_from_gcs(gcs_data_path)
        train_s, val_s, test_s, scaler = normalize_data(train, val, test)
        
        mlflow.log_metrics({
            'n_train_samples': len(train),
            'n_val_samples': len(val),
            'n_test_samples': len(test),
            'n_features': len(features)
        })
        
        # Create loaders
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(train_s)), 
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(val_s)), 
            batch_size=batch_size, shuffle=False
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Train
        model = train_vae(train_loader, val_loader, train_s.shape[1], 
                         hidden_dims, latent_dim, epochs, beta, device)
        
        # Generate
        scenarios = generate_scenarios(model, scaler, features, 
                                       n_baseline, n_adverse, n_severe, n_extreme, device)
        
        # Validate BEFORE classification
        results = validate(test, scenarios, features, gcs_output_dir)
        
        # Add stress classification AFTER validation
        scenarios = classify_stress_level(scenarios, features)
        
        # Visualize
        create_plots(test, scenarios, features, gcs_output_dir)
        
        training_time = time.time() - start_time
        mlflow.log_metric('training_time_seconds', training_time)
        
        # Save to GCS
        print("="*60)
        print("SAVING TO GCS")
        print("="*60 + "\n")
        
        gcs_path_prefix = gcs_output_dir.replace('gs://', '')
        
        # Save scenarios CSV
        with fs.open(f"{gcs_path_prefix}/dense_vae_scenarios.csv", 'w') as f:
            scenarios.to_csv(f, index=False)
        print(f"✓ Saved: {gcs_output_dir}/dense_vae_scenarios.csv")
        
        # Save model to temporary file then upload
        import pickle
        model_package = {
            'model': model.state_dict(),
            'scaler': scaler,
            'features': features,
            'config': {
                'hidden_dims': hidden_dims,
                'latent_dim': latent_dim,
                'beta': beta
            }
        }
        
        # Save model
        with fs.open(f"{gcs_path_prefix}/dense_vae_model.pth", 'wb') as f:
            torch.save(model_package, f)
        print(f"✓ Saved: {gcs_output_dir}/dense_vae_model.pth\n")
        
        # Final summary
        print("="*60)
        print("RESULTS")
        print("="*60)
        print(f"✓ KS Pass Rate: {results['pass_rate']:.1f}%")
        print(f"✓ Correlation MAE: {results['correlation']:.4f}")
        print(f"✓ Wasserstein: {results['wasserstein']:.1f}")
        print(f"✓ Training Time: {training_time:.1f}s")
        print(f"✓ MLflow Run: {run.info.run_id}")
        print(f"✓ GCS Output: {gcs_output_dir}")
        print("="*60 + "\n")
        
        return model, scenarios, results, run.info.run_id


# ============================================
# 8. RUN
# ============================================

if __name__ == "__main__":
    
    from dotenv import load_dotenv
    load_dotenv()
    
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'Financial_Stress_Test_Scenarios')
    
    print("="*60)
    print("DENSE VAE - GCS INTEGRATION")
    print("="*60)
    print(f"Reading from: {GCS_DATA_PATH}")
    print(f"Writing to: {GCS_OUTPUT_BASE}")
    print(f"MLflow: {mlflow_uri}")
    print("="*60 + "\n")
    
    model, scenarios, results, run_id = main(
        gcs_data_path=GCS_DATA_PATH,
        gcs_output_dir=GCS_OUTPUT_BASE,
        hidden_dims=[256, 128, 64],
        latent_dim=32,
        batch_size=64,
        epochs=300,
        beta=0.5,
        n_baseline=10,
        n_adverse=20,
        n_severe=50,
        n_extreme=20,
        mlflow_tracking_uri=mlflow_uri,
        experiment_name=experiment_name
    )
    
    print("="*60)
    print("🎉 SUCCESS!")
    print("="*60)
    print(f"\nAll outputs saved to GCS:")
    print(f"  {GCS_OUTPUT_BASE}/")
    print(f"\nMLflow Run ID: {run_id}")
    print("="*60)