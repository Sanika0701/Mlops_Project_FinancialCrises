"""
FEATURE ENGINEERING PIPELINE FOR CRISIS PREDICTION - WITH GCS
==============================================================

Creates features for:
- M2: XGBoost + LSTM (Predictive model)
- M3: Isolation Forest (Anomaly detector)
- C1: SHAP Explainer (Interpretability)

Input: quarterly_data_complete.csv from GCS (ONE ROW PER QUARTER PER COMPANY)
Output: features_engineered.csv to GCS (with 150+ features)

Author: Financial ML Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from google.cloud import storage
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for crisis prediction models with GCS integration."""
    
    def __init__(self, 
                 bucket_name='mlops-financial-stress-data',
                 local_input_dir='data/processed',
                 local_output_dir='data/processed',
                 use_gcs=True):
        
        self.bucket_name = bucket_name
        self.local_input_dir = Path(local_input_dir)
        self.local_output_dir = Path(local_output_dir)
        self.use_gcs = use_gcs
        
        # Create local directories
        self.local_input_dir.mkdir(parents=True, exist_ok=True)
        self.local_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GCS client
        if self.use_gcs:
            try:
                self.storage_client = storage.Client()
                self.bucket = self.storage_client.bucket(bucket_name)
                logger.info(f"✓ Connected to GCS bucket: gs://{bucket_name}/")
            except Exception as e:
                logger.warning(f"⚠️  GCS connection failed: {e}")
                logger.warning("  Will use local files only")
                self.use_gcs = False
    
    # ========== GCS FUNCTIONS ==========
    
    def download_from_gcs(self, filename, subfolder='data/processed'):
        """Download a file from GCS to local storage."""
        if not self.use_gcs:
            logger.info(f"  Using local file: {self.local_input_dir / filename}")
            return self.local_input_dir / filename
        
        try:
            blob_path = f"{subfolder}/{filename}"
            blob = self.bucket.blob(blob_path)
            
            local_path = self.local_input_dir / filename
            blob.download_to_filename(str(local_path))
            
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ Downloaded: {filename} ({size_mb:.2f} MB)")
            return local_path
            
        except Exception as e:
            logger.error(f"  ✗ Failed to download {filename}: {e}")
            # Try local fallback
            local_path = self.local_input_dir / filename
            if local_path.exists():
                logger.info(f"  Using local fallback: {filename}")
                return local_path
            raise
    
    def upload_to_gcs(self, df, filename, subfolder='data/processed'):
        """Upload DataFrame to GCS."""
        if not self.use_gcs:
            logger.info(f"  GCS upload disabled - file saved locally only")
            return False
        
        try:
            blob_path = f"{subfolder}/{filename}"
            blob = self.bucket.blob(blob_path)
            
            # Convert DataFrame to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            # Upload
            blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
            
            size_mb = len(csv_buffer.getvalue()) / (1024 * 1024)
            logger.info(f"  ✓ Uploaded to GCS: gs://{self.bucket_name}/{blob_path} ({size_mb:.2f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ GCS upload failed: {e}")
            return False
    
    def save_data(self, df, filename):
        """Save data locally and to GCS."""
        # Save locally
        local_path = self.local_output_dir / filename
        df.to_csv(local_path, index=False)
        
        size_mb = local_path.stat().st_size / (1024 * 1024)
        rows = len(df)
        logger.info(f"  ✓ Saved locally: {filename} ({rows:,} rows, {size_mb:.2f} MB)")
        
        # Upload to GCS
        if self.use_gcs:
            self.upload_to_gcs(df, filename)
    
    # ========== 1. LAGGED FEATURES (for LSTM & time-series patterns) ==========
    
    def create_lagged_features(self, df):
        """
        Create lagged features (previous quarters).
        
        Critical for LSTM and XGBoost to learn temporal patterns.
        """
        logger.info("\n[1/8] Creating lagged features...")
        
        df = df.sort_values(['Company', 'Date']).copy()
        
        # Define columns to lag
        lag_cols = [
            # Financial metrics
            'Revenue', 'Net_Income', 'Total_Assets', 'Total_Debt',
            'Gross_Profit', 'Operating_Income', 'EBITDA',
            
            # Ratios
            'Debt_to_Equity', 'Current_Ratio',
            
            # Stock metrics
            'q_return', 'q_price',
            
            # Macro
            'GDP', 'Unemployment_Rate', 'vix_q_mean', 'sp500_q_return'
        ]
        
        # Create lags for 1, 2, 4 quarters back
        lags = [1, 2, 4]
        feature_count = 0
        
        for col in lag_cols:
            if col in df.columns:
                for lag in lags:
                    new_col = f"{col}_lag_{lag}q"
                    df[new_col] = df.groupby('Company')[col].shift(lag)
                    feature_count += 1
        
        logger.info(f"  ✓ Created {feature_count} lagged features")
        logger.info(f"    Example: Revenue_lag_1q (previous quarter revenue)")
        
        return df
    
    # ========== 2. GROWTH RATES (momentum & trends) ==========
    
    def create_growth_features(self, df):
        """
        Calculate quarter-over-quarter and year-over-year growth rates.
        
        Critical for detecting deteriorating fundamentals.
        """
        logger.info("\n[2/8] Creating growth rate features...")
        
        growth_cols = [
            'Revenue', 'Net_Income', 'Total_Assets', 'Total_Debt',
            'Gross_Profit', 'Operating_Income', 'Cash'
        ]
        
        feature_count = 0
        
        for col in growth_cols:
            if col in df.columns:
                # Quarter-over-quarter growth (1q)
                df[f"{col}_growth_1q"] = df.groupby('Company')[col].pct_change(1) * 100
                
                # Year-over-year growth (4q)
                df[f"{col}_growth_4q"] = df.groupby('Company')[col].pct_change(4) * 100
                
                feature_count += 2
        
        logger.info(f"  ✓ Created {feature_count} growth features")
        logger.info(f"    Example: Revenue_growth_1q (QoQ revenue growth %)")
        
        return df
    
    # ========== 2.5. CALCULATE EPS (if possible) ==========
    
    def calculate_eps(self, df):
        """
        Calculate EPS (Earnings Per Share).
        
        Method 1: If we have shares outstanding → EPS = Net_Income / Shares
        Method 2: If EPS column exists but has nulls → try to calculate
        Method 3: Use diluted share count estimate
        
        For financial companies, EPS is often already reported, so we preserve it.
        """
        logger.info("\n[2.5/8] Calculating/Validating EPS...")
        
        # Check if EPS column exists
        if 'EPS' not in df.columns:
            logger.info("  EPS column not found - creating it")
            df['EPS'] = 0.0
        
        # Count existing EPS values
        existing_eps = (~df['EPS'].isna() & (df['EPS'] != 0)).sum()
        total_rows = len(df)
        
        logger.info(f"  Existing valid EPS values: {existing_eps}/{total_rows} ({existing_eps/total_rows*100:.1f}%)")
        
        # If most EPS are missing, try to calculate
        if existing_eps / total_rows < 0.5:
            logger.info(f"  Attempting to calculate missing EPS...")
            
            # Method 1: Estimate shares from market cap (if available)
            # Market Cap = Price × Shares
            # We can estimate shares if we know typical P/E ratios
            
            # Method 2: For missing values, use sector median EPS as proxy
            if 'Sector' in df.columns and 'Net_Income' in df.columns:
                for sector in df['Sector'].unique():
                    sector_mask = df['Sector'] == sector
                    sector_median_eps = df.loc[sector_mask, 'EPS'][df.loc[sector_mask, 'EPS'] > 0].median()
                    
                    if pd.notna(sector_median_eps):
                        # For companies with no EPS, estimate based on Net_Income relative to sector
                        missing_mask = sector_mask & ((df['EPS'].isna()) | (df['EPS'] == 0))
                        if missing_mask.sum() > 0:
                            # Simple estimation: scale by Net_Income
                            sector_median_ni = df.loc[sector_mask, 'Net_Income'].median()
                            if sector_median_ni > 0:
                                df.loc[missing_mask, 'EPS_estimated'] = (
                                    df.loc[missing_mask, 'Net_Income'] / sector_median_ni * sector_median_eps
                                )
            
            # Method 3: Rough approximation from Net_Income
            # Assume typical share count of 1B for large caps
            if 'EPS_estimated' not in df.columns:
                logger.info("  Using rough EPS approximation: Net_Income / 1B shares")
                missing_mask = (df['EPS'].isna()) | (df['EPS'] == 0)
                df.loc[missing_mask, 'EPS_calculated'] = df.loc[missing_mask, 'Net_Income'] / 1_000_000_000
        
        # Fill remaining zeros with calculated/estimated values
        if 'EPS_estimated' in df.columns:
            df['EPS'] = df['EPS'].replace(0, np.nan)
            df['EPS'] = df['EPS'].fillna(df['EPS_estimated'])
            df = df.drop(columns=['EPS_estimated'])
            logger.info("  ✓ Filled missing EPS with sector-based estimates")
        
        if 'EPS_calculated' in df.columns:
            df['EPS'] = df['EPS'].replace(0, np.nan)  
            df['EPS'] = df['EPS'].fillna(df['EPS_calculated'])
            df = df.drop(columns=['EPS_calculated'])
            logger.info("  ✓ Filled missing EPS with Net_Income approximation")
        
        # Final EPS stats
        final_eps = (~df['EPS'].isna() & (df['EPS'] != 0)).sum()
        logger.info(f"  Final EPS coverage: {final_eps}/{total_rows} ({final_eps/total_rows*100:.1f}%)")
        
        if final_eps > 0:
            logger.info(f"  EPS range: ${df['EPS'].min():.2f} to ${df['EPS'].max():.2f}")
            logger.info(f"  EPS mean: ${df['EPS'].mean():.2f}")
        
        return df
    
    # ========== 3. FINANCIAL RATIOS (health indicators) ==========
    
    def create_financial_ratios(self, df):
        """
        Create financial health ratios.
        
        Critical for anomaly detection and crisis prediction.
        """
        logger.info("\n[3/8] Creating financial ratio features...")
        
        feature_count = 0
        
        # NEW: P/E Ratio (if we have EPS)
        if 'EPS' in df.columns and 'q_price' in df.columns:
            df['pe_ratio'] = np.where(
                df['EPS'] > 0,
                df['q_price'] / df['EPS'],
                np.nan
            )
            logger.info(f"  ✓ Created P/E ratio (Price/EPS)")
            feature_count += 1
        
        # NEW: EPS growth
        if 'EPS' in df.columns:
            df['eps_growth_1q'] = df.groupby('Company')['EPS'].pct_change(1) * 100
            df['eps_growth_4q'] = df.groupby('Company')['EPS'].pct_change(4) * 100
            logger.info(f"  ✓ Created EPS growth rates")
            feature_count += 2
        
        # Profitability ratios
        if 'Net_Income' in df.columns and 'Revenue' in df.columns:
            df['net_margin'] = np.where(df['Revenue'] > 0, 
                                        df['Net_Income'] / df['Revenue'] * 100, np.nan)
            feature_count += 1
        
        if 'Gross_Profit' in df.columns and 'Revenue' in df.columns:
            df['gross_margin'] = np.where(df['Revenue'] > 0,
                                          df['Gross_Profit'] / df['Revenue'] * 100, np.nan)
            feature_count += 1
        
        if 'Operating_Income' in df.columns and 'Revenue' in df.columns:
            df['operating_margin'] = np.where(df['Revenue'] > 0,
                                              df['Operating_Income'] / df['Revenue'] * 100, np.nan)
            feature_count += 1
        
        # Asset efficiency
        if 'Net_Income' in df.columns and 'Total_Assets' in df.columns:
            df['roa'] = np.where(df['Total_Assets'] > 0,
                                df['Net_Income'] / df['Total_Assets'] * 100, np.nan)
            feature_count += 1
        
        if 'Net_Income' in df.columns and 'Total_Equity' in df.columns:
            df['roe'] = np.where(df['Total_Equity'] > 0,
                                df['Net_Income'] / df['Total_Equity'] * 100, np.nan)
            feature_count += 1
        
        # Leverage ratios
        if 'Total_Debt' in df.columns and 'Total_Assets' in df.columns:
            df['debt_to_assets'] = np.where(df['Total_Assets'] > 0,
                                           df['Total_Debt'] / df['Total_Assets'], np.nan)
            feature_count += 1
        
        if 'Total_Debt' in df.columns and 'EBITDA' in df.columns:
            df['debt_to_ebitda'] = np.where(df['EBITDA'] > 0,
                                           df['Total_Debt'] / df['EBITDA'], np.nan)
            feature_count += 1
        
        # Liquidity ratios
        if 'Cash' in df.columns and 'Current_Liabilities' in df.columns:
            df['cash_ratio'] = np.where(df['Current_Liabilities'] > 0,
                                       df['Cash'] / df['Current_Liabilities'], np.nan)
            feature_count += 1
        
        logger.info(f"  ✓ Created {feature_count} financial ratio features")
        
        return df
    
    # ========== 4. ROLLING STATISTICS (trends & patterns) ==========
    
    def create_rolling_features(self, df):
        """
        Create rolling window statistics (4-quarter windows).
        
        Captures trends and volatility over time.
        """
        logger.info("\n[4/8] Creating rolling window features...")
        
        rolling_cols = [
            'q_return', 'Revenue', 'Net_Income', 'net_margin', 'roa', 'roe',
            'debt_to_assets', 'vix_q_mean', 'sp500_q_return'
        ]
        
        feature_count = 0
        
        for col in rolling_cols:
            if col in df.columns:
                # 4-quarter rolling mean
                df[f"{col}_rolling4q_mean"] = df.groupby('Company')[col].transform(
                    lambda x: x.rolling(window=4, min_periods=1).mean()
                )
                
                # 4-quarter rolling std (volatility)
                df[f"{col}_rolling4q_std"] = df.groupby('Company')[col].transform(
                    lambda x: x.rolling(window=4, min_periods=1).std()
                )
                
                # 4-quarter rolling min/max
                df[f"{col}_rolling4q_min"] = df.groupby('Company')[col].transform(
                    lambda x: x.rolling(window=4, min_periods=1).min()
                )
                df[f"{col}_rolling4q_max"] = df.groupby('Company')[col].transform(
                    lambda x: x.rolling(window=4, min_periods=1).max()
                )
                
                feature_count += 4
        
        logger.info(f"  ✓ Created {feature_count} rolling window features")
        logger.info(f"    Captures 4-quarter trends and volatility")
        
        return df
    
    # ========== 5. MOMENTUM & ACCELERATION (rate of change) ==========
    
    def create_momentum_features(self, df):
        """
        Create momentum features (change in growth rates).
        
        Detects accelerating/decelerating trends.
        """
        logger.info("\n[5/8] Creating momentum features...")
        
        feature_count = 0
        
        # Revenue acceleration
        if 'Revenue_growth_1q' in df.columns:
            df['revenue_acceleration'] = df.groupby('Company')['Revenue_growth_1q'].diff()
            feature_count += 1
        
        # Margin trends
        if 'net_margin' in df.columns:
            df['net_margin_trend'] = df.groupby('Company')['net_margin'].diff()
            feature_count += 1
        
        # Debt accumulation rate
        if 'debt_to_assets' in df.columns:
            df['debt_accumulation'] = df.groupby('Company')['debt_to_assets'].diff()
            feature_count += 1
        
        # Stock momentum
        if 'q_return' in df.columns:
            df['return_momentum'] = df.groupby('Company')['q_return'].diff()
            feature_count += 1
        
        logger.info(f"  ✓ Created {feature_count} momentum features")
        
        return df
    
    # ========== 6. CRISIS INDICATORS (anomaly detection features) ==========
    
    def create_crisis_indicators(self, df):
        """
        Create crisis-specific indicators.
        
        Designed for Isolation Forest anomaly detection.
        """
        logger.info("\n[6/8] Creating crisis indicator features...")
        
        feature_count = 0
        
        # 1. Stress flags
        if 'vix_q_max' in df.columns:
            df['vix_stress'] = (df['vix_q_max'] > 30).astype(int)  # High fear
            feature_count += 1
        
        if 'Unemployment_Rate' in df.columns:
            df['unemployment_stress'] = (df['Unemployment_Rate'] > 7).astype(int)
            feature_count += 1
        
        # 2. Yield curve inversion (recession predictor)
        if 'Yield_Curve_Spread_mean' in df.columns:
            df['yield_curve_inverted'] = (df['Yield_Curve_Spread_mean'] < 0).astype(int)
            feature_count += 1
        
        # 3. Revenue decline flag
        if 'Revenue_growth_1q' in df.columns:
            df['revenue_declining'] = (df['Revenue_growth_1q'] < -5).astype(int)
            feature_count += 1
        
        # 4. Margin compression
        if 'net_margin' in df.columns and 'net_margin_lag_1q' in df.columns:
            df['margin_compression'] = ((df['net_margin'] - df['net_margin_lag_1q']) < -2).astype(int)
            feature_count += 1
        
        # 5. Excessive leverage
        if 'debt_to_assets' in df.columns:
            # Flag if debt > 70% of assets
            df['high_leverage'] = (df['debt_to_assets'] > 0.7).astype(int)
            feature_count += 1
        
        # 6. Liquidity crunch
        if 'Current_Ratio' in df.columns:
            df['liquidity_risk'] = (df['Current_Ratio'] < 1.0).astype(int)
            feature_count += 1
        
        # 7. Combined stress score (composite)
        stress_cols = [col for col in df.columns if col in [
            'vix_stress', 'unemployment_stress', 'yield_curve_inverted',
            'revenue_declining', 'margin_compression', 'high_leverage', 'liquidity_risk'
        ]]
        
        if stress_cols:
            df['composite_stress_score'] = df[stress_cols].sum(axis=1)
            feature_count += 1
        
        logger.info(f"  ✓ Created {feature_count} crisis indicator features")
        
        return df
    
    # ========== 7. RELATIVE FEATURES (vs market & sector) ==========
    
    def create_relative_features(self, df):
        """
        Create relative performance features.
        
        Compare company vs market and sector peers.
        """
        logger.info("\n[7/8] Creating relative performance features...")
        
        feature_count = 0
        
        # 1. Stock return vs SP500
        if 'q_return' in df.columns and 'sp500_q_return' in df.columns:
            df['excess_return'] = df['q_return'] - df['sp500_q_return']
            df['beta_proxy'] = df['q_return'] / df['sp500_q_return'].replace(0, np.nan)
            feature_count += 2
        
        # 2. Performance vs sector (if multiple companies per sector)
        if 'Sector' in df.columns and 'q_return' in df.columns:
            df['sector_avg_return'] = df.groupby(['Quarter', 'Sector'])['q_return'].transform('mean')
            df['return_vs_sector'] = df['q_return'] - df['sector_avg_return']
            feature_count += 2
        
        # 3. Revenue vs sector
        if 'Sector' in df.columns and 'Revenue_growth_1q' in df.columns:
            df['sector_avg_revenue_growth'] = df.groupby(['Quarter', 'Sector'])['Revenue_growth_1q'].transform('mean')
            df['revenue_growth_vs_sector'] = df['Revenue_growth_1q'] - df['sector_avg_revenue_growth']
            feature_count += 2
        
        logger.info(f"  ✓ Created {feature_count} relative features")
        
        return df
    
    # ========== 8. INTERACTION FEATURES (non-linear relationships) ==========
    
    def create_interaction_features(self, df):
        """
        Create interaction features for XGBoost.
        
        Captures non-linear relationships between variables.
        """
        logger.info("\n[8/8] Creating interaction features...")
        
        feature_count = 0
        
        # 1. Leverage × Market stress
        if 'debt_to_assets' in df.columns and 'vix_q_mean' in df.columns:
            df['leverage_x_vix'] = df['debt_to_assets'] * df['vix_q_mean']
            feature_count += 1
        
        # 2. Margin × Market return
        if 'net_margin' in df.columns and 'sp500_q_return' in df.columns:
            df['margin_x_market'] = df['net_margin'] * df['sp500_q_return']
            feature_count += 1
        
        # 3. Debt × Interest rates
        if 'debt_to_assets' in df.columns and 'Federal_Funds_Rate_mean' in df.columns:
            df['debt_x_rates'] = df['debt_to_assets'] * df['Federal_Funds_Rate_mean']
            feature_count += 1
        
        # 4. Revenue decline × High VIX (crisis signal)
        if 'Revenue_growth_1q' in df.columns and 'vix_q_mean' in df.columns:
            df['revenue_decline_x_vix'] = df['Revenue_growth_1q'] * df['vix_q_mean']
            feature_count += 1
        
        logger.info(f"  ✓ Created {feature_count} interaction features")
        
        return df
    
    # ========== BONUS: NORMALIZATION & SCALING ==========
    
    def create_normalized_features(self, df):
        """
        Create log-transformed and z-score normalized features.
        
        Important for Isolation Forest (needs similar scales).
        """
        logger.info("\n[BONUS] Creating normalized features...")
        
        feature_count = 0
        
        # Log transform large numbers (Revenue, Assets)
        log_cols = ['Revenue', 'Total_Assets', 'Total_Debt', 'Cash']
        
        for col in log_cols:
            if col in df.columns:
                df[f"log_{col}"] = np.log1p(df[col])  # log(1 + x) handles zeros
                feature_count += 1
        
        # Z-score normalize key ratios (within company)
        zscore_cols = ['net_margin', 'roa', 'roe', 'debt_to_assets']
        
        for col in zscore_cols:
            if col in df.columns:
                df[f"{col}_zscore"] = df.groupby('Company')[col].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                )
                feature_count += 1
        
        logger.info(f"  ✓ Created {feature_count} normalized features")
        
        return df
    
    # ========== TARGET VARIABLE CREATION ==========
    
    def create_target_variables(self, df):
        """
        Create target variables for supervised learning.
        
        - next_q_return: Predict next quarter's stock return
        - crisis_flag: Binary crisis indicator
        """
        logger.info("\n[TARGET] Creating target variables...")
        
        # 1. Next quarter return (regression target)
        if 'q_return' in df.columns:
            df['next_q_return'] = df.groupby('Company')['q_return'].shift(-1)
            logger.info(f"  ✓ next_q_return (predict future return)")
        
        # 2. Crisis flag (classification target)
        # Define crisis as: large stock drop + high VIX + negative growth
        crisis_conditions = []
        
        if 'q_return' in df.columns:
            crisis_conditions.append(df['q_return'] < -20)  # >20% drop
        
        if 'vix_q_mean' in df.columns:
            crisis_conditions.append(df['vix_q_mean'] > 30)  # High fear
        
        if 'Revenue_growth_1q' in df.columns:
            crisis_conditions.append(df['Revenue_growth_1q'] < -10)  # Revenue crash
        
        if crisis_conditions:
            df['crisis_flag'] = sum(crisis_conditions).ge(2).astype(int)  # 2+ conditions = crisis
            crisis_count = df['crisis_flag'].sum()
            logger.info(f"  ✓ crisis_flag ({crisis_count} crisis quarters, {crisis_count/len(df)*100:.1f}%)")
        
        return df
    
    # ========== MAIN PIPELINE ==========
    
    def run_pipeline(self):
        """Execute complete feature engineering pipeline with GCS."""
        logger.info("\n" + "="*80)
        logger.info("FEATURE ENGINEERING PIPELINE WITH GCS")
        logger.info("="*80)
        logger.info(f"GCS Bucket: gs://{self.bucket_name}/")
        logger.info(f"GCS Enabled: {self.use_gcs}")
        logger.info("="*80)
        
        # Step 0: Download from GCS
        logger.info("\n" + "="*80)
        logger.info("STEP 0: DOWNLOADING FROM GCS")
        logger.info("="*80)
        
        try:
            input_file = self.download_from_gcs('quarterly_data_complete.csv')
        except Exception as e:
            logger.error(f"✗ Failed to download data: {e}")
            return None
        
        # Load cleaned data
        logger.info(f"\nLoading data from: {input_file}")
        try:
            df = pd.read_csv(input_file)
            df['Date'] = pd.to_datetime(df['Date'])
            logger.info(f"✓ Loaded: {df.shape}")
        except Exception as e:
            logger.error(f"✗ Error loading data: {e}")
            return None
        
        original_cols = df.shape[1]
        
        # Execute feature engineering steps
        logger.info("\n" + "="*80)
        logger.info("CREATING FEATURES")
        logger.info("="*80)
        
        df = self.calculate_eps(df)  # Calculate EPS FIRST
        df = self.create_lagged_features(df)
        df = self.create_growth_features(df)
        df = self.create_financial_ratios(df)  # Now can use EPS for P/E ratio
        df = self.create_rolling_features(df)
        df = self.create_momentum_features(df)
        df = self.create_crisis_indicators(df)
        df = self.create_relative_features(df)
        df = self.create_interaction_features(df)
        df = self.create_normalized_features(df)
        df = self.create_target_variables(df)
        
        # Summary
        new_cols = df.shape[1]
        added_cols = new_cols - original_cols
        
        logger.info("\n" + "="*80)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("="*80)
        
        logger.info(f"\n📊 Feature Summary:")
        logger.info(f"   Original columns: {original_cols}")
        logger.info(f"   New features added: {added_cols}")
        logger.info(f"   Total columns: {new_cols}")
        
        # Feature categories
        logger.info(f"\n📋 Feature Categories:")
        logger.info(f"   1. Lagged features: ~{len([c for c in df.columns if 'lag_' in c])}")
        logger.info(f"   2. Growth rates: ~{len([c for c in df.columns if 'growth_' in c])}")
        logger.info(f"   3. Financial ratios: ~{len([c for c in df.columns if any(x in c for x in ['margin', 'roa', 'roe', 'ratio'])])}")
        logger.info(f"   4. Rolling stats: ~{len([c for c in df.columns if 'rolling' in c])}")
        logger.info(f"   5. Crisis indicators: ~{len([c for c in df.columns if any(x in c for x in ['stress', 'flag', 'risk'])])}")
        
        # Check missing data
        logger.info(f"\n⚠️  Missing Data After Feature Engineering:")
        missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False).head(10)
        for col, pct in missing_pct.items():
            if pct > 20:
                logger.info(f"   {col}: {pct:.1f}%")
        
        # Save locally and upload to GCS
        logger.info("\n" + "="*80)
        logger.info("SAVING OUTPUT")
        logger.info("="*80)
        
        self.save_data(df, 'features_engineered.csv')
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("READY FOR MODELING ✅")
        logger.info("="*80)
        
        logger.info(f"\n📁 Output Locations:")
        logger.info(f"   Local: {self.local_output_dir / 'features_engineered.csv'}")
        if self.use_gcs:
            logger.info(f"   GCS: gs://{self.bucket_name}/data/processed/features_engineered.csv")
        
        logger.info(f"\n🎯 Model-Specific Features:")
        logger.info(f"\n   M2: XGBoost + LSTM (Predictive)")
        logger.info(f"   ✓ Lagged features (temporal context)")
        logger.info(f"   ✓ Growth rates (momentum)")
        logger.info(f"   ✓ Rolling statistics (trends)")
        logger.info(f"   → Use for: Forecasting Revenue, EPS, Stock Returns")
        
        logger.info(f"\n   M3: Isolation Forest (Anomaly Detection)")
        logger.info(f"   ✓ Crisis indicators (flags)")
        logger.info(f"   ✓ Normalized features (z-scores)")
        logger.info(f"   ✓ Extreme value detection (rolling min/max)")
        logger.info(f"   → Use for: Detecting financial stress, crises")
        
        logger.info(f"\n   C1: SHAP Explainer")
        logger.info(f"   ✓ All features are SHAP-compatible")
        logger.info(f"   ✓ Interpretable names (e.g., 'net_margin', 'vix_stress')")
        logger.info(f"   → Use for: Explaining model predictions")
        
        return df


# ========== USAGE ==========

if __name__ == "__main__":
    engineer = FeatureEngineer(
        bucket_name='mlops-financial-stress-data',
        local_input_dir='data/processed',
        local_output_dir='data/processed',
        use_gcs=True  # Set to False to disable GCS and use local files only
    )
    
    features_df = engineer.run_pipeline()
    
    if features_df is not None:
        print("\n" + "="*80)
        print("SUCCESS! Features engineered and ready for modeling.")
        print("="*80)
        print(f"\nLocal Output: data/processed/features_engineered.csv")
        print(f"GCS Output: gs://mlops-financial-stress-data/data/processed/features_engineered.csv")
        print(f"\nNext steps:")
        print(f"1. Train M2 (XGBoost + LSTM) on features + targets")
        print(f"2. Train M3 (Isolation Forest) on crisis indicators")
        print(f"3. Apply SHAP explainer to interpret predictions")