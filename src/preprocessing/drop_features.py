

# """
# src/preprocessing/drop_leakage_features.py

# Drops leakage features AFTER targets are created.
# Keeps target columns + SAFE PREDICTIVE FEATURES.

# CRITICAL: Only drops CURRENT QUARTER values. Keeps LAGGED and ROLLING features!

# Input:  data/features/quarterly_data_with_targets.csv (has targets + leakage)
# Output: data/features/quarterly_data_with_targets_clean.csv (targets + safe features only)
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')


# def drop_leakage_features(input_file: str, output_file: str):
#     """
#     Drop features that cause data leakage (but KEEP target columns and SAFE predictive features!)
    
#     DROPS: Current quarter values (Revenue, EPS, q_return, etc.)
#     KEEPS: Lagged values (Revenue_lag_1q), Rolling aggregates (Revenue_rolling4q_mean)
    
#     Args:
#         input_file: Path to quarterly_data_with_targets.csv
#         output_file: Path to save cleaned file
#     """
    
#     print("="*80)
#     print("DROPPING LEAKAGE FEATURES (KEEPING SAFE PREDICTIVE FEATURES)")
#     print("="*80)
    
#     # ========================================
#     # 1. LOAD DATA
#     # ========================================
    
#     print("\n[Step 1] Loading data with targets...")
    
#     input_path = Path(input_file)
#     output_path = Path(output_file)
    
#     if not input_path.exists():
#         raise FileNotFoundError(f"File not found: {input_file}")
    
#     df = pd.read_csv(input_path)
    
#     print(f"   [SUCCESS] Loaded: {len(df):,} rows x {len(df.columns)} columns")
#     print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
#     original_columns = df.columns.tolist()
#     original_shape = df.shape
    
#     # Check for target columns
#     target_cols = [col for col in df.columns if col.startswith('target_')]
#     print(f"   Target columns found: {len(target_cols)}")
#     for col in target_cols:
#         print(f"      [KEEP] {col}")
    
#     if not target_cols:
#         print(f"   [WARNING] No target columns found!")
#         print(f"      Expected: target_revenue, target_eps, etc.")
#         print(f"      Run create_targets.py first!")
    
#     # ========================================
#     # 2. DEFINE FEATURES TO DROP (CURRENT QUARTER ONLY!)
#     # ========================================
    
#     print("\n[Step 2] Defining leakage features to drop...")
#     print("   [INFO] Only dropping CURRENT QUARTER values")
#     print("   [INFO] KEEPING all _lag_, rolling, and historical features!")
    
#     FEATURES_TO_DROP = [
#         # ========================================
#         # A. CURRENT QUARTER VALUES OF TARGETS ONLY
#         # Note: We KEEP lagged versions (e.g., Revenue_lag_1q)
#         # ========================================
#         'Revenue',                    # Current only (KEEP Revenue_lag_1q, Revenue_lag_2q, etc.)
#         'EPS',                        # Current only (KEEP EPS_lag_*)
#         'Debt_to_Equity',             # Current only (KEEP Debt_to_Equity_lag_1q, lag_2q, lag_4q!)
#         'net_margin',                 # Current only (KEEP net_margin_rolling4q_*)
#         'net_margin_q',               # Same as net_margin
#         'gross_margin',               # Current only
#         'operating_margin',           # Current only
#         'roa',                        # Current only (KEEP roa_rolling4q_*)
#         'roe',                        # Current only (KEEP roe_rolling4q_*)
#         'q_return',                   # Current only (KEEP q_return_lag_1q, lag_2q, lag_4q!)
#         'stock_q_return',             # Same as q_return
#         'next_q_return',              # FUTURE return (major leakage!)
        
#         # ========================================
#         # B. GROWTH FROM CURRENT TO LAST QUARTER ONLY
#         # Note: We KEEP year-over-year growth (e.g., *_growth_4q if calculated safely)
#         # ========================================
#         'Revenue_growth_1q',          # Current to Last quarter
#         'Net_Income_growth_1q',       
#         'Gross_Profit_growth_1q',     
#         'Operating_Income_growth_1q',
#         'eps_growth_1q',              
        
#         # ========================================
#         # C. CURRENT QUARTER FUNDAMENTALS ONLY
#         # Note: We KEEP lagged versions (Total_Debt_lag_1q, etc.)
#         # ========================================
#         'Net_Income',                 # Current only (KEEP Net_Income_lag_*)
#         'Gross_Profit',               # Current only (KEEP Gross_Profit_lag_*)
#         'Operating_Income',           # Current only
#         'EBITDA',                     # Current only (KEEP EBITDA_lag_*)
#         'Total_Debt',                 # Current only (KEEP Total_Debt_lag_1q, lag_2q, lag_4q!)
#         'Total_Assets',               # Current only (KEEP Total_Assets_lag_1q, lag_2q, lag_4q!)
#         'Total_Liabilities',          # Current only
#         'Total_Equity',               # Current only (KEEP Total_Equity_lag_*)
#         'Current_Assets',             # Current only
#         'Current_Liabilities',        # Current only
#         'Long_Term_Debt',             # Current only (KEEP Long_Term_Debt_lag_*)
#         'Short_Term_Debt',            # Current only
#         'Cash',                       # Current only (KEEP Cash_lag_*)
#         'Current_Ratio',              # Current only (KEEP Current_Ratio_lag_1q, lag_2q!)
        
#         # ========================================
#         # D. CURRENT QUARTER STOCK PRICES ONLY
#         # Note: We KEEP lagged prices (q_price_lag_1q, etc.)
#         # ========================================
#         'Stock_Price',                # Current only
#         'q_price',                    # Current only (KEEP q_price_lag_*)
#         'q_volume',                   # Current only
#         'q_high',                     # Current only
#         'q_low',                      # Current only
#         'q_open',                     # Current only
#         'q_price_range_pct',          # Current only
#         'Open',                       # Daily data - drop
#         'High',
#         'Low',
#         'Close',
#         'Adj_Close',
#         'Volume',
        
#         # ========================================
#         # E. RATIOS USING CURRENT QUARTER TARGETS
#         # Note: We KEEP lagged ratios and rolling aggregates
#         # ========================================
#         'pe_ratio',                   # Current only (uses current EPS)
#         'debt_to_assets',             # Current only (KEEP debt_to_assets_rolling4q_mean!)
#         'debt_to_ebitda',             # Current only
#         'cash_ratio',                 # Current only
        
#         # ========================================
#         # F. ENGINEERED FEATURES USING CURRENT QUARTER
#         # ========================================
#         'revenue_acceleration',       # Uses current Revenue growth
#         'net_margin_trend',           # Uses current net_margin
#         'return_momentum',            # Uses current q_return
#         'revenue_declining',          # Uses current Revenue
#         'high_leverage',              # Uses current Debt_to_Equity
#         'liquidity_risk',             # Uses current ratios
#         'composite_stress_score',     # Composite of current targets
#         'leverage_x_vix',             # Uses current leverage
#         'margin_x_market',            # Uses current margin
#         'revenue_decline_x_vix',      # Uses current Revenue
#         'excess_return',              # Uses current q_return
#         'return_vs_sector',           # Uses current q_return
#         'revenue_growth_vs_sector',   # Uses current Revenue growth
#         'debt_x_rates',               # Uses current debt
        
#         # ========================================
#         # G. Z-SCORES OF CURRENT TARGETS
#         # ========================================
#         'net_margin_zscore',          
#         'roa_zscore',
#         'roe_zscore',
#         'debt_to_assets_zscore',
        
#         # ========================================
#         # H. LOG OF CURRENT VALUES
#         # ========================================
#         'log_Revenue',                # Current quarter
#         'log_Total_Assets',
#         'log_Total_Debt',
#         'log_Cash',
        
#         # ========================================
#         # I. CLASSIFICATION LABELS
#         # ========================================
#         'crisis_flag',
        
#         # ========================================
#         # J. REDUNDANT COLUMNS
#         # ========================================
#         'Quarter_End_Date',
#         'Original_Quarter_End',
#         'Quarter_End_Date_fred',
#         'Company_Name',
        
#         # ========================================
#         # K. INTERMEDIATE CALCULATIONS
#         # ========================================
#         'EPS_calculated',             
#         'profit_margin_calculated',   
#         'return_calculated',          
        
#         # ========================================
#         # L. YEAR-OVER-YEAR GROWTH (IF USING CURRENT VALUES)
#         # Note: If *_growth_4q uses lagged values, it's safe. 
#         # Dropping to be conservative.
#         # ========================================
#         'Revenue_growth_4q',          # Often uses current Revenue
#         'Net_Income_growth_4q',
#         'Gross_Profit_growth_4q',
#         'Operating_Income_growth_4q',
#         'eps_growth_4q',
#     ]
    
#     # ========================================
#     # 2b. VERIFY SAFE FEATURES ARE NOT IN DROP LIST
#     # ========================================
    
#     print(f"\n   Defined {len(FEATURES_TO_DROP)} features to drop")
#     print(f"\n   [VERIFY] Checking SAFE features are NOT being dropped:")
    
#     SAFE_FEATURES_THAT_SHOULD_BE_KEPT = [
#         # Critical for debt_equity prediction
#         'Debt_to_Equity_lag_1q', 'Debt_to_Equity_lag_2q', 'Debt_to_Equity_lag_4q',
#         'Total_Debt_lag_1q', 'Total_Debt_lag_2q', 'Total_Debt_lag_4q',
#         'Total_Assets_lag_1q', 'Total_Assets_lag_2q', 'Total_Assets_lag_4q',
#         'debt_to_assets_rolling4q_mean', 'debt_to_assets_rolling4q_std',
        
#         # Critical for stock_return prediction
#         'q_return_lag_1q', 'q_return_lag_2q', 'q_return_lag_4q',
#         'q_return_rolling4q_mean', 'q_return_rolling4q_std',
#         'sp500_q_return_lag_1q', 'sp500_q_return_lag_2q',
#         'sector_avg_return',
        
#         # General lagged fundamentals
#         'Revenue_lag_1q', 'Revenue_lag_2q', 'Revenue_lag_4q',
#         'Net_Income_lag_1q', 'Net_Income_lag_2q',
#         'Current_Ratio_lag_1q', 'Current_Ratio_lag_2q',
        
#         # Rolling features
#         'Revenue_rolling4q_mean', 'Revenue_rolling4q_std',
#         'Net_Income_rolling4q_mean', 'net_margin_rolling4q_mean',
#         'roa_rolling4q_mean', 'roe_rolling4q_mean',
#     ]
    
#     # Check if any safe features are accidentally in drop list
#     accidentally_dropping = [f for f in SAFE_FEATURES_THAT_SHOULD_BE_KEPT if f in FEATURES_TO_DROP]
    
#     if accidentally_dropping:
#         print(f"\n   [ERROR] Safe features found in drop list!")
#         for f in accidentally_dropping:
#             print(f"      [FAIL] {f} - THIS SHOULD BE KEPT!")
#         raise ValueError(f"Safe features in drop list: {accidentally_dropping}")
#     else:
#         print(f"      [SUCCESS] All {len(SAFE_FEATURES_THAT_SHOULD_BE_KEPT)} critical features will be kept")
    
#     # ========================================
#     # 3. CHECK WHICH FEATURES EXIST
#     # ========================================
    
#     print("\n[Step 3] Checking which features exist...")
    
#     # CRITICAL: Don't drop target columns!
#     existing_to_drop = [
#         col for col in FEATURES_TO_DROP 
#         if col in df.columns and not col.startswith('target_')
#     ]
    
#     missing_to_drop = [col for col in FEATURES_TO_DROP if col not in df.columns]
    
#     # Check how many safe features actually exist in the data
#     existing_safe = [f for f in SAFE_FEATURES_THAT_SHOULD_BE_KEPT if f in df.columns]
#     missing_safe = [f for f in SAFE_FEATURES_THAT_SHOULD_BE_KEPT if f not in df.columns]
    
#     print(f"   Will drop: {len(existing_to_drop)} leakage features")
#     print(f"   Already absent: {len(missing_to_drop)}")
#     print(f"   \n   [SUCCESS] Safe features found in data: {len(existing_safe)}/{len(SAFE_FEATURES_THAT_SHOULD_BE_KEPT)}")
    
#     if missing_safe:
#         print(f"   [WARNING] {len(missing_safe)} critical features missing from data:")
#         for f in missing_safe[:10]:
#             print(f"      - {f}")
#         if len(missing_safe) > 10:
#             print(f"      ... and {len(missing_safe)-10} more")
#         print(f"\n   [WARNING] This may explain low R-squared for debt_equity and stock_return!")
#         print(f"   [INFO] These features should exist if feature engineering was run correctly")
    
#     if existing_to_drop:
#         print(f"\n   [DROP] Dropping {len(existing_to_drop)} leakage features:")
#         for i, col in enumerate(existing_to_drop[:15], 1):
#             print(f"      {i}. {col}")
#         if len(existing_to_drop) > 15:
#             print(f"      ... and {len(existing_to_drop) - 15} more")
    
#     # ========================================
#     # 4. DROP FEATURES (KEEP TARGETS & SAFE FEATURES!)
#     # ========================================
    
#     print(f"\n[Step 4] Dropping leakage features (keeping targets & safe features)...")
    
#     df_clean = df.drop(columns=existing_to_drop)
    
#     print(f"   [SUCCESS] Dropped {len(existing_to_drop)} leakage features")
#     print(f"   [SUCCESS] Kept {len(existing_safe)} safe predictive features")
#     print(f"   Before: {original_shape[1]} columns")
#     print(f"   After:  {df_clean.shape[1]} columns")
#     print(f"   Reduction: {len(existing_to_drop)} columns")
    
#     # Verify target columns are still there
#     remaining_targets = [col for col in df_clean.columns if col.startswith('target_')]
#     print(f"\n   [SUCCESS] Target columns preserved: {len(remaining_targets)}")
#     for col in remaining_targets:
#         valid_count = df_clean[col].notna().sum()
#         print(f"      [KEEP] {col}: {valid_count:,} valid values")
    
#     # Verify safe features are still there
#     remaining_safe = [f for f in SAFE_FEATURES_THAT_SHOULD_BE_KEPT if f in df_clean.columns]
#     print(f"\n   [SUCCESS] Safe predictive features kept: {len(remaining_safe)}")
#     if len(remaining_safe) < len(SAFE_FEATURES_THAT_SHOULD_BE_KEPT):
#         print(f"      [WARNING] Only {len(remaining_safe)}/{len(SAFE_FEATURES_THAT_SHOULD_BE_KEPT)} found")
    
#     # ========================================
#     # 5. VALIDATE REMAINING FEATURES
#     # ========================================
    
#     print(f"\n[Step 5] Validating cleaned dataset...")
    
#     remaining_cols = df_clean.columns.tolist()
    
#     # Categorize
#     identifier_cols = [c for c in remaining_cols if c in ['Date', 'Year', 'Quarter', 'Quarter_Num', 'Company', 'Sector']]
#     target_features = [c for c in remaining_cols if c.startswith('target_')]
#     macro_cols = [c for c in remaining_cols if any(x in c for x in ['GDP', 'CPI', 'Unemployment', 'Federal', 'Yield', 'Consumer', 'Oil', 'vix', 'sp500'])]
#     lag_cols = [c for c in remaining_cols if '_lag_' in c]
#     rolling_cols = [c for c in remaining_cols if 'rolling' in c]
    
#     print(f"\n   [SUMMARY] Cleaned dataset composition:")
#     print(f"      Identifiers:      {len(identifier_cols)}")
#     print(f"      Target variables: {len(target_features)} (KEPT)")
#     print(f"      Macro features:   {len(macro_cols)}")
#     print(f"      Lagged features:  {len(lag_cols)} (KEPT - critical for prediction)")
#     print(f"      Rolling features: {len(rolling_cols)} (KEPT - critical for prediction)")
#     print(f"      TOTAL:            {len(remaining_cols)}")
    
#     # Show some lagged features that were kept
#     print(f"\n   [EXAMPLES] KEPT lagged features:")
#     example_lags = [c for c in lag_cols if any(x in c for x in ['Debt_to_Equity', 'q_return', 'Revenue'])]
#     for lag in example_lags[:10]:
#         print(f"      [KEEP] {lag}")
    
#     # Check for any remaining leakage
#     suspicious = []
#     leakage_keywords = ['Revenue', 'EPS', 'net_margin', 'Debt_to_Equity', 'q_return', 'Stock_Price']
    
#     for col in remaining_cols:
#         # Skip target columns (they're supposed to be there!)
#         if col.startswith('target_'):
#             continue
#         # Check for current-quarter values
#         if any(keyword in col for keyword in leakage_keywords):
#             # Allow lagged/rolling/sector versions
#             if not any(x in col for x in ['_lag_', 'rolling', 'sector_avg', 'sp500', 'vix']):
#                 suspicious.append(col)
    
#     if suspicious:
#         print(f"\n   [WARNING] Potentially leaky features remain:")
#         for col in suspicious:
#             print(f"      - {col}")
#     else:
#         print(f"\n   [SUCCESS] No leakage detected in remaining features")
    
#     # ========================================
#     # 6. SAVE TO NEW FILE
#     # ========================================
    
#     print(f"\n[Step 6] Saving cleaned dataset to NEW file...")
    
#     # Create output directory if needed
#     output_path.parent.mkdir(parents=True, exist_ok=True)
    
#     # Save to NEW file (not overwriting!)
#     df_clean.to_csv(output_path, index=False)
    
#     output_size = output_path.stat().st_size / (1024*1024)
    
#     print(f"   [SUCCESS] Saved to: {output_path}")
#     print(f"      Size: {output_size:.1f} MB")
#     print(f"      Rows: {df_clean.shape[0]:,}")
#     print(f"      Columns: {df_clean.shape[1]}")
    
#     # ========================================
#     # 7. SAVE DROPPED FEATURES LOG
#     # ========================================
    
#     print(f"\n[Step 7] Saving log...")
    
#     log_file = output_path.parent / 'dropped_features_log.txt'
    
#     with open(log_file, 'w') as f:
#         f.write("="*80 + "\n")
#         f.write("DROPPED FEATURES LOG\n")
#         f.write("="*80 + "\n")
#         f.write(f"Date: {datetime.now().isoformat()}\n")
#         f.write(f"Input file: {input_file}\n")
#         f.write(f"Output file: {output_file}\n")
#         f.write(f"Original columns: {original_shape[1]}\n")
#         f.write(f"Remaining columns: {df_clean.shape[1]}\n")
#         f.write(f"Dropped: {len(existing_to_drop)}\n")
#         f.write("\n" + "="*80 + "\n")
#         f.write("FEATURES DROPPED:\n")
#         f.write("="*80 + "\n\n")
        
#         for i, col in enumerate(existing_to_drop, 1):
#             f.write(f"{i}. {col}\n")
        
#         f.write("\n" + "="*80 + "\n")
#         f.write("TARGET COLUMNS PRESERVED:\n")
#         f.write("="*80 + "\n\n")
        
#         for col in remaining_targets:
#             f.write(f"[KEEP] {col}\n")
        
#         f.write("\n" + "="*80 + "\n")
#         f.write("SAFE PREDICTIVE FEATURES KEPT:\n")
#         f.write("="*80 + "\n\n")
        
#         for col in remaining_safe:
#             f.write(f"[KEEP] {col}\n")
        
#         f.write("\n" + "="*80 + "\n")
#         f.write("SAFE FEATURES MISSING FROM DATA:\n")
#         f.write("="*80 + "\n\n")
        
#         if missing_safe:
#             for col in missing_safe:
#                 f.write(f"[MISSING] {col}\n")
#         else:
#             f.write("None - all expected safe features present!\n")
        
#         f.write("\n" + "="*80 + "\n")
#         f.write("FEATURES NOT FOUND (already absent):\n")
#         f.write("="*80 + "\n\n")
        
#         for col in missing_to_drop:
#             f.write(f"- {col}\n")
    
#     print(f"   [SUCCESS] Log saved: {log_file}")
    
#     # ========================================
#     # 8. SUMMARY
#     # ========================================
    
#     print(f"\n{'='*80}")
#     print(f"FEATURE CLEANING SUMMARY")
#     print(f"{'='*80}")
    
#     print(f"\n[INPUT FILE] (preserved):")
#     print(f"   {input_file}")
#     print(f"   Rows: {original_shape[0]:,}")
#     print(f"   Columns: {original_shape[1]}")
    
#     print(f"\n[OUTPUT FILE] (cleaned):")
#     print(f"   {output_file}")
#     print(f"   Rows: {df_clean.shape[0]:,}")
#     print(f"   Columns: {df_clean.shape[1]}")
    
#     print(f"\n[CHANGES]:")
#     print(f"   Leakage features dropped: {len(existing_to_drop)}")
#     print(f"   Target columns preserved: {len(remaining_targets)}")
#     print(f"   Safe features kept: {len(remaining_safe)}")
#     print(f"   Reduction: {len(existing_to_drop)/original_shape[1]*100:.1f}%")
    
#     print(f"\n[FILES]:")
#     print(f"   Original: {input_file} (unchanged)")
#     print(f"   Cleaned:  {output_file} (new)")
#     print(f"   Log:      {log_file}")
    
#     if missing_safe:
#         print(f"\n[IMPORTANT]:")
#         print(f"   {len(missing_safe)} critical features are missing from your data!")
#         print(f"   This may explain low R-squared for debt_equity (0.38) and stock_return (0.0008)")
#         print(f"\n   [SOLUTION]:")
#         print(f"   1. Check if feature engineering created these features")
#         print(f"   2. Run: python src/features/build_features.py")
#         print(f"   3. Ensure lagged and rolling features are created")
    
#     print(f"\n{'='*80}")
#     print(f"[SUCCESS] FEATURE CLEANING COMPLETE!")
#     print(f"{'='*80}")
    
#     print(f"\n[NEXT STEP]:")
#     print(f"   Create temporal splits:")
#     print(f"   python src/preprocessing/temporal_split.py")
    
#     return df_clean, existing_to_drop


# if __name__ == "__main__":
#     """
#     Main execution: Drop leakage features after targets are created
#     """
    
#     # Input: File WITH targets (from create_targets.py)
#     input_file = 'data/features/quarterly_data_with_targets.csv'
    
#     # Output: NEW cleaned file (targets + safe features only)
#     output_file = 'data/features/quarterly_data_with_targets_clean.csv'
    
#     try:
#         print(f"\n{'='*80}")
#         print(f"STARTING FEATURE CLEANING")
#         print(f"{'='*80}")
#         print(f"\n[INPUT]  {input_file}")
#         print(f"[OUTPUT] {output_file}")
#         print(f"   (Original file will be preserved!)")
        
#         # Drop leakage features
#         df_clean, dropped = drop_leakage_features(
#             input_file=input_file,
#             output_file=output_file
#         )
        
#         print(f"\n{'='*80}")
#         print(f"[SUCCESS] Feature cleaning completed!")
#         print(f"{'='*80}")
        
#         print(f"\n[SUMMARY]:")
#         print(f"   [DONE] Read: {input_file}")
#         print(f"   [DONE] Dropped {len(dropped)} leakage features (CURRENT quarter only)")
#         print(f"   [DONE] Kept target columns")
#         print(f"   [DONE] Kept lagged & rolling features (PAST quarters)")
#         print(f"   [DONE] Saved: {output_file}")
        
#         print(f"\n[FILES]:")
#         print(f"   Original (with leakage): {input_file}")
#         print(f"   Cleaned (safe):          {output_file}")
        
#     except FileNotFoundError as e:
#         print(f"\n[ERROR] FILE NOT FOUND:")
#         print(f"   {e}")
#         print(f"\n[INFO] Run create_targets.py first to create:")
#         print(f"   data/features/quarterly_data_with_targets.csv")
        
#     except Exception as e:
#         print(f"\n[ERROR]:")
#         print(f"   {e}")
#         import traceback
#         traceback.print_exc()



"""
src/preprocessing/drop_leakage_features.py

Drops leakage features AFTER targets are created.
Keeps target columns + SAFE PREDICTIVE FEATURES.

CRITICAL: Only drops CURRENT QUARTER values. Keeps LAGGED and ROLLING features!

Input:  data/features/quarterly_data_with_targets.csv (has targets + leakage)
Output: data/features/quarterly_data_with_targets_clean.csv (targets + safe features only)
Auto-uploads to GCS: gs://mlops-financial-stress-data/data/features/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import subprocess
import os
warnings.filterwarnings('ignore')


def upload_to_gcs(local_file: str, bucket_name: str, gcs_path: str):
    """
    Upload file to Google Cloud Storage using gsutil
    
    Args:
        local_file: Path to local file
        bucket_name: GCS bucket name
        gcs_path: Destination path in bucket (e.g., 'data/features/file.csv')
    """
    print(f"\n{'='*80}")
    print(f"📤 UPLOADING TO GOOGLE CLOUD STORAGE")
    print(f"{'='*80}")
    
    gcs_uri = f"gs://{bucket_name}/{gcs_path}"
    
    print(f"\n   Local file: {local_file}")
    print(f"   GCS destination: {gcs_uri}")
    
    try:
        # Check if file exists
        if not Path(local_file).exists():
            print(f"\n   ❌ File not found: {local_file}")
            return False
        
        # Get file size
        file_size = Path(local_file).stat().st_size / (1024*1024)
        print(f"   File size: {file_size:.2f} MB")
        
        # Upload using gsutil
        result = subprocess.run(
            ["gsutil", "cp", local_file, gcs_uri],
            check=True,
            capture_output=True,
            text=True
        )
        
        print(f"\n   ✅ Successfully uploaded to: {gcs_uri}")
        print(f"   Upload time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n   ❌ Upload failed:")
        print(f"      Error: {e.stderr}")
        print(f"\n   💡 Manual upload command:")
        print(f"      gsutil cp {local_file} {gcs_uri}")
        return False
        
    except FileNotFoundError:
        print(f"\n   ⚠️  gsutil not found. Install Google Cloud SDK:")
        print(f"      https://cloud.google.com/sdk/docs/install")
        print(f"\n   💡 Manual upload command:")
        print(f"      gsutil cp {local_file} {gcs_uri}")
        return False
        
    except Exception as e:
        print(f"\n   ❌ Unexpected error: {e}")
        return False


def drop_leakage_features(input_file: str, output_file: str, upload_to_cloud: bool = True, bucket_name: str = "mlops-financial-stress-data"):
    """
    Drop features that cause data leakage (but KEEP target columns and SAFE predictive features!)
    
    DROPS: Current quarter values (Revenue, EPS, q_return, etc.)
    KEEPS: Lagged values (Revenue_lag_1q), Rolling aggregates (Revenue_rolling4q_mean)
    
    Args:
        input_file: Path to quarterly_data_with_targets.csv
        output_file: Path to save cleaned file
        upload_to_cloud: If True, automatically upload to GCS
        bucket_name: GCS bucket name
    """
    
    print("="*80)
    print("DROPPING LEAKAGE FEATURES (KEEPING SAFE PREDICTIVE FEATURES)")
    print("="*80)
    
    # ========================================
    # 1. LOAD DATA
    # ========================================
    
    print("\n[Step 1] Loading data with targets...")
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_file}")
    
    df = pd.read_csv(input_path)
    
    print(f"   [SUCCESS] Loaded: {len(df):,} rows x {len(df.columns)} columns")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    original_columns = df.columns.tolist()
    original_shape = df.shape
    
    # Check for target columns
    target_cols = [col for col in df.columns if col.startswith('target_')]
    print(f"   Target columns found: {len(target_cols)}")
    for col in target_cols:
        print(f"      [KEEP] {col}")
    
    if not target_cols:
        print(f"   [WARNING] No target columns found!")
        print(f"      Expected: target_revenue, target_eps, etc.")
        print(f"      Run create_targets.py first!")
    
    # ========================================
    # 2. DEFINE FEATURES TO DROP (CURRENT QUARTER ONLY!)
    # ========================================
    
    print("\n[Step 2] Defining leakage features to drop...")
    print("   [INFO] Only dropping CURRENT QUARTER values")
    print("   [INFO] KEEPING all _lag_, rolling, and historical features!")
    
    FEATURES_TO_DROP = [
        # Current quarter values only
        'Revenue', 'EPS', 'Debt_to_Equity', 'net_margin', 'net_margin_q',
        'gross_margin', 'operating_margin', 'roa', 'roe', 'q_return',
        'stock_q_return', 'next_q_return',
        
        # Growth from current to last quarter
        'Revenue_growth_1q', 'Net_Income_growth_1q', 'Gross_Profit_growth_1q',
        'Operating_Income_growth_1q', 'eps_growth_1q',
        
        # Current quarter fundamentals
        'Net_Income', 'Gross_Profit', 'Operating_Income', 'EBITDA',
        'Total_Debt', 'Total_Assets', 'Total_Liabilities', 'Total_Equity',
        'Current_Assets', 'Current_Liabilities', 'Long_Term_Debt',
        'Short_Term_Debt', 'Cash', 'Current_Ratio',
        
        # Current quarter stock prices
        'Stock_Price', 'q_price', 'q_volume', 'q_high', 'q_low', 'q_open',
        'q_price_range_pct', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume',
        
        # Ratios using current quarter targets
        'pe_ratio', 'debt_to_assets', 'debt_to_ebitda', 'cash_ratio',
        
        # Engineered features using current quarter
        'revenue_acceleration', 'net_margin_trend', 'return_momentum',
        'revenue_declining', 'high_leverage', 'liquidity_risk',
        'composite_stress_score', 'leverage_x_vix', 'margin_x_market',
        'revenue_decline_x_vix', 'excess_return', 'return_vs_sector',
        'revenue_growth_vs_sector', 'debt_x_rates',
        
        # Z-scores of current targets
        'net_margin_zscore', 'roa_zscore', 'roe_zscore', 'debt_to_assets_zscore',
        
        # Log of current values
        'log_Revenue', 'log_Total_Assets', 'log_Total_Debt', 'log_Cash',
        
        # Other
        'crisis_flag', 'Quarter_End_Date', 'Original_Quarter_End',
        'Quarter_End_Date_fred', 'Company_Name',
        
        # Intermediate calculations
        'EPS_calculated', 'profit_margin_calculated', 'return_calculated',
        
        # Year-over-year growth (conservative drop)
        'Revenue_growth_4q', 'Net_Income_growth_4q', 'Gross_Profit_growth_4q',
        'Operating_Income_growth_4q', 'eps_growth_4q',
    ]
    
    # ========================================
    # 2b. VERIFY SAFE FEATURES ARE NOT IN DROP LIST
    # ========================================
    
    print(f"\n   Defined {len(FEATURES_TO_DROP)} features to drop")
    print(f"\n   [VERIFY] Checking SAFE features are NOT being dropped:")
    
    SAFE_FEATURES_THAT_SHOULD_BE_KEPT = [
        'Debt_to_Equity_lag_1q', 'Debt_to_Equity_lag_2q', 'Debt_to_Equity_lag_4q',
        'Total_Debt_lag_1q', 'Total_Debt_lag_2q', 'Total_Debt_lag_4q',
        'Total_Assets_lag_1q', 'Total_Assets_lag_2q', 'Total_Assets_lag_4q',
        'debt_to_assets_rolling4q_mean', 'debt_to_assets_rolling4q_std',
        'q_return_lag_1q', 'q_return_lag_2q', 'q_return_lag_4q',
        'q_return_rolling4q_mean', 'q_return_rolling4q_std',
        'sp500_q_return_lag_1q', 'sp500_q_return_lag_2q',
        'sector_avg_return',
        'Revenue_lag_1q', 'Revenue_lag_2q', 'Revenue_lag_4q',
        'Net_Income_lag_1q', 'Net_Income_lag_2q',
        'Current_Ratio_lag_1q', 'Current_Ratio_lag_2q',
        'Revenue_rolling4q_mean', 'Revenue_rolling4q_std',
        'Net_Income_rolling4q_mean', 'net_margin_rolling4q_mean',
        'roa_rolling4q_mean', 'roe_rolling4q_mean',
    ]
    
    accidentally_dropping = [f for f in SAFE_FEATURES_THAT_SHOULD_BE_KEPT if f in FEATURES_TO_DROP]
    
    if accidentally_dropping:
        print(f"\n   [ERROR] Safe features found in drop list!")
        for f in accidentally_dropping:
            print(f"      [FAIL] {f} - THIS SHOULD BE KEPT!")
        raise ValueError(f"Safe features in drop list: {accidentally_dropping}")
    else:
        print(f"      [SUCCESS] All {len(SAFE_FEATURES_THAT_SHOULD_BE_KEPT)} critical features will be kept")
    
    # ========================================
    # 3. CHECK WHICH FEATURES EXIST
    # ========================================
    
    print("\n[Step 3] Checking which features exist...")
    
    existing_to_drop = [
        col for col in FEATURES_TO_DROP 
        if col in df.columns and not col.startswith('target_')
    ]
    
    missing_to_drop = [col for col in FEATURES_TO_DROP if col not in df.columns]
    existing_safe = [f for f in SAFE_FEATURES_THAT_SHOULD_BE_KEPT if f in df.columns]
    missing_safe = [f for f in SAFE_FEATURES_THAT_SHOULD_BE_KEPT if f not in df.columns]
    
    print(f"   Will drop: {len(existing_to_drop)} leakage features")
    print(f"   Already absent: {len(missing_to_drop)}")
    print(f"   \n   [SUCCESS] Safe features found in data: {len(existing_safe)}/{len(SAFE_FEATURES_THAT_SHOULD_BE_KEPT)}")
    
    if missing_safe:
        print(f"   [WARNING] {len(missing_safe)} critical features missing from data:")
        for f in missing_safe[:10]:
            print(f"      - {f}")
        if len(missing_safe) > 10:
            print(f"      ... and {len(missing_safe)-10} more")
    
    if existing_to_drop:
        print(f"\n   [DROP] Dropping {len(existing_to_drop)} leakage features:")
        for i, col in enumerate(existing_to_drop[:15], 1):
            print(f"      {i}. {col}")
        if len(existing_to_drop) > 15:
            print(f"      ... and {len(existing_to_drop) - 15} more")
    
    # ========================================
    # 4. DROP FEATURES
    # ========================================
    
    print(f"\n[Step 4] Dropping leakage features (keeping targets & safe features)...")
    
    df_clean = df.drop(columns=existing_to_drop)
    
    print(f"   [SUCCESS] Dropped {len(existing_to_drop)} leakage features")
    print(f"   [SUCCESS] Kept {len(existing_safe)} safe predictive features")
    print(f"   Before: {original_shape[1]} columns")
    print(f"   After:  {df_clean.shape[1]} columns")
    
    remaining_targets = [col for col in df_clean.columns if col.startswith('target_')]
    print(f"\n   [SUCCESS] Target columns preserved: {len(remaining_targets)}")
    
    # ========================================
    # 5. VALIDATE REMAINING FEATURES
    # ========================================
    
    print(f"\n[Step 5] Validating cleaned dataset...")
    
    remaining_cols = df_clean.columns.tolist()
    identifier_cols = [c for c in remaining_cols if c in ['Date', 'Year', 'Quarter', 'Quarter_Num', 'Company', 'Sector']]
    target_features = [c for c in remaining_cols if c.startswith('target_')]
    macro_cols = [c for c in remaining_cols if any(x in c for x in ['GDP', 'CPI', 'Unemployment', 'Federal', 'Yield', 'Consumer', 'Oil', 'vix', 'sp500'])]
    lag_cols = [c for c in remaining_cols if '_lag_' in c]
    rolling_cols = [c for c in remaining_cols if 'rolling' in c]
    
    print(f"\n   [SUMMARY] Cleaned dataset composition:")
    print(f"      Identifiers:      {len(identifier_cols)}")
    print(f"      Target variables: {len(target_features)} (KEPT)")
    print(f"      Macro features:   {len(macro_cols)}")
    print(f"      Lagged features:  {len(lag_cols)} (KEPT - critical for prediction)")
    print(f"      Rolling features: {len(rolling_cols)} (KEPT - critical for prediction)")
    print(f"      TOTAL:            {len(remaining_cols)}")
    
    # ========================================
    # 6. SAVE TO NEW FILE
    # ========================================
    
    print(f"\n[Step 6] Saving cleaned dataset...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    output_size = output_path.stat().st_size / (1024*1024)
    
    print(f"   [SUCCESS] Saved to: {output_path}")
    print(f"      Size: {output_size:.1f} MB")
    print(f"      Rows: {df_clean.shape[0]:,}")
    print(f"      Columns: {df_clean.shape[1]}")
    
    # ========================================
    # 7. SAVE LOG
    # ========================================
    
    print(f"\n[Step 7] Saving log...")
    
    log_file = output_path.parent / 'dropped_features_log.txt'
    
    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DROPPED FEATURES LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Original columns: {original_shape[1]}\n")
        f.write(f"Remaining columns: {df_clean.shape[1]}\n")
        f.write(f"Dropped: {len(existing_to_drop)}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("FEATURES DROPPED:\n")
        f.write("="*80 + "\n\n")
        
        for i, col in enumerate(existing_to_drop, 1):
            f.write(f"{i}. {col}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TARGET COLUMNS PRESERVED:\n")
        f.write("="*80 + "\n\n")
        
        for col in remaining_targets:
            f.write(f"[KEEP] {col}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SAFE PREDICTIVE FEATURES KEPT:\n")
        f.write("="*80 + "\n\n")
        
        remaining_safe = [f for f in SAFE_FEATURES_THAT_SHOULD_BE_KEPT if f in df_clean.columns]
        for col in remaining_safe:
            f.write(f"[KEEP] {col}\n")
    
    print(f"   [SUCCESS] Log saved: {log_file}")
    
    # ========================================
    # 8. UPLOAD TO GCS (AUTOMATIC!)
    # ========================================
    
    if upload_to_cloud:
        # Upload cleaned file
        upload_success = upload_to_gcs(
            local_file=str(output_path),
            bucket_name=bucket_name,
            gcs_path=f"data/features/{output_path.name}"
        )
        
        # Upload log file
        if log_file.exists():
            upload_to_gcs(
                local_file=str(log_file),
                bucket_name=bucket_name,
                gcs_path=f"data/features/{log_file.name}"
            )
        
        # Also upload original file with targets (optional)
        if input_path.exists():
            upload_to_gcs(
                local_file=str(input_path),
                bucket_name=bucket_name,
                gcs_path=f"data/features/{input_path.name}"
            )
    else:
        print(f"\n[INFO] Skipping cloud upload (upload_to_cloud=False)")
    
    # ========================================
    # 9. SUMMARY
    # ========================================
    
    print(f"\n{'='*80}")
    print(f"FEATURE CLEANING SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n[INPUT FILE] (preserved):")
    print(f"   {input_file}")
    print(f"   Rows: {original_shape[0]:,}")
    print(f"   Columns: {original_shape[1]}")
    
    print(f"\n[OUTPUT FILE] (cleaned):")
    print(f"   {output_file}")
    print(f"   Rows: {df_clean.shape[0]:,}")
    print(f"   Columns: {df_clean.shape[1]}")
    
    print(f"\n[CHANGES]:")
    print(f"   Leakage features dropped: {len(existing_to_drop)}")
    print(f"   Target columns preserved: {len(remaining_targets)}")
    print(f"   Safe features kept: {len(existing_safe)}")
    
    print(f"\n[FILES]:")
    print(f"   Local cleaned: {output_file}")
    print(f"   Local log: {log_file}")
    if upload_to_cloud:
        print(f"   GCS bucket: gs://{bucket_name}/data/features/")
    
    print(f"\n{'='*80}")
    print(f"[SUCCESS] FEATURE CLEANING COMPLETE!")
    print(f"{'='*80}")
    
    return df_clean, existing_to_drop


if __name__ == "__main__":
    """
    Main execution: Drop leakage features and auto-upload to GCS
    """
    
    input_file = 'data/features/quarterly_data_with_targets.csv'
    output_file = 'data/features/quarterly_data_with_targets_clean.csv'
    
    # Set to False to skip cloud upload (useful for local testing)
    UPLOAD_TO_CLOUD = True
    BUCKET_NAME = "mlops-financial-stress-data"
    
    try:
        print(f"\n{'='*80}")
        print(f"STARTING FEATURE CLEANING")
        print(f"{'='*80}")
        print(f"\n[INPUT]  {input_file}")
        print(f"[OUTPUT] {output_file}")
        print(f"[CLOUD]  {'Enabled' if UPLOAD_TO_CLOUD else 'Disabled'}")
        
        df_clean, dropped = drop_leakage_features(
            input_file=input_file,
            output_file=output_file,
            upload_to_cloud=UPLOAD_TO_CLOUD,
            bucket_name=BUCKET_NAME
        )
        
        print(f"\n{'='*80}")
        print(f"[SUCCESS] All tasks completed!")
        print(f"{'='*80}")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] FILE NOT FOUND:")
        print(f"   {e}")
        print(f"\n[INFO] Run create_targets.py first")
        
    except Exception as e:
        print(f"\n[ERROR]:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()