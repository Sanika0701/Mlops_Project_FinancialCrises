"""
============================================================================
Automatic Threshold Extraction from EDA Results (WITH GCS SUPPORT)
============================================================================
Author: Parth Saraykar
Purpose: Intelligently extract thresholds from EDA outputs and generate
         Snorkel labeling function thresholds automatically
         Supports loading from GCS and uploading results to GCS
============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, Any
import yaml
import json
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.gcs_data_loader import GCSDataLoader, GCSOutputUploader


def clean_for_yaml(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for clean YAML serialization
    
    Args:
        obj: Object to convert (can be numpy type, dict, list, etc.)
        
    Returns:
        Clean Python native type
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: clean_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_yaml(i) for i in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


def load_config():
    """Load configuration"""
    with open('configs/eda_config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_eda_comparison_from_gcs_or_local(config, logger):
    """
    Load EDA comparison file from GCS or local
    
    Returns:
        DataFrame with crisis vs normal comparison
    """
    source_type = config.get('eda', {}).get('source_type', 'local')
    
    if source_type == 'gcs':
        logger.info("Loading EDA results from GCS...")
        
        try:
            from google.cloud import storage
            
            bucket_name = config['eda']['data']['gcs']['bucket']
            eda_file_path = "outputs/eda/data/crisis_vs_normal_comparison.csv"
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(eda_file_path)
            
            # Download to cache
            local_cache = Path("data/cache/crisis_vs_normal_comparison.csv")
            local_cache.parent.mkdir(parents=True, exist_ok=True)
            
            blob.download_to_filename(str(local_cache))
            logger.info(f"✓ Downloaded from GCS to: {local_cache}")
            
            df = pd.read_csv(local_cache)
            
        except Exception as e:
            logger.warning(f"GCS load failed: {e}")
            logger.info("Trying direct pandas read from GCS...")
            
            bucket_name = config['eda']['data']['gcs']['bucket']
            gcs_uri = f"gs://{bucket_name}/outputs/eda/data/crisis_vs_normal_comparison.csv"
            df = pd.read_csv(gcs_uri)
    
    else:
        # Load from local
        local_path = "outputs/eda/data/crisis_vs_normal_comparison.csv"
        logger.info(f"Loading EDA results from local: {local_path}")
        
        if not Path(local_path).exists():
            raise FileNotFoundError(
                f"EDA comparison file not found: {local_path}\n"
                "Please run EDA first: python src/eda/eda.py"
            )
        
        df = pd.read_csv(local_path)
    
    logger.info(f"✓ Loaded {len(df)} features from EDA comparison")
    return df


def upload_thresholds_to_gcs(thresholds_file, config, logger):
    """Upload extracted thresholds to GCS"""
    
    if not config.get('snorkel', {}).get('output', {}).get('upload_to_gcs', False):
        logger.info("GCS upload disabled")
        return
    
    logger.info("\n" + "="*80)
    logger.info("UPLOADING THRESHOLDS TO GCS")
    logger.info("="*80)
    
    try:
        uploader = GCSOutputUploader(config.get('snorkel', config.get('eda', {})), logger)
        
        # Upload the thresholds file
        uploader.upload_file(str(thresholds_file), "outputs/snorkel/data")
        
        logger.info(f"✓ Thresholds uploaded to GCS")
        logger.info(f"  Location: gs://{uploader.bucket}/outputs/snorkel/data/{thresholds_file.name}")
        
    except Exception as e:
        logger.warning(f"Failed to upload thresholds: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("AUTOMATIC THRESHOLD EXTRACTION WITH GCS SUPPORT")
    logger.info("="*80)
    
    try:
        # Load config
        config = load_config()
        
        # Load EDA comparison results (GCS or local)
        comparison_df = load_eda_comparison_from_gcs_or_local(config, logger)
        
        def get_feature(name):
            match = comparison_df[comparison_df['Feature'] == name]
            return match.iloc[0] if len(match) > 0 else None
        
        # Extract thresholds
        thresholds = {}
        
        # ====================================================================
        # MARKET STRESS INDICATORS
        # ====================================================================
        
        # VIX
        vix = get_feature('vix_q_mean')
        if vix is not None:
            crisis_mean = float(vix['Crisis_Mean'])
            normal_mean = float(vix['Normal_Mean'])
            crisis_std = float(vix['Crisis_Std'])
            thresholds['vix_high'] = float(max(30.0, crisis_mean - 0.3 * crisis_std))
            thresholds['vix_elevated'] = float(normal_mean + 0.5 * (crisis_mean - normal_mean))
            thresholds['vix_calm'] = float(min(20.0, normal_mean))
            logger.info(f"VIX: high={thresholds['vix_high']:.2f}, elevated={thresholds['vix_elevated']:.2f}")
        
        # Financial Stress Index
        fsi = get_feature('Financial_Stress_Index_mean')
        if fsi is not None:
            thresholds['financial_stress_high'] = float(fsi['Normal_Mean'] + 0.6 * (fsi['Crisis_Mean'] - fsi['Normal_Mean']))
            logger.info(f"FSI: high={thresholds['financial_stress_high']:.2f}")
        
        # Composite Stress
        comp = get_feature('composite_stress_score')
        if comp is not None:
            thresholds['composite_stress_high'] = float(comp['Normal_Mean'] + 0.5 * (comp['Crisis_Mean'] - comp['Normal_Mean']))
            logger.info(f"Composite Stress: high={thresholds['composite_stress_high']:.2f}")
        
        # ====================================================================
        # REVENUE & GROWTH
        # ====================================================================
        
        # Revenue 1Q
        rev1q = get_feature('Revenue_growth_1q')
        if rev1q is not None:
            crisis_mean = float(rev1q['Crisis_Mean'])
            crisis_std = float(rev1q['Crisis_Std'])
            normal_mean = float(rev1q['Normal_Mean'])
            
            if abs(crisis_mean) > 1.0:
                crisis_mean /= 100.0
                crisis_std /= 100.0
                normal_mean /= 100.0
            
            thresholds['revenue_severe_decline'] = float(crisis_mean)
            thresholds['revenue_decline_1q'] = float(crisis_mean + 0.5 * crisis_std)
            thresholds['revenue_growth_strong'] = float(max(0.10, normal_mean * 1.5))
            logger.info(f"Revenue 1Q: severe={thresholds['revenue_severe_decline']:.2%}, decline={thresholds['revenue_decline_1q']:.2%}")
        
        # Revenue 4Q
        rev4q = get_feature('Revenue_growth_4q')
        if rev4q is not None:
            crisis_mean = float(rev4q['Crisis_Mean'])
            if abs(crisis_mean) > 1.0:
                crisis_mean /= 100.0
            thresholds['revenue_decline_4q'] = float(crisis_mean + 0.5 * abs(crisis_mean))
            logger.info(f"Revenue 4Q: decline={thresholds['revenue_decline_4q']:.2%}")
        
        # ====================================================================
        # PROFITABILITY
        # ====================================================================
        
        # Net margin
        margin = get_feature('net_margin')
        if margin is not None:
            crisis_mean = float(margin['Crisis_Mean'])
            crisis_std = float(margin['Crisis_Std'])
            normal_mean = float(margin['Normal_Mean'])
            
            if abs(crisis_mean) > 1.0:
                crisis_mean /= 100.0
                crisis_std /= 100.0
                normal_mean /= 100.0
            
            thresholds['net_margin_critical'] = float(max(0.03, crisis_mean))
            thresholds['net_margin_low'] = float(max(0.05, crisis_mean + 0.5 * crisis_std))
            thresholds['net_margin_healthy'] = float(normal_mean * 1.2)
            logger.info(f"Net Margin: critical={thresholds['net_margin_critical']:.2%}, low={thresholds['net_margin_low']:.2%}")
        
        # ROA
        roa = get_feature('roa')
        if roa is not None:
            crisis_mean = float(roa['Crisis_Mean'])
            if abs(crisis_mean) > 1.0:
                crisis_mean /= 100.0
            thresholds['roa_low'] = float(max(0.01, crisis_mean + 0.5 * abs(crisis_mean)))
            logger.info(f"ROA: low={thresholds['roa_low']:.2%}")
        
        # ROE
        thresholds['roe_low'] = 0.05
        
        # ====================================================================
        # LEVERAGE
        # ====================================================================
        
        thresholds['debt_to_equity_critical'] = 3.5
        thresholds['debt_to_equity_high'] = 2.5
        thresholds['debt_to_equity_healthy'] = 1.5
        thresholds['current_ratio_low'] = 1.2
        thresholds['debt_to_assets_high'] = 0.6
        logger.info(f"Leverage: D/E high=2.5, critical=3.5")
        
        # ====================================================================
        # MACRO
        # ====================================================================
        
        # Unemployment
        unemp = get_feature('Unemployment_Rate_last')
        if unemp is not None:
            crisis_mean = float(unemp['Crisis_Mean'])
            normal_mean = float(unemp['Normal_Mean'])
            thresholds['unemployment_critical'] = float(crisis_mean)
            thresholds['unemployment_high'] = float(normal_mean + 0.6 * (crisis_mean - normal_mean))
            logger.info(f"Unemployment: critical={thresholds['unemployment_critical']:.1f}%, high={thresholds['unemployment_high']:.1f}%")
        
        # Yield curve and GDP
        thresholds['yield_curve_inverted'] = 0.0
        thresholds['gdp_decline'] = 0.0
        
        # Composite
        thresholds['perfect_storm_conditions'] = 3
        
        logger.info(f"\n✓ Extracted {len(thresholds)} thresholds")
        
        # ====================================================================
        # SAVE THRESHOLDS LOCALLY
        # ====================================================================
        
        output_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'source': 'outputs/eda/data/crisis_vs_normal_comparison.csv',
                'extraction_method': 'automatic',
                'num_thresholds': len(thresholds)
            },
            'thresholds': thresholds
        }
        
        # Clean numpy types
        output_data = clean_for_yaml(output_data)
        
        # Save locally
        output_dir = Path("outputs/snorkel/data")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        thresholds_file = output_dir / "extracted_thresholds.yaml"
        with open(thresholds_file, 'w') as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"\n✓ Saved to: {thresholds_file}")
        
        # Also save as JSON for easier parsing
        json_file = output_dir / "extracted_thresholds.json"
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"✓ Saved to: {json_file}")
        
        # ====================================================================
        # UPLOAD TO GCS (NEW)
        # ====================================================================
        
        if config.get('snorkel', {}).get('output', {}).get('upload_to_gcs', False):
            logger.info("\n" + "="*80)
            logger.info("UPLOADING THRESHOLDS TO GCS")
            logger.info("="*80)
            
            try:
                uploader = GCSOutputUploader(config.get('snorkel', {}), logger)
                
                # Upload YAML file
                uploader.upload_file(str(thresholds_file), "outputs/snorkel/data")
                logger.info("✓ Uploaded thresholds.yaml")
                
                # Upload JSON file
                uploader.upload_file(str(json_file), "outputs/snorkel/data")
                logger.info("✓ Uploaded thresholds.json")
                
                bucket = config['snorkel']['data']['gcs']['bucket']
                logger.info(f"\n✅ Thresholds uploaded to GCS!")
                logger.info(f"   Location: gs://{bucket}/outputs/snorkel/data/")
                
            except Exception as e:
                logger.warning(f"Failed to upload to GCS: {e}")
                logger.info("Continuing with local files only")
        
        # ====================================================================
        # PRINT SUMMARY
        # ====================================================================
        
        print("\n" + "="*80)
        print("EXTRACTED THRESHOLDS:")
        print("="*80)
        for key, value in sorted(thresholds.items()):
            if isinstance(value, float) and abs(value) < 1:
                print(f"  {key:35s}: {value:8.2%}")
            else:
                print(f"  {key:35s}: {value:8.4f}" if isinstance(value, float) else f"  {key:35s}: {value}")
        
        print("="*80)
        print("✓ Threshold extraction complete!")
        print("\nNext step: python src/labeling/snorkel_pipeline.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)