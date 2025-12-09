"""
STEP 5: BIAS DETECTION WITH EXPLICIT DATA SLICING - WITH GCS

FIXED FOR MODIFIED PIPELINE:
- 50 companies (was 25)
- Date range: 1990-2025 (was 2005-2025)
- Quarterly data (was daily)
- Updated temporal periods
- Fixed SliceAnalyzer bug
- GCS integration for input/output

Implements data slicing for bias detection as required by MLOps assignment.

Data Slicing Methodology:
- Creates meaningful slices of data based on categorical features
- Analyzes each slice independently  
- Compares slices to detect bias
- Generates mitigation strategies

Slicing Dimensions:
1. Company (50 companies)
2. Sector (categorical grouping)
3. Time Period (1990-2025, multiple crisis periods)
4. Market Regime (volatility-based slicing)

Usage:
    python step5_bias_detection_with_explicit_slicing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import json
from scipy import stats
from google.cloud import storage
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATA SLICER - Creates slices from dataset
# ============================================================================

class DataSlicer:
    """Creates slices of data for bias analysis."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.slices = {}
    
    def create_slices_by_feature(self, feature: str) -> Dict[str, pd.DataFrame]:
        """Create slices based on a categorical feature."""
        logger.info(f"\n   Creating slices by '{feature}'...")
        
        if feature not in self.df.columns:
            logger.warning(f"   ⚠️  Feature '{feature}' not found")
            return {}
        
        slices = {}
        unique_values = self.df[feature].unique()
        
        for value in unique_values:
            slice_key = f"{feature}={value}"
            slice_df = self.df[self.df[feature] == value].copy()
            slices[slice_key] = slice_df
        
        logger.info(f"   ✓ Created {len(slices)} slices")
        
        self.slices[feature] = slices
        return slices
    
    def create_temporal_slices(self, periods: Dict[str, Tuple[int, int]]) -> Dict[str, pd.DataFrame]:
        """Create slices based on time periods."""
        logger.info(f"\n   Creating temporal slices...")
        
        if 'Date' not in self.df.columns:
            logger.warning(f"   ⚠️  No Date column")
            return {}
        
        # Ensure Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'], format='mixed', errors='coerce')
        
        self.df['Year'] = self.df['Date'].dt.year
        slices = {}
        
        for period_name, (start_year, end_year) in periods.items():
            period_mask = (self.df['Year'] >= start_year) & (self.df['Year'] <= end_year)
            slice_df = self.df[period_mask].copy()
            slices[period_name] = slice_df
        
        logger.info(f"   ✓ Created {len(slices)} temporal slices")
        
        self.slices['temporal'] = slices
        return slices
    
    def create_regime_slices(self, regime_column: str = 'VIX_Regime') -> Dict[str, pd.DataFrame]:
        """Create slices based on market regime."""
        logger.info(f"\n   Creating market regime slices...")
        
        if regime_column in self.df.columns:
            return self.create_slices_by_feature(regime_column)
        elif 'VIX' in self.df.columns:
            # Create regime from VIX
            logger.info(f"   Creating regime from VIX values...")
            
            self.df['Market_Regime'] = pd.cut(
                self.df['VIX'], 
                bins=[0, 15, 25, 100],
                labels=['Low_Vol', 'Medium_Vol', 'High_Vol']
            )
            
            return self.create_slices_by_feature('Market_Regime')
        else:
            logger.warning(f"   ⚠️  Cannot create regime slices - no VIX column")
            return {}
    
    def get_all_slices(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Get all created slices."""
        return self.slices


# ============================================================================
# SLICE ANALYZER - Analyzes each slice for bias
# ============================================================================

class SliceAnalyzer:
    """Analyzes slices to detect bias."""
    
    def __init__(self):
        self.slice_metrics = {}
    
    def analyze_slice(self, slice_name: str, slice_df: pd.DataFrame) -> Dict:
        """
        Analyze a single slice.
        
        Computes:
        - Sample size
        - Missing value percentage
        - Key feature statistics
        - Data quality metrics
        """
        metrics = {
            'slice_name': str(slice_name),
            'n_samples': int(len(slice_df))
        }
        
        # Handle empty slices
        if len(slice_df) == 0:
            metrics['missing_pct'] = 0.0
            metrics['date_range_days'] = 0
            return metrics
        
        # Calculate missing percentage (handle division by zero)
        total_elements = slice_df.size
        if total_elements > 0:
            missing_count = slice_df.isna().sum().sum()
            metrics['missing_pct'] = float(round((missing_count / total_elements) * 100, 2))
        else:
            metrics['missing_pct'] = 0.0
        
        # Add feature statistics
        numeric_cols = slice_df.select_dtypes(include=[np.number]).columns
        
        # Use Stock_Price instead of Stock_Return_1D (we don't have derived features)
        if 'Stock_Price' in numeric_cols:
            stock_mean = slice_df['Stock_Price'].mean()
            stock_std = slice_df['Stock_Price'].std()
            if not pd.isna(stock_mean):
                metrics['stock_price_mean'] = float(round(stock_mean, 2))
            if not pd.isna(stock_std):
                metrics['stock_price_std'] = float(round(stock_std, 2))
        
        if 'Revenue' in numeric_cols:
            revenue_mean = slice_df['Revenue'].mean()
            if not pd.isna(revenue_mean):
                metrics['revenue_mean'] = float(revenue_mean)
        
        # Handle date range calculation
        if 'Date' in slice_df.columns:
            date_min = slice_df['Date'].min()
            date_max = slice_df['Date'].max()
            
            if pd.notna(date_min) and pd.notna(date_max):
                metrics['date_range_days'] = int((date_max - date_min).days)
            else:
                metrics['date_range_days'] = 0
        
        return metrics
    
    def compare_slices(self, slices: Dict[str, pd.DataFrame], 
                       feature: str = 'Stock_Price') -> List[Dict]:
        """
        Compare feature distributions across slices.
        
        Uses Kolmogorov-Smirnov test to detect distribution differences.
        """
        comparisons = []
        
        # Check if feature exists in any slice
        feature_exists = False
        for slice_df in slices.values():
            if feature in slice_df.columns:
                feature_exists = True
                break
        
        if not feature_exists:
            logger.warning(f"   ⚠️  Feature '{feature}' not found in slices")
            return comparisons
        
        slice_names = list(slices.keys())
        
        # Compare each pair of slices
        for i in range(len(slice_names)):
            for j in range(i + 1, len(slice_names)):
                slice1_name = slice_names[i]
                slice2_name = slice_names[j]
                
                if feature not in slices[slice1_name].columns or feature not in slices[slice2_name].columns:
                    continue
                
                slice1_data = slices[slice1_name][feature].dropna()
                slice2_data = slices[slice2_name][feature].dropna()
                
                if len(slice1_data) < 30 or len(slice2_data) < 30:
                    continue
                
                # KS test
                ks_stat, p_value = stats.ks_2samp(slice1_data, slice2_data)
                
                if p_value < 0.05:  # Significant difference
                    comparisons.append({
                        'feature': str(feature),
                        'slice1': str(slice1_name),
                        'slice2': str(slice2_name),
                        'ks_statistic': float(ks_stat),
                        'p_value': float(p_value),
                        'mean_diff': float(slice1_data.mean() - slice2_data.mean())
                    })
        
        return comparisons


# ============================================================================
# BIAS DETECTOR WITH EXPLICIT SLICING AND GCS
# ============================================================================

class BiasDetectorWithSlicing:
    """Bias detector with explicit data slicing implementation and GCS integration."""
    
    def __init__(self, dataset_name: str, bucket_name: str = 'mlops-financial-stress-data', use_gcs: bool = True):
        self.dataset_name = dataset_name
        self.bucket_name = bucket_name
        self.use_gcs = use_gcs
        
        self.bias_report = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'slicing_summary': {},
            'biases_detected': [],
            'slice_comparisons': [],
            'mitigation_recommendations': []
        }
        
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
    
    def upload_to_gcs(self, content: str, filename: str, subfolder: str = 'data/bias_reports'):
        """Upload text content to GCS."""
        if not self.use_gcs:
            return False
        
        try:
            blob_path = f"{subfolder}/{filename}"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(content, content_type='application/json')
            
            logger.info(f"  ✓ Uploaded to GCS: gs://{self.bucket_name}/{blob_path}")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ GCS upload failed: {e}")
            return False
    
    def upload_csv_to_gcs(self, df: pd.DataFrame, filename: str, subfolder: str = 'data/bias_reports'):
        """Upload DataFrame to GCS as CSV."""
        if not self.use_gcs:
            return False
        
        try:
            blob_path = f"{subfolder}/{filename}"
            blob = self.bucket.blob(blob_path)
            
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
            
            logger.info(f"  ✓ Uploaded to GCS: gs://{self.bucket_name}/{blob_path}")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ GCS upload failed: {e}")
            return False
    
    def run_bias_detection(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Run complete bias detection with data slicing."""
        logger.info("\n" + "="*80)
        logger.info("BIAS DETECTION WITH DATA SLICING")
        logger.info("="*80)
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"GCS Enabled: {self.use_gcs}")
        logger.info("="*80)
        
        # === STEP 1: CREATE SLICES ===
        logger.info("\n" + "="*80)
        logger.info("STEP 1: CREATING DATA SLICES")
        logger.info("="*80)
        
        slicer = DataSlicer(df)
        
        # Slice by Company
        company_slices = slicer.create_slices_by_feature('Company')
        logger.info(f"   ✓ Company slices: {len(company_slices)}")
        
        # Slice by Sector
        sector_slices = slicer.create_slices_by_feature('Sector')
        logger.info(f"   ✓ Sector slices: {len(sector_slices)}")
        
        # UPDATED: Temporal periods for 1990-2025
        periods = {
            'Early Years (1990-1999)': (1990, 1999),
            'Dot-com (2000-2002)': (2000, 2002),
            'Pre-Crisis (2003-2007)': (2003, 2007),
            'Financial Crisis (2008-2009)': (2008, 2009),
            'Recovery (2010-2019)': (2010, 2019),
            'COVID (2020-2021)': (2020, 2021),
            'Recent (2022-2025)': (2022, 2025)
        }
        temporal_slices = slicer.create_temporal_slices(periods)
        logger.info(f"   ✓ Temporal slices: {len(temporal_slices)}")
        
        # Slice by Market Regime
        regime_slices = slicer.create_regime_slices()
        logger.info(f"   ✓ Market regime slices: {len(regime_slices)}")
        
        # === STEP 2: ANALYZE EACH SLICE ===
        logger.info("\n" + "="*80)
        logger.info("STEP 2: ANALYZING SLICES")
        logger.info("="*80)
        
        analyzer = SliceAnalyzer()
        
        # Analyze company slices
        logger.info("\n[A] Company Slice Analysis (50 companies):")
        company_metrics = []
        
        for slice_name, slice_df in company_slices.items():
            metrics = analyzer.analyze_slice(slice_name, slice_df)
            company_metrics.append(metrics)
        
        # Show top/bottom companies by sample size
        company_df = pd.DataFrame(company_metrics)
        company_df = company_df.sort_values('n_samples', ascending=False)
        
        logger.info("\n   Top 5 companies by sample size:")
        print(company_df.head(5).to_string(index=False))
        
        logger.info("\n   Bottom 5 companies by sample size:")
        print(company_df.tail(5).to_string(index=False))
        
        self.bias_report['slicing_summary']['company_slices'] = company_metrics
        
        # Analyze sector slices
        logger.info("\n[B] Sector Slice Analysis:")
        sector_metrics = []
        
        for slice_name, slice_df in sector_slices.items():
            metrics = analyzer.analyze_slice(slice_name, slice_df)
            sector_metrics.append(metrics)
        
        sector_df = pd.DataFrame(sector_metrics)
        print(sector_df.to_string(index=False))
        
        self.bias_report['slicing_summary']['sector_slices'] = sector_metrics
        
        # Analyze temporal slices
        logger.info("\n[C] Temporal Slice Analysis:")
        temporal_metrics = []
        
        for slice_name, slice_df in temporal_slices.items():
            metrics = analyzer.analyze_slice(slice_name, slice_df)
            temporal_metrics.append(metrics)
        
        temporal_df = pd.DataFrame(temporal_metrics)
        print(temporal_df.to_string(index=False))
        
        self.bias_report['slicing_summary']['temporal_slices'] = temporal_metrics
        
        # === STEP 3: DETECT BIAS ACROSS SLICES ===
        logger.info("\n" + "="*80)
        logger.info("STEP 3: DETECTING BIAS ACROSS SLICES")
        logger.info("="*80)
        
        biases = self._detect_bias_from_slices(company_metrics, sector_metrics, temporal_metrics)
        
        # === STEP 4: GENERATE RECOMMENDATIONS ===
        logger.info("\n" + "="*80)
        logger.info("STEP 4: MITIGATION RECOMMENDATIONS")
        logger.info("="*80)
        
        recommendations = self._generate_mitigation_recommendations(biases)
        
        # === SUMMARY ===
        self._print_summary()
        
        # Save report (local + GCS)
        self._save_report()
        
        return df, self.bias_report
    
    def _detect_bias_from_slices(self, company_metrics: List[Dict], 
                                  sector_metrics: List[Dict],
                                  temporal_metrics: List[Dict]) -> List[Dict]:
        """Detect bias by comparing slice metrics."""
        
        biases = []
        
        # === BIAS 1: Representation Bias (UPDATED for 50 companies) ===
        logger.info("\n   [A] Checking representation bias (50 companies)...")
        
        total_samples = sum(m['n_samples'] for m in company_metrics)
        expected_per_company = total_samples / len(company_metrics)
        
        underrep = []
        overrep = []
        
        for metrics in company_metrics:
            samples = metrics['n_samples']
            deviation = (samples - expected_per_company) / expected_per_company
            
            # UPDATED: More lenient for quarterly data (40% vs 30%)
            if deviation < -0.4:  # 40% below expected
                underrep.append(metrics['slice_name'])
            elif deviation > 0.4:  # 40% above expected
                overrep.append(metrics['slice_name'])
        
        if underrep or overrep:
            logger.warning(f"   ⚠️  Representation bias detected:")
            logger.warning(f"      Underrepresented: {len(underrep)} companies")
            logger.warning(f"      Overrepresented: {len(overrep)} companies")
            
            biases.append({
                'type': 'Representation Bias',
                'dimension': 'Company',
                'severity': 'HIGH',
                'underrepresented': underrep,
                'overrepresented': overrep
            })
        else:
            logger.info(f"   ✓ No representation bias (within 40% deviation)")
        
        # === BIAS 2: Sector Imbalance ===
        logger.info("\n   [B] Checking sector imbalance...")
        
        sector_samples = [m['n_samples'] for m in sector_metrics]
        max_samples = max(sector_samples) if sector_samples else 0
        min_samples = min(sector_samples) if sector_samples else 0
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
        if imbalance_ratio > 5:
            logger.warning(f"   ⚠️  Sector imbalance: {imbalance_ratio:.1f}x difference")
            
            biases.append({
                'type': 'Sector Imbalance',
                'severity': 'MEDIUM',
                'imbalance_ratio': float(imbalance_ratio)
            })
        else:
            logger.info(f"   ✓ Sector balance OK ({imbalance_ratio:.1f}x)")
        
        # === BIAS 3: Temporal Data Quality Bias (UPDATED threshold for quarterly) ===
        logger.info("\n   [C] Checking temporal data quality bias...")
        
        biases_found = False
        for metrics in temporal_metrics:
            # UPDATED: More lenient for quarterly data (30% vs 5%)
            if metrics['missing_pct'] > 30:
                logger.warning(f"   ⚠️  {metrics['slice_name']}: {metrics['missing_pct']:.1f}% missing")
                
                biases.append({
                    'type': 'Temporal Quality Bias',
                    'period': metrics['slice_name'],
                    'severity': 'MEDIUM',
                    'missing_pct': float(metrics['missing_pct'])
                })
                biases_found = True
        
        if not biases_found:
            logger.info(f"   ✓ No temporal quality bias (all periods < 30% missing)")
        
        self.bias_report['biases_detected'] = biases
        
        return biases
    
    def _generate_mitigation_recommendations(self, biases: List[Dict]) -> List[str]:
        """Generate mitigation recommendations based on detected biases."""
        
        recommendations = []
        
        if not biases:
            logger.info("\n   ✓ No biases detected - no mitigation needed")
            recommendations.append("No bias mitigation required - data is well-balanced")
            return recommendations
        
        # Check for representation bias
        rep_biases = [b for b in biases if 'Representation' in b['type']]
        if rep_biases:
            logger.info("\n   📋 Recommendation 1: Stratified Sampling")
            logger.info("      Use stratified train/test split to ensure all companies represented")
            recommendations.append("Stratified sampling by Company in train/test split")
            
            logger.info("\n   📋 Recommendation 2: Weighted Loss Function")
            logger.info("      Apply sample weights inversely proportional to company size")
            recommendations.append("Weighted loss: weight = 1 / company_sample_count")
        
        # Check for sector imbalance
        sector_biases = [b for b in biases if 'Sector' in b['type']]
        if sector_biases:
            logger.info("\n   📋 Recommendation 3: Sector Stratification")
            logger.info("      Ensure each sector proportionally in train/val/test")
            recommendations.append("Stratified split by Sector")
        
        # Check for temporal bias
        temporal_biases = [b for b in biases if 'Temporal' in b['type']]
        if temporal_biases:
            logger.info("\n   📋 Recommendation 4: Crisis Data Handling")
            logger.info("      Ensure crisis periods (1990s, 2000-2002, 2008-2009, 2020) in validation")
            recommendations.append("Include all crisis periods in validation for realistic stress testing")
        
        self.bias_report['mitigation_recommendations'] = recommendations
        
        return recommendations
    
    def _print_summary(self):
        """Print bias detection summary."""
        logger.info("\n" + "="*80)
        logger.info("BIAS DETECTION SUMMARY")
        logger.info("="*80)
        
        # Count slices created
        total_slices = 0
        for dimension, metrics in self.bias_report['slicing_summary'].items():
            total_slices += len(metrics)
        
        logger.info(f"\n📊 Data Slicing:")
        logger.info(f"   Total slices created: {total_slices}")
        for dimension, metrics in self.bias_report['slicing_summary'].items():
            logger.info(f"   {dimension}: {len(metrics)} slices")
        
        # Count biases
        biases = self.bias_report.get('biases_detected', [])
        logger.info(f"\n📊 Biases Detected: {len(biases)}")
        
        for bias in biases:
            severity_icon = "🚨" if bias['severity'] == 'CRITICAL' else "⚠️" if bias['severity'] == 'HIGH' else "ℹ️"
            logger.info(f"   {severity_icon} {bias['type']} ({bias['severity']})")
        
        # Print recommendations
        recommendations = self.bias_report.get('mitigation_recommendations', [])
        if recommendations:
            logger.info(f"\n💡 Mitigation Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"   {i}. {rec}")
    
    def _save_report(self):
        """Save bias detection report locally and to GCS."""
        output_dir = Path("data/bias_reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Custom JSON encoder
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif pd.isna(obj):
                    return None
                return super().default(obj)
        
        # Save JSON report locally
        json_path = output_dir / f"bias_report_{self.dataset_name}_{timestamp}.json"
        
        with open(json_path, 'w') as f:
            json.dump(self.bias_report, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"\n💾 Bias report saved locally: {json_path}")
        
        # Upload JSON to GCS
        if self.use_gcs:
            json_content = json.dumps(self.bias_report, indent=2, cls=NumpyEncoder)
            self.upload_to_gcs(json_content, f"bias_report_{self.dataset_name}_{timestamp}.json")
        
        # Save slice statistics as CSV locally and to GCS
        for slice_type, metrics in self.bias_report['slicing_summary'].items():
            if metrics:
                csv_filename = f"{slice_type}_statistics_{timestamp}.csv"
                csv_path = output_dir / csv_filename
                metrics_df = pd.DataFrame(metrics)
                metrics_df.to_csv(csv_path, index=False)
                logger.info(f"💾 {slice_type} statistics saved locally: {csv_path}")
                
                # Upload CSV to GCS
                if self.use_gcs:
                    self.upload_csv_to_gcs(metrics_df, csv_filename)


def main():
    """Execute bias detection with data slicing and GCS integration."""
    
    # GCS configuration
    bucket_name = 'mlops-financial-stress-data'
    use_gcs = True
    
    # Initialize GCS client and download data
    if use_gcs:
        try:
            logger.info(f"Connecting to GCS bucket: gs://{bucket_name}/")
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            
            # Download features_engineered.csv from GCS
            blob_path = "data/processed/features_engineered.csv"
            blob = bucket.blob(blob_path)
            
            local_dir = Path("data/processed")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / "features_engineered.csv"
            
            logger.info(f"Downloading: gs://{bucket_name}/{blob_path}")
            blob.download_to_filename(str(local_path))
            
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Downloaded: {local_path} ({size_mb:.2f} MB)")
            
            filepath = local_path
            
        except Exception as e:
            logger.warning(f"⚠️  GCS download failed: {e}")
            logger.info("Falling back to local file...")
            use_gcs = False
            filepath = Path("data/processed/features_engineered.csv")
    else:
        filepath = Path("data/processed/features_engineered.csv")
    
    if not filepath.exists():
        logger.error("❌ No features_engineered.csv found!")
        logger.error("Run feature engineering pipeline first")
        return
    
    logger.info(f"Loading: {filepath}")
    df = pd.read_csv(filepath)

    # Ensure Date is datetime
    if 'Date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            logger.info(f"   ✓ Converted Date to datetime: {df['Date'].dtype}")
    
    # Run bias detection
    detector = BiasDetectorWithSlicing(
        dataset_name=filepath.stem,
        bucket_name=bucket_name,
        use_gcs=use_gcs
    )
    df_analyzed, report = detector.run_bias_detection(df)
    
    # Final status
    logger.info("\n" + "="*80)
    logger.info("✅ BIAS DETECTION WITH DATA SLICING COMPLETE")
    logger.info("="*80)
    
    total_biases = len(report.get('biases_detected', []))
    
    if total_biases == 0:
        logger.info("\n✅ No significant biases detected")
        logger.info("✅ Data is well-balanced across slices")
    else:
        logger.warning(f"\n⚠️  {total_biases} biases detected")
        logger.warning("   Review bias report and apply mitigation strategies")
    
    logger.info(f"\n📁 Reports saved:")
    logger.info(f"   Local: data/bias_reports/")
    if use_gcs:
        logger.info(f"   GCS: gs://{bucket_name}/data/bias_reports/")
    
    logger.info("\n➡️  Next: Apply stratified sampling in model training")


if __name__ == "__main__":
    main()