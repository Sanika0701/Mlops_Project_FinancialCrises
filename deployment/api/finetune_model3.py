#!/usr/bin/env python3
"""
Fine-tune Model 3 risk scoring thresholds
Use this to adjust how risk scores are calculated from anomaly scores
"""

import pandas as pd
import numpy as np
from pathlib import Path
from google.cloud import storage
import pickle
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model3FineTuner:
    """Fine-tune Model 3 risk score calibration"""
    
    def __init__(self, bucket_name: str = "mlops-financial-stress-data"):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
    def load_model3(self):
        """Load Model 3 from GCS"""
        logger.info("📥 Loading Model 3 from GCS...")
        
        # Load model
        blob = self.bucket.blob("models/anomaly_detection/model.pkl")
        model_bytes = blob.download_as_bytes()
        model = pickle.load(io.BytesIO(model_bytes))
        
        # Load scaler
        blob = self.bucket.blob("models/anomaly_detection/scaler.pkl")
        scaler_bytes = blob.download_as_bytes()
        scaler = pickle.load(io.BytesIO(scaler_bytes))
        
        # Load features
        blob = self.bucket.blob("models/anomaly_detection/features.json")
        import json
        features = json.loads(blob.download_as_text())["features"]
        
        logger.info(f"   ✓ Loaded model with {len(features)} features")
        
        return model, scaler, features
    
    def load_validation_data(self):
        """Load validation data to test thresholds"""
        logger.info("📥 Loading validation data...")
        
        # Use train_data.csv instead (which exists)
        blob = self.bucket.blob("data/splits/train_data.csv")
        csv_data = blob.download_as_text()
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Get validation set (2019-2020)
        val_df = df[(df['Year'] >= 2019) & (df['Year'] <= 2020)].copy()
        
        # Check if AT_RISK column exists, if not create from crisis_flag
        if 'AT_RISK' not in val_df.columns:
            if 'crisis_flag' in val_df.columns:
                val_df['AT_RISK'] = val_df['crisis_flag']
                logger.info("   ℹ️  Using 'crisis_flag' as AT_RISK label")
            else:
                # Create simple label based on known crisis periods
                val_df['AT_RISK'] = ((val_df['Year'] == 2020) | 
                                     ((val_df['Year'] == 2008) & (val_df['Quarter'] >= 3))).astype(int)
                logger.info("   ℹ️  Created AT_RISK labels from crisis periods")
        
        logger.info(f"   ✓ Loaded {len(val_df)} validation samples")
        logger.info(f"   ✓ At-risk rate: {val_df['AT_RISK'].sum() / len(val_df):.1%}")
        
        return val_df
    
    def analyze_score_distribution(self, model, scaler, features, val_df):
        """Analyze distribution of anomaly scores"""
        logger.info("\n" + "="*80)
        logger.info("📊 ANALYZING ANOMALY SCORE DISTRIBUTION")
        logger.info("="*80)
        
        # Prepare validation features
        X_val = val_df[features].copy()
        X_val = X_val.fillna(X_val.median())
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_val.median())
        
        if scaler:
            X_val_scaled = scaler.transform(X_val)
        else:
            X_val_scaled = X_val.values
        
        # Get predictions and scores
        predictions = model.predict(X_val_scaled)
        scores = model.decision_function(X_val_scaled)
        
        # Get true labels
        y_true = val_df['AT_RISK'].values
        
        # Analyze score distribution
        logger.info(f"\n📈 Score Statistics:")
        logger.info(f"   Min score:    {scores.min():.3f}")
        logger.info(f"   25th percentile: {np.percentile(scores, 25):.3f}")
        logger.info(f"   Median:       {np.median(scores):.3f}")
        logger.info(f"   75th percentile: {np.percentile(scores, 75):.3f}")
        logger.info(f"   Max score:    {scores.max():.3f}")
        
        logger.info(f"\n🎯 Prediction Distribution:")
        logger.info(f"   Predicted anomalies: {(predictions == -1).sum()} ({(predictions == -1).sum()/len(predictions):.1%})")
        logger.info(f"   Predicted normal:    {(predictions == 1).sum()} ({(predictions == 1).sum()/len(predictions):.1%})")
        
        logger.info(f"\n✅ Ground Truth:")
        logger.info(f"   Actually at-risk: {y_true.sum()} ({y_true.sum()/len(y_true):.1%})")
        logger.info(f"   Actually safe:    {len(y_true) - y_true.sum()} ({(len(y_true) - y_true.sum())/len(y_true):.1%})")
        
        # Score distribution by true label
        at_risk_scores = scores[y_true == 1]
        safe_scores = scores[y_true == 0]
        
        logger.info(f"\n📊 Score Distribution by True Label:")
        logger.info(f"   AT-RISK companies - Median score: {np.median(at_risk_scores):.3f}")
        logger.info(f"   SAFE companies    - Median score: {np.median(safe_scores):.3f}")
        
        return scores, predictions, y_true
    
    def test_threshold_mapping(self, scores, y_true):
        """Test different threshold mappings for risk score calculation"""
        logger.info("\n" + "="*80)
        logger.info("🔧 TESTING RISK SCORE THRESHOLDS")
        logger.info("="*80)
        
        # Current mapping
        def current_mapping(score, pred):
            if pred == -1:
                return 50 + min(abs(score) * 25, 50)
            else:
                return max(50 - abs(score) * 25, 0)
        
        # Alternative mapping 1: More aggressive
        def aggressive_mapping(score, pred):
            if pred == -1:
                return 60 + min(abs(score) * 20, 40)
            else:
                return max(40 - abs(score) * 20, 0)
        
        # Alternative mapping 2: More conservative
        def conservative_mapping(score, pred):
            if pred == -1:
                return 40 + min(abs(score) * 30, 60)
            else:
                return max(60 - abs(score) * 30, 0)
        
        predictions = np.where(scores < 0, -1, 1)
        
        # Test each mapping
        for name, mapping_func in [
            ("Current", current_mapping),
            ("Aggressive", aggressive_mapping),
            ("Conservative", conservative_mapping)
        ]:
            logger.info(f"\n{name} Mapping:")
            
            risk_scores = [mapping_func(s, p) for s, p in zip(scores, predictions)]
            
            # Statistics
            logger.info(f"   Average risk (all):     {np.mean(risk_scores):.1f}")
            logger.info(f"   Average risk (at-risk): {np.mean([r for r, y in zip(risk_scores, y_true) if y == 1]):.1f}")
            logger.info(f"   Average risk (safe):    {np.mean([r for r, y in zip(risk_scores, y_true) if y == 0]):.1f}")
            
            # Category distribution
            categories = [
                ("LOW", sum(1 for r in risk_scores if r < 25)),
                ("MODERATE", sum(1 for r in risk_scores if 25 <= r < 50)),
                ("HIGH", sum(1 for r in risk_scores if 50 <= r < 75)),
                ("CRITICAL", sum(1 for r in risk_scores if r >= 75))
            ]
            
            logger.info(f"   Category distribution:")
            for cat, count in categories:
                pct = count / len(risk_scores) * 100
                logger.info(f"      {cat:10s}: {count:4d} ({pct:5.1f}%)")
    
    def recommend_calibration(self, scores, y_true):
        """Recommend optimal threshold calibration"""
        logger.info("\n" + "="*80)
        logger.info("💡 CALIBRATION RECOMMENDATIONS")
        logger.info("="*80)
        
        # Find score that separates at-risk from safe
        at_risk_scores = scores[y_true == 1]
        safe_scores = scores[y_true == 0]
        
        median_at_risk = np.median(at_risk_scores)
        median_safe = np.median(safe_scores)
        
        logger.info(f"\nOptimal threshold around: {(median_at_risk + median_safe) / 2:.3f}")
        logger.info(f"\nRecommended risk_score calculation:")
        logger.info(f"```python")
        logger.info(f"if anomaly_pred == -1:  # Anomaly detected")
        logger.info(f"    # Map anomaly_score to 50-100 range")
        logger.info(f"    risk = 50 + min(abs(anomaly_score) * 20, 50)")
        logger.info(f"else:  # Normal")
        logger.info(f"    # Map to 0-50 range")
        logger.info(f"    risk = max(50 - abs(anomaly_score) * 20, 0)")
        logger.info(f"```")
        
        logger.info(f"\nThis would give:")
        logger.info(f"   AT-RISK companies: Avg risk ~65-85")
        logger.info(f"   SAFE companies:    Avg risk ~15-35")

def main():
    """Run fine-tuning analysis"""
    
    print("="*80)
    print("MODEL 3 FINE-TUNING ANALYSIS")
    print("="*80)
    
    tuner = Model3FineTuner()
    
    # Load model
    model, scaler, features = tuner.load_model3()
    
    # Load validation data
    val_df = tuner.load_validation_data()
    
    # Analyze distribution
    scores, predictions, y_true = tuner.analyze_score_distribution(
        model, scaler, features, val_df
    )
    
    # Test different thresholds
    tuner.test_threshold_mapping(scores, y_true)
    
    # Get recommendations
    tuner.recommend_calibration(scores, y_true)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the threshold mappings above")
    print("2. Update _calculate_risk_score() in pipeline.py if needed")
    print("3. Restart API and test")

if __name__ == "__main__":
    main()