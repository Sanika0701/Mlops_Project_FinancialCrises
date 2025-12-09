"""
Configuration for Financial Stress Test API
"""
from pathlib import Path

class Config:
    """Application configuration"""
    
    # GCS Settings
    GCS_BUCKET = "mlops-financial-stress-data"
    GCS_PROJECT = "ninth-iris-422916-f2"  # Update if different
    
    # Model Paths in GCS (VERIFIED)
    MODEL_PATHS = {
        "model1": {
            "type": "ensemble_vae",
            "gcs_path": "models/vae/deployment/",
            "files": {
                "model": "best_model_deployment.pkl",
                "metadata": "deployment_metadata.json"
            },
            "n_features": 72
        },
        
        "model2": {
            "type": "hybrid",
            "gcs_path": "models/predictor/",
            "targets": {
                "revenue": "revenue_best.pkl",
                "eps": "eps_best.pkl",
                "debt_equity": "debt_equity_best.pkl",
                "profit_margin": "profit_margin_best.pkl",
                "stock_return": "stock_return_best.pkl"
            },
            "n_features": 211  # 116 macro + 95 company
        },
        
        "model3": {
            "type": "one_class_svm",
            "gcs_path": "models/anomaly_detection/",
            "files": {
                "model": "model.pkl",
                "scaler": "scaler.pkl",
                "features": "features.json",
                "metadata": "model_metadata.json"
            },
            "n_features": 14
        }
    }
    
    # Data Paths in GCS
    DATA_PATHS = {
        "train_data": "data/splits/train_data.csv",
        "macro_features": "data/features/macro_features_clean.csv"
    }
    
    # Local cache directory
    LOCAL_CACHE_DIR = Path("/tmp/models")
    LOCAL_DATA_DIR = Path("/tmp/data")
    
    # API Settings
    API_TITLE = "Financial Stress Test API"
    API_VERSION = "1.0.0"
    API_PORT = 8000
    HOST = "0.0.0.0"
    
    # Scenario Generation
    DEFAULT_N_SCENARIOS = 10
    SCENARIO_SEVERITIES = {
        "baseline": {"sigma": 0.5, "count": 2},
        "adverse": {"sigma": 1.5, "count": 3},
        "severe": {"sigma": 2.5, "count": 4},
        "extreme": {"sigma": 3.5, "count": 1}
    }
    
    # Feature Mappings (VAE 72 → Model 2 116)
    VAE_TO_MODEL2_MAPPING = {
        # Core mappings
        "GDP": "GDP_last",
        "CPI": "CPI_last",
        "Unemployment_Rate": "Unemployment_Rate_last",
        "Federal_Funds_Rate": "Federal_Funds_Rate_mean",
        "VIX": "vix_q_mean",
        "SP500_Return_90D": "sp500_q_return",
        "Financial_Stress_Index": "Financial_Stress_Index_mean",
        "Corporate_Bond_Spread": "Corporate_Bond_Spread_mean",
        "High_Yield_Spread": "High_Yield_Spread_mean",
        "Oil_Price": "Oil_Price_mean",
        "TED_Spread": "TED_Spread_mean",
        "Treasury_10Y_Yield": "Treasury_10Y_Yield_mean",
        "Yield_Curve_Spread": "Yield_Curve_Spread_mean",
        "Consumer_Confidence": "Consumer_Confidence_mean",
        "Trade_Balance": "Trade_Balance_mean",
    }
    
    # Model 3 Feature Names (14 features)
    MODEL3_FEATURES = [
        "GDP_last", "CPI_last", "Unemployment_Rate_last",
        "Federal_Funds_Rate_mean", "vix_q_mean", "sp500_q_return",
        "Financial_Stress_Index_mean", "Revenue", "Net_Income",
        "Debt_to_Equity", "Current_Ratio", "net_margin", "roa", "roe"
    ]
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"