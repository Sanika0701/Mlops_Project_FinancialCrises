"""
Load models from GCS bucket
"""
import logging
import pickle
import json
import time
from pathlib import Path
from typing import Dict, Any
from google.cloud import storage
import io

logger = logging.getLogger(__name__)

class GCSModelLoader:
    """Load models from GCS bucket"""
    
    def __init__(self, bucket_name: str, config: Dict):
        """
        Initialize GCS model loader
        
        Args:
            bucket_name: GCS bucket name
            config: Model configuration from Config.MODEL_PATHS
        """
        self.bucket_name = bucket_name
        self.config = config
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Local cache directory
        self.cache_dir = Path("/tmp/models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, gcs_path: str, local_path: Path) -> Path:
        """Download single file from GCS"""
        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(str(local_path))
        logger.info(f"   ✓ Downloaded: {gcs_path}")
        return local_path
    
    def load_model1(self) -> Dict[str, Any]:
        """Load Model 1 (VAE Scenario Generator) - Scaler-based approach"""
        logger.info("📥 Loading Model 1: VAE Scenario Generator")
        
        config = self.config["model1"]
        gcs_base = config["gcs_path"]
        local_dir = self.cache_dir / "model1"
        local_dir.mkdir(exist_ok=True)
        
        # Download model file
        model_path = f"{gcs_base}{config['files']['model']}"
        local_model = local_dir / config['files']['model']
        self.download_file(model_path, local_model)
        
        # Load pickled dict
        with open(local_model, 'rb') as f:
            vae_dict = pickle.load(f)
        
        logger.info(f"   ✓ Loaded VAE dict with keys: {list(vae_dict.keys())}")
        
        # Extract components (NO PyTorch model reconstruction needed!)
        scaler = vae_dict.get("scaler")
        features = vae_dict.get("features", [])
        vae_config = vae_dict.get("config", {})
        generation_params = vae_dict.get("generation_params", {})
        
        logger.info(f"   ✓ Extracted scaler for {len(features)} features")
        logger.info(f"   ✓ Latent dim: {vae_config.get('latent_dim', 'unknown')}")
        
        return {
            "scaler": scaler,
            "features": features,
            "config": vae_config,
            "generation_params": generation_params,
            "state_dict": vae_dict.get("model_state_dict"),  # Keep for reference
            "type": "vae_scaler",
            "n_features": len(features)
        }
    
    def load_model2(self) -> Dict[str, Any]:
        """Load Model 2 (Predictive Models - 5 targets)"""
        logger.info("📥 Loading Model 2: Predictive Models (5 targets)")
        
        config = self.config["model2"]
        gcs_base = config["gcs_path"]
        local_dir = self.cache_dir / "model2"
        local_dir.mkdir(exist_ok=True)
        
        models = {}
        scalers = {}
        feature_names = None  # Will be same for all targets
        
        # Load each target's best model
        for target, filename in config["targets"].items():
            gcs_path = f"{gcs_base}{filename}"
            local_path = local_dir / filename
            
            self.download_file(gcs_path, local_path)
            
            with open(local_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if it's a dict (like VAE) or direct model
            if isinstance(model_data, dict):
                # Extract actual model from dict
                if 'model' in model_data:
                    models[target] = model_data['model']
                    scalers[target] = model_data.get('scaler')
                    
                    # Get feature names (should be same for all targets)
                    if feature_names is None and 'feature_names' in model_data:
                        feature_names = model_data['feature_names']
                        logger.info(f"   ✓ Found {len(feature_names)} feature names")
                    
                    logger.info(f"   ✓ Loaded {target} model (from dict)")
                else:
                    logger.error(f"   ❌ {target}: dict has keys {list(model_data.keys())}, no 'model' key")
                    models[target] = None
            else:
                # Direct model object
                models[target] = model_data
                scalers[target] = None
                logger.info(f"   ✓ Loaded {target} model (direct)")
        
        return {
            "models": models,
            "scalers": scalers,
            "feature_names": feature_names,  # ← ADD THIS
            "type": config["type"],
            "n_features": config["n_features"]
        }
    
    def load_model3(self) -> Dict[str, Any]:
        """Load Model 3 (Anomaly Detection)"""
        logger.info("📥 Loading Model 3: Anomaly Detection")
        
        config = self.config["model3"]
        gcs_base = config["gcs_path"]
        local_dir = self.cache_dir / "model3"
        local_dir.mkdir(exist_ok=True)
        
        # Download model
        model_path = f"{gcs_base}{config['files']['model']}"
        local_model = local_dir / config['files']['model']
        self.download_file(model_path, local_model)
        
        with open(local_model, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler = None
        scaler_path = f"{gcs_base}{config['files']['scaler']}"
        local_scaler = local_dir / config['files']['scaler']
        try:
            self.download_file(scaler_path, local_scaler)
            with open(local_scaler, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"   ✓ Loaded scaler")
        except Exception as e:
            logger.warning(f"   ⚠️  No scaler found")
        
        # Load features
        features_path = f"{gcs_base}{config['files']['features']}"
        local_features = local_dir / config['files']['features']
        self.download_file(features_path, local_features)
        
        with open(local_features, 'r') as f:
            features_data = json.load(f)
            features = features_data.get("features", [])
        
        logger.info(f"   ✓ Loaded model ({len(features)} features)")
        
        return {
            "model": model,
            "scaler": scaler,
            "features": features,
            "type": config["type"],
            "n_features": len(features)
        }
    
    def load_all_models(self) -> Dict[str, Any]:
        """Load all three models SEQUENTIALLY with delays"""
        logger.info("="*80)
        logger.info("🚀 LOADING ALL MODELS FROM GCS")
        logger.info("="*80)
        
        try:
            # Load Model 1 (VAE)
            logger.info("\n⏳ Loading Model 1...")
            model1 = self.load_model1()
            time.sleep(1)
            logger.info("✅ Model 1 loaded and stabilized")
            
            # Load Model 2 (Predictive)
            logger.info("\n⏳ Loading Model 2...")
            model2 = self.load_model2()
            time.sleep(1)
            logger.info("✅ Model 2 loaded")
            
            # Load Model 3 (Anomaly)
            logger.info("\n⏳ Loading Model 3...")
            model3 = self.load_model3()
            logger.info("✅ Model 3 loaded")
            
            logger.info("\n" + "="*80)
            logger.info("✅ ALL MODELS LOADED SUCCESSFULLY")
            logger.info("="*80)
            
            return {
                "model1": model1,
                "model2": model2,
                "model3": model3
            }
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise