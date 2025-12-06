"""
GCS Data Loader and Uploader Utility
Handles loading data from and uploading to Google Cloud Storage

Author: Parth Saraykar
Usage:
    from src.utils.gcs_data_loader import GCSDataLoader, GCSOutputUploader
"""

import pandas as pd
from pathlib import Path
import logging
from typing import Optional
import subprocess


class GCSDataLoader:
    """Load data from GCS or local filesystem"""
    
    def __init__(self, config: dict, logger: logging.Logger):
        """
        Args:
            config: Configuration dictionary with data paths
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.source_type = config.get('source_type', 'local')
    
    def load_csv(self, data_key: str, cache_local: bool = True) -> pd.DataFrame:
        """
        Load CSV from GCS or local path
        
        Args:
            data_key: Key in config (e.g., 'input_path')
            cache_local: If True, cache GCS downloads locally
        
        Returns:
            DataFrame
        """
        if self.source_type == 'gcs':
            return self._load_from_gcs(data_key, cache_local)
        else:
            return self._load_from_local(data_key)
    
    def _load_from_local(self, data_key: str) -> pd.DataFrame:
        """Load from local filesystem"""
        
        # Handle nested config structure
        if 'data' in self.config and 'local' in self.config['data']:
            path = self.config['data']['local'].get(data_key)
        else:
            path = self.config.get(data_key)
        
        if not path:
            raise KeyError(f"Config key not found: {data_key}")
        
        self.logger.info(f"Loading from local: {path}")
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Local file not found: {path}")
        
        df = pd.read_csv(path)
        self.logger.info(f"✓ Loaded {len(df):,} rows from local file")
        
        return df
    
    def _load_from_gcs(self, data_key: str, cache_local: bool = True) -> pd.DataFrame:
        """Load from GCS bucket"""
        
        # Handle nested config structure
        if 'data' in self.config and 'gcs' in self.config['data']:
            bucket = self.config['data']['gcs'].get('bucket')
            gcs_path = self.config['data']['gcs'].get(data_key)
        else:
            bucket = self.config.get('bucket')
            gcs_path = self.config.get(data_key)
        
        if not bucket or not gcs_path:
            raise KeyError(f"GCS config not found for: {data_key}")
        
        gcs_uri = f"gs://{bucket}/{gcs_path}"
        
        self.logger.info(f"Loading from GCS: {gcs_uri}")
        
        # Try direct pandas read first
        try:
            df = pd.read_csv(gcs_uri)
            self.logger.info(f"✓ Loaded {len(df):,} rows from GCS")
            
            # Optionally cache locally
            if cache_local:
                local_path = Path("data/cache") / Path(gcs_path).name
                local_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(local_path, index=False)
                self.logger.info(f"✓ Cached to: {local_path}")
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Direct GCS read failed: {e}")
            # Fallback to gsutil
            return self._download_with_gsutil(gcs_uri, gcs_path, cache_local)
    
    def _download_with_gsutil(self, gcs_uri: str, gcs_path: str, cache_local: bool) -> pd.DataFrame:
        """Fallback: download using gsutil command"""
        
        self.logger.info(f"Attempting download with gsutil...")
        
        # Create local cache directory
        local_path = Path("data/cache") / Path(gcs_path).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download using gsutil
        try:
            cmd = ["gsutil", "cp", gcs_uri, str(local_path)]
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info(f"✓ Downloaded to: {local_path}")
            
            # Load from local cache
            df = pd.read_csv(local_path)
            self.logger.info(f"✓ Loaded {len(df):,} rows")
            
            return df
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download from GCS: {e.stderr.decode() if e.stderr else str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to download from GCS: {str(e)}")


class GCSOutputUploader:
    """Upload model outputs, reports, and artifacts to GCS"""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Get bucket name from config
        if 'data' in config and 'gcs' in config['data']:
            self.bucket = config['data']['gcs'].get('bucket')
        else:
            self.bucket = config.get('bucket')
        
        if not self.bucket:
            self.bucket = "mlops-financial-stress-data"  # Default
        
        # Check if upload is enabled
        if 'output' in config:
            self.upload_enabled = config['output'].get('upload_to_gcs', True)
        else:
            self.upload_enabled = True
    
    def upload_file(self, local_path: str, gcs_subdir: str) -> bool:
        """
        Upload single file to GCS
        
        Args:
            local_path: Local file path
            gcs_subdir: GCS subdirectory (e.g., "outputs/eda/data")
        
        Returns:
            True if successful, False otherwise
        """
        if not self.upload_enabled:
            self.logger.info("GCS upload disabled in config")
            return False
        
        local_file = Path(local_path)
        if not local_file.exists():
            self.logger.warning(f"File not found: {local_path}")
            return False
        
        gcs_uri = f"gs://{self.bucket}/{gcs_subdir}/{local_file.name}"
        
        self.logger.info(f"📤 Uploading to: {gcs_uri}")
        
        try:
            cmd = ["gsutil", "cp", str(local_path), gcs_uri]
            result = subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info(f"✓ Uploaded: {local_file.name}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Upload failed: {e.stderr.decode() if e.stderr else str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Upload failed: {str(e)}")
            return False
    
    def upload_directory(self, local_dir: str, gcs_subdir: str) -> bool:
        """
        Upload entire directory to GCS
        
        Args:
            local_dir: Local directory path
            gcs_subdir: GCS subdirectory path
        """
        if not self.upload_enabled:
            return False
        
        local_path = Path(local_dir)
        if not local_path.exists():
            self.logger.warning(f"Directory not found: {local_dir}")
            return False
        
        gcs_uri = f"gs://{self.bucket}/{gcs_subdir}/"
        
        self.logger.info(f"📤 Uploading directory to: {gcs_uri}")
        
        try:
            cmd = ["gsutil", "-m", "cp", "-r", f"{local_dir}/*", gcs_uri]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Count files
            file_count = sum(1 for _ in local_path.rglob('*') if _.is_file())
            self.logger.info(f"✓ Uploaded {file_count} files from {local_dir}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Upload failed: {e.stderr.decode() if e.stderr else str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Upload failed: {str(e)}")
            return False
    
    def upload_labeled_data(self, labeled_data_path: str):
        """Upload Snorkel labeled data to GCS"""
        self.logger.info(f"\n📤 Uploading labeled data: {labeled_data_path}")
        
        if Path(labeled_data_path).exists():
            return self.upload_file(labeled_data_path, "outputs/snorkel/data")
        else:
            self.logger.warning(f"Labeled data not found: {labeled_data_path}")
            return False
    
    def upload_model_artifacts(self, model_name: str, model_dir: str):
        """Upload all model artifacts (pkl, scaler, features, etc.)"""
        self.logger.info(f"\n📤 Uploading {model_name} artifacts...")
        
        model_path = Path(model_dir)
        if not model_path.exists():
            self.logger.warning(f"Model directory not found: {model_dir}")
            return False
        
        # Upload entire model directory
        gcs_path = f"models/model3/anomaly_detection/{model_name}"
        return self.upload_directory(str(model_path), gcs_path)
    
    def upload_ensemble(self, ensemble_dir: str = "models/Ensemble"):
        """Upload ensemble model"""
        self.logger.info("\n📤 Uploading ensemble model...")
        gcs_path = "models/model3/ensemble"
        return self.upload_directory(ensemble_dir, gcs_path)
    
    def upload_outputs(self):
        """Upload all outputs (plots, reports, results)"""
        self.logger.info("\n📤 Uploading model outputs...")
        
        output_dirs = {
            'plots': 'outputs/models/plots',
            'reports': 'outputs/models/reports',
            'results': 'outputs/models/results'
        }
        
        for output_type, local_dir in output_dirs.items():
            if Path(local_dir).exists():
                gcs_path = f"outputs/model3/{output_type}"
                self.upload_directory(local_dir, gcs_path)
        
        return True


# Convenience function
def load_data_smart(config: dict, logger: logging.Logger, data_key: str = 'input_path') -> pd.DataFrame:
    """
    Smart data loader that handles both local and GCS
    
    Usage:
        df = load_data_smart(config, logger, 'input_path')
    """
    loader = GCSDataLoader(config, logger)
    return loader.load_csv(data_key)