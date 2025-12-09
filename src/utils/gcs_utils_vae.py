"""
Google Cloud Storage Utilities
Helper functions for reading/writing files from/to GCS
"""

import os
import pandas as pd
import json
from pathlib import Path
from google.cloud import storage
import tempfile
import shutil

class GCSHelper:
    """Helper class for Google Cloud Storage operations"""
    
    def __init__(self, bucket_name=None):
        """
        Initialize GCS client
        
        Args:
            bucket_name: Name of GCS bucket (without gs://)
                        e.g., 'mlops-financial-stress-data'
        """
        self.client = storage.Client()
        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(bucket_name) if bucket_name else None
        
        print(f"✅ GCS Client initialized")
        if bucket_name:
            print(f"✅ Connected to bucket: gs://{bucket_name}")
    
    def parse_gcs_path(self, gcs_path):
        """
        Parse GCS path into bucket and blob name
        
        Args:
            gcs_path: Full GCS path like 'gs://bucket-name/path/to/file.csv'
        
        Returns:
            (bucket_name, blob_path)
        """
        if not gcs_path.startswith('gs://'):
            raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with gs://")
        
        path_without_prefix = gcs_path[5:]  # Remove 'gs://'
        parts = path_without_prefix.split('/', 1)
        
        bucket_name = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ''
        
        return bucket_name, blob_path
    
    def read_csv(self, gcs_path):
        """
        Read CSV from GCS
        
        Args:
            gcs_path: Full GCS path like 'gs://bucket/path/to/file.csv'
        
        Returns:
            pandas DataFrame
        """
        print(f"📥 Reading from GCS: {gcs_path}")
        
        bucket_name, blob_path = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Download to temporary file
        with tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix='.csv') as tmp_file:
            blob.download_to_filename(tmp_file.name)
            df = pd.read_csv(tmp_file.name)
        
        # Clean up temp file
        os.unlink(tmp_file.name)
        
        print(f"✅ Loaded {len(df):,} rows from GCS\n")
        return df
    
    def write_csv(self, df, gcs_path):
        """
        Write CSV to GCS
        
        Args:
            df: pandas DataFrame
            gcs_path: Full GCS path like 'gs://bucket/path/to/file.csv'
        """
        print(f"📤 Writing to GCS: {gcs_path}")
        
        bucket_name, blob_path = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Write to temporary file then upload
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            blob.upload_from_filename(tmp_file.name)
        
        # Clean up temp file
        os.unlink(tmp_file.name)
        
        print(f"✅ Uploaded to GCS: {gcs_path}\n")
    
    def write_json(self, data, gcs_path):
        """
        Write JSON to GCS
        
        Args:
            data: Python dict/list
            gcs_path: Full GCS path like 'gs://bucket/path/to/file.json'
        """
        print(f"📤 Writing JSON to GCS: {gcs_path}")
        
        bucket_name, blob_path = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Write to temporary file then upload
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            json.dump(data, tmp_file, indent=2)
            tmp_file.flush()
            blob.upload_from_filename(tmp_file.name)
        
        # Clean up temp file
        os.unlink(tmp_file.name)
        
        print(f"✅ Uploaded JSON to GCS\n")
    
    def write_file(self, local_path, gcs_path):
        """
        Upload any file to GCS
        
        Args:
            local_path: Local file path
            gcs_path: Full GCS path like 'gs://bucket/path/to/file.png'
        """
        bucket_name, blob_path = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        blob.upload_from_filename(local_path)
        print(f"✅ Uploaded: {local_path} → {gcs_path}")
    
    def upload_directory(self, local_dir, gcs_prefix):
        """
        Upload entire directory to GCS
        
        Args:
            local_dir: Local directory path (e.g., 'outputs/output_Dense_VAE_optimized')
            gcs_prefix: GCS prefix (e.g., 'gs://bucket/outputs/output_Dense_VAE_optimized')
        """
        print(f"\n📤 Uploading directory to GCS:")
        print(f"   Local:  {local_dir}")
        print(f"   GCS:    {gcs_prefix}\n")
        
        bucket_name, blob_prefix = self.parse_gcs_path(gcs_prefix)
        bucket = self.client.bucket(bucket_name)
        
        local_path = Path(local_dir)
        uploaded_count = 0
        
        # Walk through all files in directory
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                # Get relative path
                relative_path = file_path.relative_to(local_path)
                
                # Construct GCS blob path
                blob_path = f"{blob_prefix}/{relative_path}".replace('\\', '/')
                blob = bucket.blob(blob_path)
                
                # Upload
                blob.upload_from_filename(str(file_path))
                print(f"   ✓ {relative_path}")
                uploaded_count += 1
        
        print(f"\n✅ Uploaded {uploaded_count} files to gs://{bucket_name}/{blob_prefix}/\n")
    
    def file_exists(self, gcs_path):
        """Check if file exists in GCS"""
        bucket_name, blob_path = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.exists()
    
    def list_files(self, gcs_prefix):
        """
        List all files under a GCS prefix
        
        Args:
            gcs_prefix: GCS path like 'gs://bucket/outputs/'
        
        Returns:
            List of blob names
        """
        bucket_name, prefix = self.parse_gcs_path(gcs_prefix)
        bucket = self.client.bucket(bucket_name)
        
        blobs = bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]


def get_gcs_helper(bucket_name='mlops-financial-stress-data'):
    """
    Factory function to create GCS helper
    
    Args:
        bucket_name: Name of your GCS bucket (default from your path)
    
    Returns:
        GCSHelper instance
    """
    return GCSHelper(bucket_name=bucket_name)


# ============================================
# HYBRID MODE: Local + GCS
# ============================================

class HybridStorage:
    """
    Hybrid storage that works locally during training,
    then uploads to GCS at the end
    """
    
    def __init__(self, local_base_dir='outputs', gcs_base_path=None):
        """
        Args:
            local_base_dir: Local directory for temporary storage
            gcs_base_path: GCS path like 'gs://bucket/outputs'
        """
        self.local_base_dir = Path(local_base_dir)
        self.gcs_base_path = gcs_base_path
        self.gcs_helper = None
        
        if gcs_base_path:
            bucket_name, _ = GCSHelper(None).parse_gcs_path(gcs_base_path)
            self.gcs_helper = GCSHelper(bucket_name)
            print(f"✅ Hybrid mode: Local ({local_base_dir}) + GCS ({gcs_base_path})")
        else:
            print(f"✅ Local-only mode: {local_base_dir}")
    
    def get_local_path(self, relative_path):
        """Get full local path"""
        return self.local_base_dir / relative_path
    
    def save_csv(self, df, relative_path):
        """Save CSV locally (and optionally to GCS)"""
        local_path = self.get_local_path(relative_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save locally
        df.to_csv(local_path, index=False)
        print(f"✅ Saved locally: {local_path}")
        
        # Also save to GCS if configured
        if self.gcs_helper and self.gcs_base_path:
            gcs_path = f"{self.gcs_base_path}/{relative_path}".replace('\\', '/')
            self.gcs_helper.write_csv(df, gcs_path)
    
    def sync_to_gcs(self, local_subdir):
        """
        Upload entire local directory to GCS
        
        Args:
            local_subdir: Subdirectory under local_base_dir to upload
        """
        if not self.gcs_helper or not self.gcs_base_path:
            print("⚠️  GCS not configured - skipping upload")
            return
        
        local_dir = self.local_base_dir / local_subdir
        gcs_prefix = f"{self.gcs_base_path}/{local_subdir}".replace('\\', '/')
        
        self.gcs_helper.upload_directory(str(local_dir), gcs_prefix)


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    
    # Example 1: Read CSV from GCS
    print("="*70)
    print("EXAMPLE 1: Read CSV from GCS")
    print("="*70 + "\n")
    
    gcs = get_gcs_helper('mlops-financial-stress-data')
    
    df = gcs.read_csv('gs://mlops-financial-stress-data/data/features/macro_features_clean.csv')
    print(f"DataFrame shape: {df.shape}\n")
    
    # Example 2: Write CSV to GCS
    print("="*70)
    print("EXAMPLE 2: Write CSV to GCS")
    print("="*70 + "\n")
    
    test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    gcs.write_csv(test_df, 'gs://mlops-financial-stress-data/outputs/test_output.csv')
    
    # Example 3: Upload entire directory
    print("="*70)
    print("EXAMPLE 3: Upload Directory to GCS")
    print("="*70 + "\n")
    
    # Assuming you have local outputs
    gcs.upload_directory(
        local_dir='outputs/output_Dense_VAE_optimized',
        gcs_prefix='gs://mlops-financial-stress-data/outputs/output_Dense_VAE_optimized'
    )
    
    # Example 4: Hybrid mode (work locally, sync to GCS)
    print("="*70)
    print("EXAMPLE 4: Hybrid Storage")
    print("="*70 + "\n")
    
    hybrid = HybridStorage(
        local_base_dir='outputs',
        gcs_base_path='gs://mlops-financial-stress-data/outputs'
    )
    
    # Save CSV (goes to both local and GCS)
    test_df.to_csv('outputs/test.csv', index=False)
    
    # Sync entire subdirectory to GCS
    hybrid.sync_to_gcs('output_Dense_VAE_optimized')