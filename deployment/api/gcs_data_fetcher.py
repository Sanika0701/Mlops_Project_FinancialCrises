"""
Fetch company data and training data from GCS
"""
import logging
import pandas as pd
import io
from google.cloud import storage
from typing import Dict

logger = logging.getLogger(__name__)

class GCSDataFetcher:
    """Fetch data from GCS bucket"""
    
    def __init__(self, bucket_name: str, data_paths: Dict):
        self.bucket_name = bucket_name
        self.data_paths = data_paths
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Cached data
        self.train_data = None
        self.company_lookup = None
        
    def load_training_data(self):
        """Load training data into memory"""
        logger.info("📥 Loading training data from GCS...")
        
        blob = self.bucket.blob(self.data_paths["train_data"])
        csv_data = blob.download_as_text()
        self.train_data = pd.read_csv(io.StringIO(csv_data))
        
        logger.info(f"   ✓ Loaded {len(self.train_data)} rows")
        
        # Build company lookup
        self._build_company_lookup()
        
    def _build_company_lookup(self):
        """Build fast lookup dictionary for companies"""
        logger.info("📊 Building company lookup index...")
        
        self.company_lookup = {}
        
        for company_id in self.train_data['Company'].unique():
            company_df = self.train_data[self.train_data['Company'] == company_id].sort_values('Date')
            
            # Latest quarter
            latest = company_df.iloc[-1]
            
            # Historical (last 8 quarters)
            historical = company_df.tail(8)
            
            self.company_lookup[company_id] = {
                "latest": latest,
                "historical": historical,
                "sector": latest['Sector']
            }
        
        logger.info(f"   ✓ Indexed {len(self.company_lookup)} companies")
    
    def get_company_data(self, company_id: str) -> Dict:
        """Get company data for inference"""
        if self.company_lookup is None:
            self.load_training_data()
        
        if company_id not in self.company_lookup:
            raise ValueError(f"Company {company_id} not found in database")
        
        return self.company_lookup[company_id]
    
    def get_latest_macro_features(self) -> pd.Series:
        """Get latest macro features from training data"""
        if self.train_data is None:
            self.load_training_data()
        
        # Get most recent row (any company will have same macro features)
        latest = self.train_data.sort_values('Date').iloc[-1]
        
        return latest