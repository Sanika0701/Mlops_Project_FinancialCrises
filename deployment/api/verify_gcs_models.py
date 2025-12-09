"""
Verify all models exist in GCS before deployment
"""
from google.cloud import storage
from config import Config

def verify_gcs_structure():
    """Check if all required model files exist in GCS"""
    
    client = storage.Client()
    bucket = client.bucket(Config.GCS_BUCKET)
    
    print("="*80)
    print("🔍 VERIFYING GCS MODEL STRUCTURE")
    print(f"📦 Bucket: gs://{Config.GCS_BUCKET}")
    print("="*80)
    
    all_good = True
    
    # Check Model 1 (VAE)
    print("\n📦 MODEL 1: VAE Scenario Generator")
    model1_config = Config.MODEL_PATHS["model1"]
    base_path = model1_config["gcs_path"]
    
    for key, filename in model1_config["files"].items():
        full_path = f"{base_path}{filename}"
        blob = bucket.blob(full_path)
        
        if blob.exists():
            blob.reload()
            size_mb = blob.size / 1024 / 1024 if blob.size else 0
            print(f"   ✅ {full_path} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ MISSING: {full_path}")
            all_good = False
    
    # Check Model 2 (Predictors)
    print("\n📦 MODEL 2: Predictive Models")
    model2_config = Config.MODEL_PATHS["model2"]
    base_path = model2_config["gcs_path"]
    
    for target, filename in model2_config["targets"].items():
        full_path = f"{base_path}{filename}"
        blob = bucket.blob(full_path)
        
        if blob.exists():
            blob.reload()
            size_mb = blob.size / 1024 / 1024 if blob.size else 0
            print(f"   ✅ {target:15s}: {filename} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ MISSING {target:15s}: {full_path}")
            all_good = False
    
    # Check Model 3 (Anomaly)
    print("\n📦 MODEL 3: Anomaly Detection")
    model3_config = Config.MODEL_PATHS["model3"]
    base_path = model3_config["gcs_path"]
    
    for key, filename in model3_config["files"].items():
        full_path = f"{base_path}{filename}"
        blob = bucket.blob(full_path)
        
        if blob.exists():
            blob.reload()
            size_mb = blob.size / 1024 if blob.size else 0
            print(f"   ✅ {key:12s}: {filename} ({size_mb:.1f} KB)")
        else:
            print(f"   ⚠️  {key:12s}: {full_path} (not found)")
    
    # Summary
    print("\n" + "="*80)
    if all_good:
        print("✅ ALL REQUIRED MODELS VERIFIED!")
        print("🚀 Ready to deploy API")
    else:
        print("❌ SOME MODELS MISSING - Check paths above")
    print("="*80)
    
    return all_good

if __name__ == "__main__":
    verify_gcs_structure()