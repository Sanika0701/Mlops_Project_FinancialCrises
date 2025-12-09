"""
FastAPI application for Financial Stress Test API
"""
import os
import sys

# CRITICAL: Set threading BEFORE any imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['LIGHTGBM_NUM_THREADS'] = '1'

# Prevent numpy threading issues
os.environ['OMP_DISPLAY_ENV'] = 'FALSE'



import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from config import Config
from model_loader import GCSModelLoader
from feature_mapper import FeatureMapper
from gcs_data_fetcher import GCSDataFetcher
from pipeline import StressTestPipeline

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description="Financial Stress Test API with ML-powered scenario generation"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models and data
MODELS = None
PIPELINE = None
DATA_FETCHER = None

# Pydantic models
class StressTestRequest(BaseModel):
    company_id: str
    scenario_ids: List[int]
    
class GenerateScenariosRequest(BaseModel):
    n_scenarios: Optional[int] = 10

@app.on_event("startup")
async def startup_event():
    """Load models and data on startup"""
    global MODELS, PIPELINE, DATA_FETCHER
    
    logger.info("="*80)
    logger.info("🚀 STARTING FINANCIAL STRESS TEST API")
    logger.info("="*80)
    
    try:
        # Import torch first to initialize it
        import torch
        logger.info(f"ℹ️  PyTorch {torch.__version__} initialized")
        
        # Set to single-threaded mode (prevents threading issues)
        torch.set_num_threads(1)
        logger.info("ℹ️  Set PyTorch to single-threaded mode")
        
        # Load models from GCS
        logger.info("\n📥 Loading models from GCS (this may take 60 seconds)...")
        model_loader = GCSModelLoader(
            bucket_name=Config.GCS_BUCKET,
            config=Config.MODEL_PATHS
        )
        MODELS = model_loader.load_all_models()
        
        # Rest of initialization...
        logger.info("\n📊 Loading company data...")
        DATA_FETCHER = GCSDataFetcher(
            bucket_name=Config.GCS_BUCKET,
            data_paths=Config.DATA_PATHS
        )
        DATA_FETCHER.load_training_data()
        
        logger.info("\n🔧 Initializing pipeline...")
        feature_mapper = FeatureMapper(Config.VAE_TO_MODEL2_MAPPING)
        
        PIPELINE = StressTestPipeline(
            models=MODELS,
            feature_mapper=feature_mapper,
            data_fetcher=DATA_FETCHER,
            config=Config
        )
        
        logger.info("\n🎲 Pre-generating scenarios...")
        PIPELINE.generate_scenarios(n_scenarios=Config.DEFAULT_N_SCENARIOS)
        
        logger.info("\n" + "="*80)
        logger.info("✅ API READY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": Config.API_TITLE,
        "version": Config.API_VERSION
    }

@app.get("/api/v1/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": MODELS is not None,
        "pipeline_ready": PIPELINE is not None,
        "n_scenarios": len(PIPELINE.scenarios) if PIPELINE else 0,
        "n_companies": len(DATA_FETCHER.company_lookup) if DATA_FETCHER else 0
    }

@app.get("/api/v1/scenarios")
async def get_scenarios():
    """Get all generated scenarios"""
    if not PIPELINE:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "scenarios": [
            {
                "scenario_id": s["scenario_id"],
                "severity": s["severity"],
                "sigma": s["sigma"],
                "preview": {
                    "GDP": s["features"].get("GDP", 0),
                    "VIX": s["features"].get("VIX", 0),
                    "Unemployment_Rate": s["features"].get("Unemployment_Rate", 0)
                }
            }
            for s in PIPELINE.scenarios
        ],
        "total": len(PIPELINE.scenarios)
    }

@app.post("/api/v1/scenarios/generate")
async def generate_scenarios(request: GenerateScenariosRequest):
    """Generate new scenarios"""
    if not PIPELINE:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    n_scenarios = request.n_scenarios if request.n_scenarios is not None else 10
    
    logger.info(f"📝 Generating {n_scenarios} NEW scenarios")
    
    try:
        # Generate NEW scenarios (replaces old ones)
        scenarios = PIPELINE.generate_scenarios(n_scenarios=n_scenarios)
        
        logger.info(f"✅ Generated {len(scenarios)} scenarios")
        
        return {
            "message": f"Generated {len(scenarios)} scenarios",
            "n_scenarios": len(scenarios),
            "scenarios": [
                {
                    "scenario_id": s["scenario_id"],
                    "severity": s["severity"],
                    "sigma": float(s["sigma"]),
                    "crisis_type": s.get("crisis_type", "Unknown")
                }
                for s in scenarios
            ]
        }
    except Exception as e:
        logger.error(f"Scenario generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/stress-test")
async def run_stress_test(request: StressTestRequest):
    """Run stress test for company across selected scenarios"""
    if not PIPELINE:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Validate company exists
        if request.company_id not in DATA_FETCHER.company_lookup:
            raise HTTPException(
                status_code=404,
                detail=f"Company {request.company_id} not found"
            )
        
        # Run stress test
        results = PIPELINE.run_stress_test(
            company_id=request.company_id,
            scenario_ids=request.scenario_ids
        )
        
        # Calculate aggregates if multiple scenarios
        if len(results) > 1:
            avg_risk = sum(r["risk_assessment"]["risk_score"] for r in results) / len(results)
            best_case = min(results, key=lambda x: x["risk_assessment"]["risk_score"])
            worst_case = max(results, key=lambda x: x["risk_assessment"]["risk_score"])
            
            return {
                "company_id": request.company_id,
                "n_scenarios": len(results),
                "aggregated": True,
                "summary": {
                    "avg_risk_score": round(avg_risk, 1),
                    "best_case": best_case,
                    "worst_case": worst_case
                },
                "detailed_results": results
            }
        else:
            return {
                "company_id": request.company_id,
                "n_scenarios": 1,
                "aggregated": False,
                "result": results[0]
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/companies")
async def list_companies():
    """List all available companies"""
    if not DATA_FETCHER:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    companies = [
        {
            "company_id": cid,
            "sector": data["sector"]
        }
        for cid, data in DATA_FETCHER.company_lookup.items()
    ]
    
    return {
        "companies": companies,
        "total": len(companies)
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.API_PORT,
        reload=False
    )