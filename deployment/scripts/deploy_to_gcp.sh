#!/bin/bash

# Financial Stress Test API - GCP Cloud Run Deployment
# Usage: ./deploy_to_gcp.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="mlops-434900"  # Update with your project ID
REGION="us-central1"
SERVICE_NAME="financial-stress-test-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "================================================================================"
echo "🚀 DEPLOYING FINANCIAL STRESS TEST API TO GCP CLOUD RUN"
echo "================================================================================"
echo ""
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo ""

# Step 1: Check prerequisites
echo "📋 Step 1: Checking prerequisites..."

if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Please install: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"
echo ""

# Step 2: Set GCP project
echo "📋 Step 2: Setting GCP project..."
gcloud config set project ${PROJECT_ID}
echo -e "${GREEN}✅ Project set${NC}"
echo ""

# Step 3: Enable required APIs
echo "📋 Step 3: Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
echo -e "${GREEN}✅ APIs enabled${NC}"
echo ""

# Step 4: Build Docker image
echo "📋 Step 4: Building Docker image..."
cd ../..  # Go to project root

docker build -f deployment/docker/Dockerfile.api -t ${IMAGE_NAME}:latest .

echo -e "${GREEN}✅ Image built${NC}"
echo ""

# Step 5: Push to Google Container Registry
echo "📋 Step 5: Pushing image to GCR..."
docker push ${IMAGE_NAME}:latest
echo -e "${GREEN}✅ Image pushed${NC}"
echo ""

# Step 6: Deploy to Cloud Run
echo "📋 Step 6: Deploying to Cloud Run..."

gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0 \
    --set-env-vars "GCS_BUCKET=mlops-financial-stress-data,GCS_PROJECT=${PROJECT_ID}" \
    --port 8000

echo -e "${GREEN}✅ Deployment complete${NC}"
echo ""

# Step 7: Get service URL
echo "📋 Step 7: Getting service URL..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

echo ""
echo "================================================================================"
echo "✅ DEPLOYMENT SUCCESSFUL!"
echo "================================================================================"
echo ""
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test the API:"
echo "  Health check: curl ${SERVICE_URL}/api/v1/health"
echo "  List scenarios: curl ${SERVICE_URL}/api/v1/scenarios"
echo "  List companies: curl ${SERVICE_URL}/api/v1/companies"
echo ""
echo "================================================================================"