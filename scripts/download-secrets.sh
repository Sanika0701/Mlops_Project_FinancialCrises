#!/bin/bash
set -e

echo "Downloading GCS service account key..."
gsutil cp gs://mlops-financial-stress-data/secrets/ninth-iris-422916-f2-9aec4e0969c6.json \
  ./gcs-key.json

echo "✓ Secret downloaded successfully"