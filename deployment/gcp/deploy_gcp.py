#!/usr/bin/env python3
"""
GCP Vertex AI Deployment
Deploy shore model to Google Cloud Platform
"""

import os
import yaml
import logging
from pathlib import Path
from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def deploy_to_vertex_ai(
    project_id: str,
    region: str,
    model_path: str,
    endpoint_name: str = "marauder-cv-shore"
):
    """Deploy model to Vertex AI"""
    
    logger.info(f"Deploying to Vertex AI")
    logger.info(f"  Project: {project_id}")
    logger.info(f"  Region: {region}")
    logger.info(f"  Model: {model_path}")
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Upload model
    logger.info("Uploading model...")
    model = aiplatform.Model.upload(
        display_name="marauder-cv-shore",
        artifact_uri=model_path,
        serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-13:latest"
    )
    
    # Create endpoint
    logger.info("Creating endpoint...")
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    
    # Deploy model to endpoint
    logger.info("Deploying model to endpoint...")
    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=3
    )
    
    logger.info(f"✓ Deployment complete!")
    logger.info(f"  Endpoint: {endpoint.resource_name}")
    
    return endpoint


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy to GCP')
    parser.add_argument('--project-id', type=str, required=True)
    parser.add_argument('--region', type=str, default='us-central1')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--endpoint-name', type=str, default='marauder-cv-shore')
    
    args = parser.parse_args()
    
    endpoint = deploy_to_vertex_ai(
        args.project_id,
        args.region,
        args.model_path,
        args.endpoint_name
    )
    
    print(f"\n✓ Model deployed successfully!")
    print(f"  Endpoint: {endpoint.resource_name}")


if __name__ == '__main__':
    main()
