"""GCP Deployment - Cloud deployment for high-accuracy inference"""
from .deploy_gcp import deploy_to_vertex_ai

__all__ = ["deploy_to_vertex_ai"]
