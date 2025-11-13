#!/usr/bin/env python3
"""
Marauder CV Pipeline - Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="marauder-cv-pipeline",
    version="1.0.0",
    description="Complete Computer Vision System for Marine Species Detection and Counting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Marauder Team",
    author_email="contact@marauder-project.org",
    url="https://github.com/marauder/cv-pipeline",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.196",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "supervision>=0.16.0",
        "boto3>=1.28.0",
        "tqdm>=4.66.0",
        "rich>=13.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
        ],
        "gcp": [
            "google-cloud-storage>=2.10.0",
            "google-cloud-aiplatform>=1.35.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "marauder-train=training.1_baseline_yolo:main",
            "marauder-infer-nano=inference.nano_inference:main",
            "marauder-infer-shore=inference.shore_inference:main",
            "marauder-evaluate=evaluation.comprehensive_evaluator:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="computer-vision marine-biology object-detection yolo jetson-nano",
    include_package_data=True,
    zip_safe=False,
)
