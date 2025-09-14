"""Setup script for multimodal foundation model training framework."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
README = Path("README.md").read_text(encoding="utf-8")

# Read requirements
requirements = Path("requirements.txt").read_text().strip().split("\n")

setup(
    name="multimodal-foundation-model",
    version="1.0.0",
    description="Production-ready multimodal vision-language model fine-tuning framework",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/multimodal-foundation-model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "neural": [
            "mne>=1.5.0",
            "nibabel>=5.1.0",
            "nilearn>=0.10.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="multimodal, vision-language, CLIP, LLaVA, LoRA, distributed-training, MLOps",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/multimodal-foundation-model/issues",
        "Source": "https://github.com/yourusername/multimodal-foundation-model",
        "Documentation": "https://multimodal-foundation-model.readthedocs.io/",
    },
    entry_points={
        "console_scripts": [
            "multimodal-train=scripts.train_distributed:main",
            "multimodal-eval=scripts.evaluate_model:main",
            "multimodal-benchmark=scripts.benchmark:main",
        ]
    }
)