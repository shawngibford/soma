#!/usr/bin/env python3
"""Setup script for quantum drug discovery package."""

from setuptools import find_packages, setup

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core requirements (without optional packages)
core_requirements = [
    "pennylane>=0.28.0",
    "pennylane-lightning>=0.28.0", 
    "torch>=1.10.0",
    "numpy>=1.20.0",
    "scipy>=1.6.0",
    "rdkit-pypi>=2022.3.1",
    "selfies>=2.0.0",
    "pandas>=1.2.0",
    "matplotlib>=3.3.0",
    "seaborn>=0.11.0",
    "tqdm>=4.60.0",
    "optuna>=2.10.0",
    "pyyaml>=6.0",
    "hydra-core>=1.2.0",
    "click>=8.0.0",
    "python-dotenv>=0.19.0",
]

# Optional requirements
optional_requirements = {
    "syba": ["syba>=0.0.5"],  # Synthetic accessibility scoring
}

setup(
    name="quantum-drug-discovery",
    version="0.1.0",
    author="Quantum Drug Discovery Team",
    author_email="quantum.dd@example.com",
    description="Quantum-enhanced drug discovery using PennyLane",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-drug-discovery-pennylane",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/quantum-drug-discovery-pennylane/issues",
        "Documentation": "https://quantum-drug-discovery.readthedocs.io/",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
        ],
        "syba": optional_requirements["syba"],
        "docs": [
            "jupyter>=1.0.0",
            "jupyter-book>=0.13.0",
            "sphinx>=4.0.0",
        ],
        "gpu": [
            "torch-audio>=0.12.0",
            "torch-vision>=0.13.0",
        ],
        "quantum-backends": [
            "pennylane-qiskit>=0.30.0",
            "pennylane-cirq>=0.30.0",
        ],
        "distributed": [
            "dask>=2022.0.0",
            "ray>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qdd-train=quantum_drug_discovery.training.train:main",
            "qdd-generate=quantum_drug_discovery.experiments.generate:main",
            "qdd-evaluate=quantum_drug_discovery.experiments.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="quantum computing, drug discovery, machine learning, molecular generation, QCBM, pennylane",
)
