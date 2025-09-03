"""Quantum Drug Discovery with PennyLane.

A comprehensive quantum-enhanced drug discovery framework using PennyLane
for quantum computing and PyTorch for classical machine learning.
"""

__version__ = "0.1.0"
__author__ = "Quantum Drug Discovery Team"
__email__ = "quantum.dd@example.com"

from . import experiments, models, training, utils

__all__ = [
    "models",
    "utils", 
    "training",
    "experiments",
]
