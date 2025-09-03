"""Classical machine learning models for drug discovery.

This module contains classical neural network models including LSTM generators
that can be combined with quantum models for hybrid approaches.
"""

from .lstm_generator import (
    ConcatenateLayer,
    HybridQuantumLSTMGenerator,
    QuantumConditionedLSTM,
)

__all__ = [
    "QuantumConditionedLSTM",
    "HybridQuantumLSTMGenerator",
    "ConcatenateLayer",
]
