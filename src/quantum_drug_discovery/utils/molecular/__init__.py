"""Molecular processing utilities.

This module provides utilities for processing molecular representations,
including encoding/decoding and filtering for drug discovery applications.
"""

from .encoding import (
    MolecularSequence,
    PaddingOptions,
    SelfiesProcessor,
    SmilesProcessor,
    convert_bitstring_to_molecular_encoding,
)
from .filters import (
    CompositeFilter,
    LipinskiFilter,
    PAINSFilter,
    StructuralFilter,
    apply_all_filters,
    quick_lipinski_filter,
    quick_pains_filter,
)

__all__ = [
    # Encoding utilities
    "SmilesProcessor",
    "SelfiesProcessor",
    "MolecularSequence",
    "PaddingOptions",
    "convert_bitstring_to_molecular_encoding",
    # Filtering utilities
    "LipinskiFilter",
    "PAINSFilter", 
    "StructuralFilter",
    "CompositeFilter",
    "quick_lipinski_filter",
    "quick_pains_filter",
    "apply_all_filters",
]
