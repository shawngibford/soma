# Quantum Drug Discovery with PennyLane

A reimplementation of quantum-enhanced drug discovery using the PennyLane quantum computing framework. This project focuses on generating novel molecular compounds using Quantum Circuit Born Machines (QCBM) and hybrid quantum-classical generative models.

## Overview

This project reimplements the quantum-enhanced drug discovery pipeline originally built with the Orquestra framework, now using PennyLane for improved accessibility, performance, and integration with modern ML workflows.

### Key Features

- **Quantum Circuit Born Machines (QCBM)**: Quantum generative models for molecular sampling
- **Hybrid Quantum-Classical Models**: Integration of QCBM priors with LSTM generators
- **Molecular Property Prediction**: Quantum-enhanced approaches for drug-likeness assessment
- **Comprehensive Filtering**: PAINS, Lipinski's Rule of Five, and synthetic accessibility
- **Experimental Framework**: Tools for benchmarking quantum vs classical approaches

## Project Structure

```
├── src/quantum_drug_discovery/    # Main package
│   ├── models/                    # Model implementations
│   │   ├── quantum/              # Quantum models (QCBM, etc.)
│   │   ├── classical/            # Classical baselines
│   │   └── hybrid/               # Hybrid quantum-classical models
│   ├── utils/                    # Utility modules
│   │   ├── molecular/            # Molecular processing and filtering
│   │   ├── quantum/              # Quantum circuit utilities
│   │   └── data/                 # Data loading and preprocessing
│   ├── experiments/              # Experiment configurations
│   └── training/                 # Training scripts and workflows
├── notebooks/                    # Jupyter notebooks for analysis
├── tests/                        # Unit and integration tests
├── docs/                         # Documentation
├── data/                         # Data storage
├── scripts/                      # Utility scripts
└── config/                       # Configuration files
```

## Installation

### Quick Installation (Recommended)
```bash
# Navigate to project directory
cd /Users/shawngibford/dev/soma-pennylane

# Install core dependencies
pip install -r requirements-core.txt

# Install in development mode
pip install -e .
```

### Full Installation (with optional packages)
```bash
# Install all dependencies (may have compatibility issues)
pip install -r requirements.txt

# Or install with optional features
pip install -e .[syba,dev,docs]
```

### Troubleshooting Installation
If you encounter issues with specific packages:
1. Use `requirements-core.txt` for essential packages only
2. Install optional packages separately: `pip install syba` (if needed)
3. Some packages may require specific Python versions (3.8-3.11 recommended)

## Quick Start

```python
from quantum_drug_discovery.models.quantum import PennyLaneQCBM
from quantum_drug_discovery.utils.molecular import SmilesProcessor

# Initialize quantum model
qcbm = PennyLaneQCBM(n_qubits=10, n_layers=4)

# Generate molecular samples
samples = qcbm.generate(n_samples=1000)

# Process and validate molecules
processor = SmilesProcessor()
valid_molecules = processor.validate_samples(samples)
```

## Key Differences from Original Implementation

### Framework Migration
- **From**: Orquestra quantum framework
- **To**: PennyLane with PyTorch backend
- **Benefits**: Better integration with ML workflows, improved performance, wider community support

### Architectural Improvements
- Modular design with clear separation of quantum and classical components
- Enhanced configuration management
- Comprehensive testing and documentation
- Modern Python packaging and deployment

## Original Research Reference

This implementation is based on the research described in "Quantum-computing-enhanced algorithm unveils potential KRAS inhibitors" and related work on quantum-enhanced molecular generation.

## Requirements

- Python 3.8+
- PennyLane >= 0.30
- PyTorch >= 1.12
- RDKit >= 2022.09
- NumPy, SciPy, Pandas
- Jupyter (for notebooks)

## Contributing

Please read our contributing guidelines and code of conduct before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Orquestra-based implementation
- PennyLane quantum computing framework
- RDKit cheminformatics toolkit
- Research contributions from the quantum machine learning community
