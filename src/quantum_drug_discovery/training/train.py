#!/usr/bin/env python3
"""Main training script for quantum drug discovery models.

This script provides a command-line interface for training quantum-enhanced
molecular generation models using the PennyLane framework.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quantum_drug_discovery.models.classical import HybridQuantumLSTMGenerator
from quantum_drug_discovery.models.quantum import PennyLaneQCBM
from quantum_drug_discovery.training.trainer import MolecularGenerationTrainer
from quantum_drug_discovery.utils.molecular import SelfiesProcessor, SmilesProcessor


def load_molecular_data(data_path: str, mol_column: str = "smiles") -> List[str]:
    """Load molecular data from file.
    
    Args:
        data_path: Path to data file (CSV, JSON, or text)
        mol_column: Column name containing molecular strings
        
    Returns:
        List of molecular strings
    """
    data_path = Path(data_path)
    
    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
        return df[mol_column].dropna().tolist()
    elif data_path.suffix == ".json":
        with open(data_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and mol_column in data:
            return data[mol_column]
        else:
            raise ValueError(f"JSON file must contain list or dict with '{mol_column}' key")
    elif data_path.suffix == ".txt":
        with open(data_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train quantum drug discovery models")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to molecular data file")
    parser.add_argument("--mol_column", type=str, default="smiles",
                       help="Column name for molecular strings")
    parser.add_argument("--representation", type=str, choices=["smiles", "selfies"],
                       default="smiles", help="Molecular representation type")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, 
                       choices=["qcbm", "hybrid"], default="hybrid",
                       help="Type of model to train")
    parser.add_argument("--n_qubits", type=int, default=10,
                       help="Number of qubits in quantum circuit")
    parser.add_argument("--n_layers", type=int, default=4,
                       help="Number of layers in quantum circuit")
    parser.add_argument("--seq_len", type=int, default=100,
                       help="Maximum sequence length")
    parser.add_argument("--vocab_size", type=int, default=None,
                       help="Vocabulary size (auto-detected if None)")
    parser.add_argument("--embedding_dim", type=int, default=64,
                       help="Embedding dimension for LSTM")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden dimension for LSTM")
    parser.add_argument("--lstm_layers", type=int, default=2,
                       help="Number of LSTM layers")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate for LSTM parameters")
    parser.add_argument("--quantum_learning_rate", type=float, default=1e-2,
                       help="Learning rate for quantum parameters")
    parser.add_argument("--lambda_quantum", type=float, default=0.1,
                       help="Weight for quantum loss in hybrid training")
    parser.add_argument("--eval_interval", type=int, default=10,
                       help="Evaluation interval (epochs)")
    
    # Device and logging
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cpu, cuda, auto)")
    parser.add_argument("--quantum_device", type=str, default="default.qubit",
                       help="PennyLane quantum device")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Logging interval")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Quantum device: {args.quantum_device}")
    
    # Load data
    print(f"Loading molecular data from {args.data_path}...")
    molecular_data = load_molecular_data(args.data_path, args.mol_column)
    print(f"Loaded {len(molecular_data)} molecules")
    
    # Initialize molecular processor
    if args.representation == "smiles":
        processor = SmilesProcessor()
        processor.smiles_list = molecular_data  # Set data to build vocabulary
        processor._build_vocabulary()
    else:  # selfies
        processor = SelfiesProcessor()
        processor.selfies_list = molecular_data
        processor._build_vocabulary()
    
    vocab_size = args.vocab_size if args.vocab_size else processor.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize model
    if args.model_type == "qcbm":
        model = PennyLaneQCBM(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            device_name=args.quantum_device,
        )
        print(f"Initialized QCBM with {args.n_qubits} qubits and {args.n_layers} layers")
    else:  # hybrid
        model = HybridQuantumLSTMGenerator(
            vocab_size=vocab_size,
            seq_len=args.seq_len,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            lstm_layers=args.lstm_layers,
            quantum_device=args.quantum_device,
        )
        print(f"Initialized hybrid model with {vocab_size} vocabulary size")
    
    # Initialize trainer
    trainer = MolecularGenerationTrainer(
        model=model,
        molecular_processor=processor,
        device=device,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
    )
    
    # Prepare training data
    print("Preparing training data...")
    data_loader = trainer.prepare_training_data(
        molecular_data,
        batch_size=args.batch_size,
        max_length=args.seq_len,
    )
    print(f"Created data loader with {len(data_loader)} batches")
    
    # Train model
    print(f"Starting training for {args.n_epochs} epochs...")
    if args.model_type == "qcbm":
        history = trainer.train_qcbm_only(
            data_loader=data_loader,
            n_epochs=args.n_epochs,
            learning_rate=args.quantum_learning_rate,
            eval_interval=args.eval_interval,
        )
    else:  # hybrid
        # Use subset of training data as reference for novelty calculation
        reference_molecules = molecular_data[:min(1000, len(molecular_data))]
        
        history = trainer.train_hybrid(
            data_loader=data_loader,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            quantum_learning_rate=args.quantum_learning_rate,
            lambda_quantum=args.lambda_quantum,
            eval_interval=args.eval_interval,
            reference_molecules=reference_molecules,
        )
    
    # Save final model
    final_checkpoint_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config() if hasattr(model, 'config') else {},
        'training_args': vars(args),
        'history': history,
        'processor_config': {
            'type': args.representation,
            'vocab_size': vocab_size,
            'max_length': processor.max_length,
        }
    }, final_checkpoint_path)
    
    print(f"Training completed! Final model saved to {final_checkpoint_path}")
    
    # Final evaluation
    print("\\nFinal evaluation:")
    if args.model_type == "hybrid":
        eval_metrics = trainer.evaluate_generation_quality(
            n_samples=1000,
            reference_molecules=reference_molecules
        )
        print(f"Validity: {eval_metrics['validity']:.3f}")
        print(f"Uniqueness: {eval_metrics['uniqueness']:.3f}")
        print(f"Novelty: {eval_metrics['novelty']:.3f}")
        print(f"Drug-likeness: {eval_metrics['drug_likeness']:.3f}")
    else:
        eval_metrics = trainer.evaluate_generation_quality(n_samples=500)
        print(f"Generated: {eval_metrics['n_generated']}")
        print(f"Valid: {eval_metrics['n_valid']}")


if __name__ == "__main__":
    main()
