"""Training utilities for quantum-enhanced molecular generation.

This module provides training loops, evaluation metrics, and optimization
procedures for hybrid quantum-classical molecular generation models.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..models.classical import HybridQuantumLSTMGenerator
from ..models.quantum import PennyLaneQCBM
from ..utils.molecular import CompositeFilter, SelfiesProcessor, SmilesProcessor


class MolecularGenerationTrainer:
    """Trainer for quantum-enhanced molecular generation models."""
    
    def __init__(
        self,
        model: Union[HybridQuantumLSTMGenerator, PennyLaneQCBM],
        molecular_processor: Union[SmilesProcessor, SelfiesProcessor],
        device: torch.device = None,
        log_interval: int = 100,
        save_dir: str = "./checkpoints",
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            molecular_processor: Processor for molecular representations
            device: Computing device
            log_interval: Logging interval
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.molecular_processor = molecular_processor
        self.device = device if device is not None else torch.device('cpu')
        self.log_interval = log_interval
        self.save_dir = save_dir
        
        # Move model to device
        # NOTE: PennyLane's Torch backend currently assumes CUDA when migrating
        # tensors off-CPU, which is incompatible with Apple MPS. To avoid
        # "Torch not compiled with CUDA enabled" errors on macOS, we keep
        # pure-quantum models on CPU.
        if isinstance(self.model, PennyLaneQCBM):
            self.model = self.model.to(torch.device('cpu'))
            self.quantum_device = torch.device('cpu')
        else:
            # Move the full model to the requested device
            self.model = self.model.to(self.device)
            # If the model contains a QCBM submodule, keep it on CPU to avoid
            # CUDA/MPS coercion issues during PennyLane simulations
            if hasattr(self.model, 'qcbm') and isinstance(self.model.qcbm, PennyLaneQCBM):
                self.model.qcbm = self.model.qcbm.to(torch.device('cpu'))
                self.quantum_device = torch.device('cpu')
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize molecular filter for evaluation
        self.molecular_filter = CompositeFilter()
        
        # Training history
        self.history = {
            'train_loss': [],
            'quantum_loss': [],
            'validity': [],
            'uniqueness': [],
            'novelty': [],
        }
    
    def prepare_training_data(
        self,
        molecular_data: List[str],
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        """Prepare training data loader.
        
        Args:
            molecular_data: List of molecular strings (SMILES or SELFIES)
            batch_size: Batch size
            max_length: Maximum sequence length
            
        Returns:
            DataLoader for training
        """
        if max_length is None:
            max_length = self.molecular_processor.max_length
        
        # Encode all molecular data
        encoded_data = []
        for mol_str in molecular_data:
            if hasattr(self.molecular_processor, 'encode_smiles'):
                encoded = self.molecular_processor.encode_smiles(mol_str, max_length)
            else:
                encoded = self.molecular_processor.encode_selfies(mol_str, max_length)
            encoded_data.append(encoded)
        
        # Convert to tensor
        data_tensor = torch.tensor(encoded_data, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(data_tensor)
        
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
    
    def train_step_hybrid(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        quantum_optimizer: Optional[torch.optim.Optimizer] = None,
        lambda_quantum: float = 0.1,
    ) -> Dict[str, float]:
        """Perform a single training step for hybrid model.
        
        Args:
            batch: Batch of encoded sequences
            optimizer: Optimizer for LSTM parameters
            quantum_optimizer: Optimizer for quantum parameters
            lambda_quantum: Weight for quantum loss
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        batch = batch.to(self.device)
        
        # Forward pass
        logits, quantum_samples = self.model(batch)
        
        # Compute LSTM loss (cross-entropy on next token prediction)
        # Teacher forcing uses input = tokens[:, :-1] and target = tokens[:, 1:]
        # The model outputs one logit per input token, so logits already has length seq_len-1.
        # Do NOT slice logits again.
        targets = batch[:, 1:]  # Shift targets by 1
        
        lstm_loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Quantum loss (optional - train QCBM to match data distribution)
        quantum_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if quantum_optimizer is not None:
            # Create target quantum data from batch
            # This is simplified - in practice, you might want more sophisticated mapping
            target_quantum = (batch.float() > 0.5).float()  # Binary representation
            if target_quantum.size(1) > self.model.n_qubits:
                target_quantum = target_quantum[:, :self.model.n_qubits]
            elif target_quantum.size(1) < self.model.n_qubits:
                # Pad if needed
                padding = torch.zeros(
                    target_quantum.size(0),
                    self.model.n_qubits - target_quantum.size(1),
                    device=self.device,
                    dtype=torch.float32
                )
                target_quantum = torch.cat([target_quantum, padding], dim=1)
            
            # Ensure quantum data is on CPU for QCBM computations
            if hasattr(self.model, 'qcbm') and isinstance(self.model.qcbm, PennyLaneQCBM):
                target_quantum = target_quantum.to(torch.device('cpu'))
            quantum_loss = self.model.compute_quantum_loss(target_quantum)
            
            # Ensure quantum_loss is float32 before moving to device for MPS compatibility
            if quantum_loss.dtype != torch.float32:
                quantum_loss = quantum_loss.float()
        
        # Total loss
        total_loss = lstm_loss + lambda_quantum * quantum_loss.to(self.device)
        
        # Backward pass
        optimizer.zero_grad()
        if quantum_optimizer is not None:
            quantum_optimizer.zero_grad()
        
        total_loss.backward()
        
        optimizer.step()
        if quantum_optimizer is not None:
            quantum_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'lstm_loss': lstm_loss.item(),
            'quantum_loss': quantum_loss.item(),
        }
    
    def train_step_qcbm(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Perform a single training step for QCBM only.
        
        Args:
            batch: Batch of bitstring data
            optimizer: Optimizer for quantum parameters
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        # Use CPU for QCBM to avoid CUDA/MPS coercion issues inside PennyLane
        q_device = torch.device('cpu') if isinstance(self.model, PennyLaneQCBM) else self.device
        batch = batch.to(q_device)
        
        # Convert to binary representation if needed
        if batch.dtype != torch.float:
            batch = (batch.float() > 0.5).float()
        
        # Ensure correct size
        if batch.size(1) > self.model.n_qubits:
            batch = batch[:, :self.model.n_qubits]
        elif batch.size(1) < self.model.n_qubits:
            padding = torch.zeros(
                batch.size(0),
                self.model.n_qubits - batch.size(1),
                device=q_device
            )
            batch = torch.cat([batch, padding], dim=1)
        
        # Compute loss and perform optimization step
        loss = self.model.train_step(batch, optimizer)
        
        return {'loss': loss}
    
    def evaluate_generation_quality(
        self,
        n_samples: int = 1000,
        reference_molecules: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate quality of generated molecules.
        
        Args:
            n_samples: Number of molecules to generate
            reference_molecules: Reference molecules for novelty calculation
            
        Returns:
            Dictionary of quality metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Generate molecules
            if isinstance(self.model, HybridQuantumLSTMGenerator):
                generated_sequences = self.model.generate(n_samples)
            else:  # QCBM
                generated_sequences = self.model.generate(n_samples)
        
        # Decode sequences to molecular strings
        generated_molecules = []
        for seq in generated_sequences:
            if hasattr(self.molecular_processor, 'decode_sequence'):
                mol_str = self.molecular_processor.decode_sequence(seq.cpu().numpy())
            else:
                mol_str = self.molecular_processor.decode_sequence(seq.cpu().numpy())
            generated_molecules.append(mol_str)
        
        # Filter and validate molecules
        valid_molecules = []
        for mol_str in generated_molecules:
            if hasattr(self.molecular_processor, 'validate_smiles'):
                if self.molecular_processor.validate_smiles(mol_str):
                    valid_molecules.append(mol_str)
            elif hasattr(self.molecular_processor, 'validate_selfies'):
                if self.molecular_processor.validate_selfies(mol_str):
                    valid_molecules.append(mol_str)
        
        # Calculate metrics
        validity = len(valid_molecules) / len(generated_molecules)
        
        unique_molecules = list(set(valid_molecules))
        uniqueness = len(unique_molecules) / max(len(valid_molecules), 1)
        
        # Novelty (if reference molecules provided)
        novelty = 0.0
        if reference_molecules is not None:
            reference_set = set(reference_molecules)
            novel_molecules = [mol for mol in unique_molecules if mol not in reference_set]
            novelty = len(novel_molecules) / max(len(unique_molecules), 1)
        
        # Filter quality using molecular filters
        filtered_molecules, _ = self.molecular_filter.filter_molecules(valid_molecules)
        drug_likeness = len(filtered_molecules) / max(len(valid_molecules), 1)
        
        return {
            'validity': validity,
            'uniqueness': uniqueness,
            'novelty': novelty,
            'drug_likeness': drug_likeness,
            'n_generated': len(generated_molecules),
            'n_valid': len(valid_molecules),
            'n_unique': len(unique_molecules),
            'n_drug_like': len(filtered_molecules),
        }
    
    def train_hybrid(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        quantum_learning_rate: float = 1e-2,
        lambda_quantum: float = 0.1,
        eval_interval: int = 10,
        reference_molecules: Optional[List[str]] = None,
    ) -> Dict[str, List[float]]:
        """Train hybrid quantum-classical model.
        
        Args:
            data_loader: Training data loader
            n_epochs: Number of epochs
            learning_rate: Learning rate for LSTM parameters
            quantum_learning_rate: Learning rate for quantum parameters
            lambda_quantum: Weight for quantum loss
            eval_interval: Evaluation interval
            reference_molecules: Reference molecules for novelty evaluation
            
        Returns:
            Training history
        """
        # Initialize optimizers
        lstm_optimizer = optim.Adam(
            self.model.lstm_generator.parameters(),
            lr=learning_rate
        )
        quantum_optimizer = optim.Adam(
            self.model.qcbm.parameters(),
            lr=quantum_learning_rate
        )
        
        for epoch in range(n_epochs):
            epoch_losses = []
            epoch_quantum_losses = []
            
            # Training loop
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
            for batch_idx, (batch,) in enumerate(progress_bar):
                losses = self.train_step_hybrid(
                    batch,
                    lstm_optimizer,
                    quantum_optimizer,
                    lambda_quantum
                )
                
                epoch_losses.append(losses['total_loss'])
                epoch_quantum_losses.append(losses['quantum_loss'])
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{losses['total_loss']:.4f}",
                    'LSTM': f"{losses['lstm_loss']:.4f}",
                    'Quantum': f"{losses['quantum_loss']:.4f}",
                })
            
            # Record epoch metrics
            avg_loss = np.mean(epoch_losses)
            avg_quantum_loss = np.mean(epoch_quantum_losses)
            
            self.history['train_loss'].append(avg_loss)
            self.history['quantum_loss'].append(avg_quantum_loss)
            
            # Evaluation
            if epoch % eval_interval == 0:
                print(f"\\nEvaluating at epoch {epoch+1}...")
                eval_metrics = self.evaluate_generation_quality(
                    n_samples=500,
                    reference_molecules=reference_molecules
                )
                
                self.history['validity'].append(eval_metrics['validity'])
                self.history['uniqueness'].append(eval_metrics['uniqueness'])
                self.history['novelty'].append(eval_metrics['novelty'])
                
                print(f"Validity: {eval_metrics['validity']:.3f}, "
                      f"Uniqueness: {eval_metrics['uniqueness']:.3f}, "
                      f"Novelty: {eval_metrics['novelty']:.3f}, "
                      f"Drug-likeness: {eval_metrics['drug_likeness']:.3f}")
            
            # Save checkpoint
            if epoch % (eval_interval * 2) == 0:
                self.save_checkpoint(epoch, lstm_optimizer, quantum_optimizer)
        
        return self.history
    
    def train_qcbm_only(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_epochs: int = 100,
        learning_rate: float = 1e-2,
        eval_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """Train QCBM only.
        
        Args:
            data_loader: Training data loader
            n_epochs: Number of epochs
            learning_rate: Learning rate
            eval_interval: Evaluation interval
            
        Returns:
            Training history
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
            for batch_idx, (batch,) in enumerate(progress_bar):
                losses = self.train_step_qcbm(batch, optimizer)
                epoch_losses.append(losses['loss'])
                
                progress_bar.set_postfix({'Loss': f"{losses['loss']:.4f}"})
            
            avg_loss = np.mean(epoch_losses)
            self.history['train_loss'].append(avg_loss)
            
            if epoch % eval_interval == 0:
                print(f"\\nEpoch {epoch+1}, Average Loss: {avg_loss:.4f}")
                
                # Generate and evaluate samples
                eval_metrics = self.evaluate_generation_quality(n_samples=200)
                print(f"Generated {eval_metrics['n_generated']} samples, "
                      f"Valid: {eval_metrics['n_valid']}")
        
        return self.history
    
    def save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        quantum_optimizer: Optional[torch.optim.Optimizer] = None,
        filename: Optional[str] = None,
    ):
        """Save training checkpoint.
        
        Args:
            epoch: Current epoch
            optimizer: Main optimizer
            quantum_optimizer: Quantum optimizer (optional)
            filename: Custom filename (optional)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': self.history,
            'model_config': self.model.config() if hasattr(self.model, 'config') else {}
        }
        
        if quantum_optimizer is not None:
            checkpoint['quantum_optimizer_state_dict'] = quantum_optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(
        self,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        quantum_optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> int:
        """Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            optimizer: Optimizer to load state into (optional)
            quantum_optimizer: Quantum optimizer to load state into (optional)
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if quantum_optimizer is not None and 'quantum_optimizer_state_dict' in checkpoint:
            quantum_optimizer.load_state_dict(checkpoint['quantum_optimizer_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded: {filepath}, epoch {epoch}")
        
        return epoch
