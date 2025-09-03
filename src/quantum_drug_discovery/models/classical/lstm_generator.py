"""LSTM-based molecular generator with quantum prior integration.

This module implements LSTM generators that can be conditioned on quantum circuit
outputs, enabling hybrid quantum-classical molecular generation.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..quantum.qcbm import PennyLaneQCBM


class ConcatenateLayer(nn.Module):
    """Layer for concatenating tensors along a specified dimension."""
    
    def __init__(self, dim: int = -1):
        """Initialize concatenation layer.
        
        Args:
            dim: Dimension along which to concatenate
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate tensors.
        
        Args:
            tensors: List of tensors to concatenate
            
        Returns:
            Concatenated tensor
        """
        return torch.cat(tensors, dim=self.dim)


class QuantumConditionedLSTM(nn.Module):
    """LSTM conditioned on quantum circuit outputs.
    
    This model takes quantum samples as input and uses them to condition
    the generation of molecular sequences.
    """
    
    def __init__(
        self,
        vocab_size: int,
        quantum_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        projection_dim: int = 64,
        padding_idx: int = 0,
    ):
        """Initialize the quantum-conditioned LSTM.
        
        Args:
            vocab_size: Size of the molecular vocabulary
            quantum_dim: Dimension of quantum samples
            embedding_dim: Dimension of token embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            projection_dim: Dimension for quantum sample projection
            padding_idx: Index of padding token
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.quantum_dim = quantum_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.projection_dim = projection_dim
        self.padding_idx = padding_idx
        
        # Token embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        
        # Quantum sample projection
        self.quantum_projection = nn.Sequential(
            nn.Linear(quantum_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM with combined input (embeddings + quantum projection)
        lstm_input_dim = embedding_dim + projection_dim
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Concatenation layer
        self.concat = ConcatenateLayer(dim=-1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        tokens: torch.Tensor,
        quantum_samples: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the model.
        
        Args:
            tokens: Input token sequence [batch_size, seq_len]
            quantum_samples: Quantum samples [batch_size, quantum_dim]
            hidden: Initial hidden state (optional)
            
        Returns:
            Tuple of (logits, hidden_state)
        """
        batch_size, seq_len = tokens.shape
        
        # Embed tokens: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(tokens)
        
        # Project quantum samples: [batch_size, projection_dim]
        quantum_projected = self.quantum_projection(quantum_samples)
        
        # Expand quantum samples to match sequence length
        # [batch_size, seq_len, projection_dim]
        quantum_expanded = quantum_projected.unsqueeze(1).expand(
            batch_size, seq_len, self.projection_dim
        )
        
        # Concatenate embeddings with quantum features
        # [batch_size, seq_len, embedding_dim + projection_dim]
        lstm_input = self.concat([embedded, quantum_expanded])
        
        # LSTM forward pass
        lstm_output, hidden_state = self.lstm(lstm_input, hidden)
        
        # Apply dropout
        lstm_output = self.dropout(lstm_output)
        
        # Project to vocabulary: [batch_size, seq_len, vocab_size]
        logits = self.output_projection(lstm_output)
        
        return logits, hidden_state
    
    def generate_step(
        self,
        token: torch.Tensor,
        quantum_sample: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate a single token step.
        
        Args:
            token: Current token [batch_size, 1]
            quantum_sample: Quantum sample [batch_size, quantum_dim]
            hidden: Hidden state
            temperature: Sampling temperature
            
        Returns:
            Tuple of (next_token, log_probs, new_hidden_state)
        """
        logits, new_hidden = self.forward(token, quantum_sample, hidden)
        
        # Apply temperature scaling
        logits = logits / temperature
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Sample next token
        dist = Categorical(probs.squeeze(1))  # Remove seq_len dimension
        next_token = dist.sample()
        
        return next_token.unsqueeze(1), log_probs, new_hidden


class HybridQuantumLSTMGenerator(nn.Module):
    """Hybrid quantum-classical molecular generator.
    
    This model combines a quantum circuit (QCBM) with an LSTM generator
    to produce molecular sequences.
    """
    
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        n_qubits: int = 10,
        n_layers: int = 4,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        projection_dim: int = 64,
        padding_idx: int = 0,
        sos_token: int = 1,
        eos_token: int = 2,
        quantum_device: str = "default.qubit",
    ):
        """Initialize hybrid quantum-LSTM generator.
        
        Args:
            vocab_size: Size of molecular vocabulary
            seq_len: Maximum sequence length
            n_qubits: Number of qubits in quantum circuit
            n_layers: Number of quantum circuit layers
            embedding_dim: Token embedding dimension
            hidden_dim: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            projection_dim: Quantum projection dimension
            padding_idx: Padding token index
            sos_token: Start-of-sequence token
            eos_token: End-of-sequence token
            quantum_device: PennyLane device name
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_qubits = n_qubits
        self.padding_idx = padding_idx
        self.sos_token = sos_token
        self.eos_token = eos_token
        
        # Quantum circuit for generating priors
        self.qcbm = PennyLaneQCBM(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device_name=quantum_device,
            shots=None  # Use exact simulation for training
        )
        
        # LSTM generator conditioned on quantum samples
        self.lstm_generator = QuantumConditionedLSTM(
            vocab_size=vocab_size,
            quantum_dim=n_qubits,  # QCBM outputs bitstrings of length n_qubits
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
            projection_dim=projection_dim,
            padding_idx=padding_idx,
        )
    
    def forward(
        self,
        target_sequences: torch.Tensor,
        n_quantum_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.
        
        Args:
            target_sequences: Target sequences [batch_size, seq_len]
            n_quantum_samples: Number of quantum samples (defaults to batch_size)
            
        Returns:
            Tuple of (logits, quantum_samples)
        """
        batch_size = target_sequences.size(0)
        if n_quantum_samples is None:
            n_quantum_samples = batch_size
        
        # Generate quantum samples on CPU (QCBM) then move to LSTM/device
        device = target_sequences.device
        quantum_samples = self.qcbm.generate(n_quantum_samples)
        
        # Ensure float32 for MPS compatibility
        if quantum_samples.dtype != torch.float32:
            quantum_samples = quantum_samples.float()
        
        # Move to target device - create a new tensor to avoid MPS allocation issues
        quantum_samples = torch.tensor(quantum_samples.cpu().numpy(), dtype=torch.float32, device=device)
        
        # If we need more/fewer samples than generated, handle appropriately
        if quantum_samples.size(0) != batch_size:
            indices = torch.randperm(quantum_samples.size(0), device=device)[:batch_size]
            quantum_samples = quantum_samples[indices]
        
        # Teacher forcing: use target sequences as input (shifted by 1)
        input_sequences = target_sequences[:, :-1]  # Remove last token
        
        # Forward through LSTM
        logits, _ = self.lstm_generator(input_sequences, quantum_samples)
        
        return logits, quantum_samples
    
    def generate(
        self,
        n_samples: int,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        return_quantum_samples: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate molecular sequences.
        
        Args:
            n_samples: Number of sequences to generate
            max_length: Maximum sequence length (uses self.seq_len if None)
            temperature: Sampling temperature
            return_quantum_samples: Whether to return quantum samples
            
        Returns:
            Generated sequences, optionally with quantum samples
        """
        if max_length is None:
            max_length = self.seq_len
        
        # Important: choose device from LSTM parameters (qcbm stays on CPU)
        device = next(self.lstm_generator.parameters()).device
        
        # Generate quantum samples on CPU first
        quantum_samples = self.qcbm.generate(n_samples)
        
        # Ensure float32 for MPS compatibility
        if quantum_samples.dtype != torch.float32:
            quantum_samples = quantum_samples.float()
        
        # Move to target device - create a new tensor to avoid MPS allocation issues
        quantum_samples = torch.tensor(quantum_samples.cpu().numpy(), dtype=torch.float32, device=device)
        
        # Initialize sequences with SOS token - use zeros then fill to avoid MPS issues
        generated_sequences = torch.zeros(
            (n_samples, max_length), dtype=torch.long, device=device
        )
        
        # Fill with padding index first
        generated_sequences.fill_(self.padding_idx)
        
        # Set SOS token for first position
        generated_sequences[:, 0] = self.sos_token
        
        # Initialize hidden state
        hidden = None
        
        # Generate tokens sequentially
        for t in range(1, max_length):
            # Current token input - ensure it's properly allocated on device
            current_token = generated_sequences[:, t-1:t].contiguous()
            
            # Generate next token
            next_token, _, hidden = self.lstm_generator.generate_step(
                current_token, quantum_samples, hidden, temperature
            )
            
            # Store generated token - direct assignment should work with proper initialization
            generated_sequences[:, t] = next_token.squeeze(1)
            
            # Check for EOS tokens (optional early stopping)
            if (next_token == self.eos_token).all():
                break
        
        if return_quantum_samples:
            return generated_sequences, quantum_samples
        return generated_sequences
    
    def compute_quantum_loss(self, target_data: torch.Tensor) -> torch.Tensor:
        """Compute loss for quantum circuit training.
        
        Args:
            target_data: Target bitstring data for QCBM training
            
        Returns:
            Quantum circuit loss
        """
        return self.qcbm.compute_loss(target_data)
    
    def train_quantum_step(
        self,
        target_data: torch.Tensor,
        quantum_optimizer: torch.optim.Optimizer,
    ) -> float:
        """Train the quantum circuit component.
        
        Args:
            target_data: Target bitstring data
            quantum_optimizer: Optimizer for quantum parameters
            
        Returns:
            Quantum loss value
        """
        return self.qcbm.train_step(target_data, quantum_optimizer)
    
    def config(self) -> Dict:
        """Get model configuration."""
        return {
            "model_type": "HybridQuantumLSTMGenerator",
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "n_qubits": self.n_qubits,
            "quantum_config": self.qcbm.config(),
            "lstm_config": {
                "embedding_dim": self.lstm_generator.embedding_dim,
                "hidden_dim": self.lstm_generator.hidden_dim,
                "num_layers": self.lstm_generator.num_layers,
                "projection_dim": self.lstm_generator.projection_dim,
            }
        }
