"""Molecular encoding utilities for SMILES and SELFIES.

This module provides utilities for encoding and decoding molecular representations,
including SMILES and SELFIES strings, for use with quantum and classical models.
"""

import typing as t
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import selfies as sf
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from scipy.special import softmax


class PaddingOptions(Enum):
    """Options for padding sequences."""
    BEFORE = "before"
    AFTER = "after"


@dataclass
class MolecularSequence:
    """Data class for storing molecular sequence information."""
    sequence: str
    encoding: np.ndarray
    properties: Optional[Dict[str, float]] = None


class SmilesProcessor:
    """Processor for SMILES molecular representations."""
    
    def __init__(self, data_path: Optional[str] = None, padding_token: str = " "):
        """Initialize SMILES processor.
        
        Args:
            data_path: Path to CSV file containing SMILES data
            padding_token: Token used for sequence padding
        """
        self.data_path = data_path
        self.padding_token = padding_token
        
        if data_path:
            self.df = pd.read_csv(data_path)
            self.smiles_list = self.df['smiles'].tolist()
        else:
            self.smiles_list = []
            
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build vocabulary from SMILES sequences."""
        if self.smiles_list:
            all_chars = set(''.join(self.smiles_list))
            all_chars.add(self.padding_token)
            self.vocabulary = sorted(list(all_chars))
        else:
            # Default SMILES vocabulary
            self.vocabulary = [
                'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
                '(', ')', '[', ']', '=', '#', '@', '+', '-',
                '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'H', 'c', 'n', 'o', 's', 'p', self.padding_token
            ]
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.vocabulary)
    
    @property
    def max_length(self) -> int:
        """Maximum length of SMILES sequences."""
        if self.smiles_list:
            return max(len(smiles) for smiles in self.smiles_list)
        return 100  # Default max length
    
    def encode_smiles(self, smiles: str, max_length: Optional[int] = None) -> np.ndarray:
        """Encode SMILES string to integer sequence.
        
        Args:
            smiles: SMILES string to encode
            max_length: Maximum sequence length (uses self.max_length if None)
            
        Returns:
            Encoded integer sequence
        """
        if max_length is None:
            max_length = self.max_length
            
        # Convert characters to indices
        encoded = []
        for char in smiles[:max_length]:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                # Use padding token for unknown characters
                encoded.append(self.char_to_idx[self.padding_token])
        
        # Pad sequence
        while len(encoded) < max_length:
            encoded.append(self.char_to_idx[self.padding_token])
            
        return np.array(encoded, dtype=np.int32)
    
    def decode_sequence(self, sequence: np.ndarray) -> str:
        """Decode integer sequence back to SMILES string.
        
        Args:
            sequence: Encoded integer sequence
            
        Returns:
            Decoded SMILES string
        """
        chars = []
        for idx in sequence:
            if idx in self.idx_to_char and self.idx_to_char[idx] != self.padding_token:
                chars.append(self.idx_to_char[idx])
        
        return ''.join(chars)
    
    def one_hot_encode(self, smiles: str, max_length: Optional[int] = None) -> np.ndarray:
        """One-hot encode SMILES string.
        
        Args:
            smiles: SMILES string to encode
            max_length: Maximum sequence length
            
        Returns:
            One-hot encoded array of shape (max_length, vocab_size)
        """
        if max_length is None:
            max_length = self.max_length
            
        encoded_seq = self.encode_smiles(smiles, max_length)
        one_hot = np.zeros((max_length, self.vocab_size), dtype=np.float32)
        
        for i, char_idx in enumerate(encoded_seq):
            one_hot[i, char_idx] = 1.0
            
        return one_hot
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string using RDKit.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """Convert SMILES to canonical form.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Canonical SMILES string or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None
    
    def validate_samples(self, samples: List[str]) -> List[str]:
        """Validate a list of SMILES samples.
        
        Args:
            samples: List of SMILES strings
            
        Returns:
            List of valid SMILES strings
        """
        valid_samples = []
        for smiles in samples:
            canonical = self.canonicalize_smiles(smiles)
            if canonical:
                valid_samples.append(canonical)
        
        return valid_samples


class SelfiesProcessor:
    """Processor for SELFIES molecular representations."""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        padding_symbol: str = "[nop]",
        padding_mode: PaddingOptions = PaddingOptions.AFTER,
        activity_column: str = "activity",
    ):
        """Initialize SELFIES processor.
        
        Args:
            data_path: Path to CSV file containing molecular data
            padding_symbol: Symbol used for padding
            padding_mode: Whether to pad before or after sequence
            activity_column: Column name for molecular activity data
        """
        self.data_path = data_path
        self.padding_symbol = padding_symbol
        self.padding_mode = padding_mode.value
        self.activity_column = activity_column
        
        if data_path:
            self.df = pd.read_csv(data_path)
            if 'selfies' in self.df.columns:
                self.selfies_list = self.df['selfies'].tolist()
            elif 'smiles' in self.df.columns:
                # Convert SMILES to SELFIES
                self.selfies_list = []
                for smiles in self.df['smiles'].tolist():
                    selfies = sf.encoder(smiles)
                    if selfies:
                        self.selfies_list.append(selfies)
            else:
                raise ValueError("Data must contain 'smiles' or 'selfies' column")
        else:
            self.selfies_list = []
        
        self._build_vocabulary()
        self._extract_activities()
    
    def _build_vocabulary(self):
        """Build SELFIES vocabulary."""
        if self.selfies_list:
            alphabet_set = sf.get_alphabet_from_selfies(self.selfies_list)
        else:
            # Default SELFIES alphabet
            alphabet_set = {
                '[C]', '[N]', '[O]', '[S]', '[P]', '[F]', '[Cl]', '[Br]', '[I]',
                '[=C]', '[=N]', '[=O]', '[=S]', '[#C]', '[#N]',
                '[Ring1]', '[Ring2]', '[Branch1]', '[Branch2]',
                '[Expl=Ring1]', '[Expl=Ring2]'
            }
        
        alphabet_set.add(self.padding_symbol)
        self.vocabulary = sorted(list(alphabet_set))
        
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.vocabulary)}
        self.idx_to_symbol = {idx: symbol for symbol, idx in self.symbol_to_idx.items()}
        self.vocab_size = len(self.vocabulary)
    
    def _extract_activities(self):
        """Extract molecular activities if available."""
        if hasattr(self, 'df') and self.activity_column in self.df.columns:
            self.activities = self.df[self.activity_column].fillna(
                self.df[self.activity_column].mean()
            ).values
        else:
            self.activities = np.ones(len(self.selfies_list))
    
    @property
    def max_length(self) -> int:
        """Maximum length of SELFIES sequences."""
        if self.selfies_list:
            return max(sf.len_selfies(selfies) for selfies in self.selfies_list)
        return 50  # Default max length
    
    def split_selfies(self, selfies: str) -> List[str]:
        """Split SELFIES string into symbols.
        
        Args:
            selfies: SELFIES string to split
            
        Returns:
            List of SELFIES symbols
        """
        return list(sf.split_selfies(selfies))
    
    def pad_selfies(self, selfies: str, max_length: Optional[int] = None) -> str:
        """Add padding to SELFIES string.
        
        Args:
            selfies: SELFIES string to pad
            max_length: Target length (uses self.max_length if None)
            
        Returns:
            Padded SELFIES string
        """
        if max_length is None:
            max_length = self.max_length
            
        current_length = sf.len_selfies(selfies)
        n_padding = max_length - current_length
        
        if n_padding <= 0:
            return selfies
            
        padding = self.padding_symbol * n_padding
        
        if self.padding_mode == "before":
            return padding + selfies
        else:
            return selfies + padding
    
    def encode_selfies(self, selfies: str, max_length: Optional[int] = None) -> np.ndarray:
        """Encode SELFIES string to integer sequence.
        
        Args:
            selfies: SELFIES string to encode
            max_length: Maximum sequence length
            
        Returns:
            Encoded integer sequence
        """
        if max_length is None:
            max_length = self.max_length
            
        padded_selfies = self.pad_selfies(selfies, max_length)
        symbols = self.split_selfies(padded_selfies)
        
        encoded = []
        for symbol in symbols:
            if symbol in self.symbol_to_idx:
                encoded.append(self.symbol_to_idx[symbol])
            else:
                # Use padding symbol for unknown tokens
                encoded.append(self.symbol_to_idx[self.padding_symbol])
        
        return np.array(encoded, dtype=np.int32)
    
    def decode_sequence(self, sequence: np.ndarray) -> str:
        """Decode integer sequence back to SELFIES string.
        
        Args:
            sequence: Encoded integer sequence
            
        Returns:
            Decoded SELFIES string
        """
        symbols = []
        for idx in sequence:
            if idx in self.idx_to_symbol and self.idx_to_symbol[idx] != self.padding_symbol:
                symbols.append(self.idx_to_symbol[idx])
        
        return ''.join(symbols)
    
    def one_hot_encode(self, selfies: str, max_length: Optional[int] = None) -> np.ndarray:
        """One-hot encode SELFIES string.
        
        Args:
            selfies: SELFIES string to encode
            max_length: Maximum sequence length
            
        Returns:
            One-hot encoded array of shape (max_length, vocab_size)
        """
        if max_length is None:
            max_length = self.max_length
            
        encoded_seq = self.encode_selfies(selfies, max_length)
        one_hot = np.zeros((max_length, self.vocab_size), dtype=np.float32)
        
        for i, symbol_idx in enumerate(encoded_seq):
            one_hot[i, symbol_idx] = 1.0
            
        return one_hot
    
    def selfies_to_smiles(self, selfies: str) -> Optional[str]:
        """Convert SELFIES to SMILES.
        
        Args:
            selfies: SELFIES string
            
        Returns:
            SMILES string or None if conversion fails
        """
        try:
            return sf.decoder(selfies)
        except:
            return None
    
    def smiles_to_selfies(self, smiles: str) -> Optional[str]:
        """Convert SMILES to SELFIES.
        
        Args:
            smiles: SMILES string
            
        Returns:
            SELFIES string or None if conversion fails
        """
        try:
            return sf.encoder(smiles)
        except:
            return None
    
    def validate_selfies(self, selfies: str) -> bool:
        """Validate SELFIES string by converting to SMILES and checking with RDKit.
        
        Args:
            selfies: SELFIES string to validate
            
        Returns:
            True if valid, False otherwise
        """
        smiles = self.selfies_to_smiles(selfies)
        if smiles is None:
            return False
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def validate_samples(self, samples: List[str]) -> List[str]:
        """Validate a list of SELFIES samples.
        
        Args:
            samples: List of SELFIES strings
            
        Returns:
            List of valid SELFIES strings
        """
        valid_samples = []
        for selfies in samples:
            if self.validate_selfies(selfies):
                valid_samples.append(selfies)
        
        return valid_samples


def convert_bitstring_to_molecular_encoding(
    bitstrings: torch.Tensor,
    processor: Union[SmilesProcessor, SelfiesProcessor],
    method: str = "direct"
) -> List[str]:
    """Convert quantum bitstrings to molecular encodings.
    
    Args:
        bitstrings: Tensor of binary strings from quantum circuit
        processor: Molecular processor (SMILES or SELFIES)
        method: Conversion method ("direct", "mapped", "probabilistic")
        
    Returns:
        List of molecular strings
    """
    molecules = []
    
    for bitstring in bitstrings:
        if method == "direct":
            # Direct mapping from bits to vocabulary indices
            indices = []
            bits_per_symbol = int(np.ceil(np.log2(processor.vocab_size)))
            
            for i in range(0, len(bitstring), bits_per_symbol):
                bits = bitstring[i:i+bits_per_symbol]
                # Convert bits to integer
                idx = sum(bit.item() * (2 ** (len(bits) - 1 - j)) for j, bit in enumerate(bits))
                idx = idx % processor.vocab_size  # Ensure valid index
                indices.append(idx)
            
            # Decode to molecular string
            if isinstance(processor, SmilesProcessor):
                molecule = processor.decode_sequence(np.array(indices))
            else:  # SelfiesProcessor
                molecule = processor.decode_sequence(np.array(indices))
                
            molecules.append(molecule)
        
        elif method == "mapped":
            # Map bitstring length to processor vocabulary
            # This is a simplified mapping - in practice, you might want more sophisticated approaches
            n_symbols = min(len(bitstring), processor.max_length)
            indices = []
            
            for i in range(n_symbols):
                # Use multiple bits to determine each symbol
                bit_chunk = bitstring[i:i+3] if i+3 <= len(bitstring) else bitstring[i:]
                idx = sum(bit.item() * (2 ** j) for j, bit in enumerate(bit_chunk)) % processor.vocab_size
                indices.append(idx)
            
            molecule = processor.decode_sequence(np.array(indices))
            molecules.append(molecule)
    
    return molecules
