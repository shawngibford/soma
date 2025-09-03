"""Molecular filtering utilities for drug discovery.

This module provides various filters for assessing drug-like properties of molecules,
including Lipinski's Rule of Five, PAINS filters, and synthetic accessibility.
"""

import re
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Optional import for synthetic accessibility scoring
try:
    from syba import SybaClassifier
    HAS_SYBA = True
except ImportError:
    HAS_SYBA = False
    warnings.warn(
        "SYBA package not found. Synthetic accessibility scoring will be disabled. "
        "Install with: pip install syba",
        ImportWarning
    )


class LipinskiFilter:
    """Filter for Lipinski's Rule of Five and related drug-likeness metrics."""
    
    def __init__(
        self,
        mw_max: float = 500.0,
        logp_max: float = 5.0,
        hbd_max: int = 5,
        hba_max: int = 10,
        psa_max: float = 140.0,
        rotatable_bonds_max: int = 10,
        aromatic_rings_max: int = 3,
    ):
        """Initialize Lipinski filter.
        
        Args:
            mw_max: Maximum molecular weight
            logp_max: Maximum logP
            hbd_max: Maximum hydrogen bond donors
            hba_max: Maximum hydrogen bond acceptors
            psa_max: Maximum polar surface area
            rotatable_bonds_max: Maximum rotatable bonds
            aromatic_rings_max: Maximum aromatic rings
        """
        self.mw_max = mw_max
        self.logp_max = logp_max
        self.hbd_max = hbd_max
        self.hba_max = hba_max
        self.psa_max = psa_max
        self.rotatable_bonds_max = rotatable_bonds_max
        self.aromatic_rings_max = aromatic_rings_max
    
    def compute_properties(self, smiles: str) -> Optional[Dict[str, float]]:
        """Compute molecular properties.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of properties or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            properties = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Lipinski.NumHDonors(mol),
                'hba': Lipinski.NumHAcceptors(mol),
                'psa': rdMolDescriptors.CalcTPSA(mol),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'num_rings': rdMolDescriptors.CalcNumRings(mol),
                'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
                'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            }
            
            return properties
        except Exception:
            return None
    
    def passes_lipinski(self, smiles: str) -> bool:
        """Check if molecule passes Lipinski's Rule of Five.
        
        Args:
            smiles: SMILES string
            
        Returns:
            True if passes, False otherwise
        """
        properties = self.compute_properties(smiles)
        if properties is None:
            return False
        
        return (
            properties['molecular_weight'] <= self.mw_max and
            properties['logp'] <= self.logp_max and
            properties['hbd'] <= self.hbd_max and
            properties['hba'] <= self.hba_max
        )
    
    def passes_extended_lipinski(self, smiles: str) -> bool:
        """Check if molecule passes extended Lipinski criteria.
        
        Args:
            smiles: SMILES string
            
        Returns:
            True if passes, False otherwise
        """
        if not self.passes_lipinski(smiles):
            return False
        
        properties = self.compute_properties(smiles)
        if properties is None:
            return False
        
        return (
            properties['psa'] <= self.psa_max and
            properties['rotatable_bonds'] <= self.rotatable_bonds_max and
            properties['aromatic_rings'] <= self.aromatic_rings_max
        )
    
    def lipinski_violations(self, smiles: str) -> int:
        """Count number of Lipinski rule violations.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Number of violations
        """
        properties = self.compute_properties(smiles)
        if properties is None:
            return 4  # Maximum violations
        
        violations = 0
        if properties['molecular_weight'] > self.mw_max:
            violations += 1
        if properties['logp'] > self.logp_max:
            violations += 1
        if properties['hbd'] > self.hbd_max:
            violations += 1
        if properties['hba'] > self.hba_max:
            violations += 1
        
        return violations


class PAINSFilter:
    """Filter for Pan-Assay Interference Compounds (PAINS)."""
    
    def __init__(self):
        """Initialize PAINS filter."""
        # Initialize RDKit filter catalog for PAINS
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        self.catalog = FilterCatalog(params)
        
        # Additional PAINS patterns (common problematic substructures)
        self.additional_pains_patterns = [
            # Quinones
            r'C1=CC(=O)C=CC1=O',
            # Rhodanines
            r'S1C(=O)NC(=S)C1',
            # Enones
            r'C=CC(=O)',
            # Michael acceptors
            r'C=CC(=O)[OH]',
            # Aldehydes (reactive)
            r'[CH]=O',
            # Catechols
            r'c1cc(O)c(O)cc1',
        ]
        
        self.compiled_patterns = [re.compile(pattern) for pattern in self.additional_pains_patterns]
    
    def has_pains_substructure(self, smiles: str) -> bool:
        """Check if molecule contains PAINS substructures.
        
        Args:
            smiles: SMILES string
            
        Returns:
            True if contains PAINS, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return True  # Invalid molecules are considered problematic
            
            # Check RDKit PAINS catalog
            if self.catalog.HasMatch(mol):
                return True
            
            # Check additional PAINS patterns
            for pattern in self.compiled_patterns:
                if pattern.search(smiles):
                    return True
            
            return False
        except Exception:
            return True
    
    def get_pains_matches(self, smiles: str) -> List[str]:
        """Get list of PAINS patterns that match the molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of matching PAINS pattern names
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ["invalid_molecule"]
            
            matches = []
            
            # Check RDKit PAINS catalog
            for match in self.catalog.GetMatches(mol):
                matches.append(match.GetDescription())
            
            # Check additional patterns
            for i, pattern in enumerate(self.compiled_patterns):
                if pattern.search(smiles):
                    matches.append(f"additional_pains_{i}")
            
            return matches
        except Exception:
            return ["error_in_checking"]


class StructuralFilter:
    """Filter for problematic structural features."""
    
    def __init__(
        self,
        max_ring_size: int = 8,
        max_heavy_atoms: int = 50,
        min_heavy_atoms: int = 3,
        allow_metals: bool = False,
    ):
        """Initialize structural filter.
        
        Args:
            max_ring_size: Maximum ring size allowed
            max_heavy_atoms: Maximum number of heavy atoms
            min_heavy_atoms: Minimum number of heavy atoms
            allow_metals: Whether to allow metals
        """
        self.max_ring_size = max_ring_size
        self.max_heavy_atoms = max_heavy_atoms
        self.min_heavy_atoms = min_heavy_atoms
        self.allow_metals = allow_metals
        
        # Problematic elements (typically metals)
        self.problematic_elements = {
            'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',  # Alkali metals
            'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',  # Alkaline earth metals
            'Al', 'Ga', 'In', 'Tl',  # Post-transition metals
            'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',  # Transition metals
            'As', 'Sb', 'Bi',  # Metalloids/heavy elements
            'Pb', 'Hg', 'Cd',  # Heavy metals
        }
    
    def has_large_rings(self, smiles: str) -> bool:
        """Check if molecule has rings larger than max_ring_size.
        
        Args:
            smiles: SMILES string
            
        Returns:
            True if has large rings, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return True
            
            # Get ring info
            ring_info = mol.GetRingInfo()
            
            # Check ring sizes
            for ring in ring_info.AtomRings():
                if len(ring) > self.max_ring_size:
                    return True
            
            return False
        except Exception:
            return True
    
    def has_problematic_atoms(self, smiles: str) -> bool:
        """Check if molecule contains problematic atoms.
        
        Args:
            smiles: SMILES string
            
        Returns:
            True if has problematic atoms, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return True
            
            if not self.allow_metals:
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() in self.problematic_elements:
                        return True
            
            return False
        except Exception:
            return True
    
    def check_atom_count(self, smiles: str) -> bool:
        """Check if molecule has appropriate number of heavy atoms.
        
        Args:
            smiles: SMILES string
            
        Returns:
            True if within range, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            heavy_atoms = mol.GetNumHeavyAtoms()
            return self.min_heavy_atoms <= heavy_atoms <= self.max_heavy_atoms
        except Exception:
            return False
    
    def passes_structural_filters(self, smiles: str) -> bool:
        """Check if molecule passes all structural filters.
        
        Args:
            smiles: SMILES string
            
        Returns:
            True if passes, False otherwise
        """
        return (
            not self.has_large_rings(smiles) and
            not self.has_problematic_atoms(smiles) and
            self.check_atom_count(smiles)
        )


class CompositeFilter:
    """Composite filter combining multiple filtering criteria."""
    
    def __init__(
        self,
        use_lipinski: bool = True,
        use_pains: bool = True,
        use_structural: bool = True,
        lipinski_violations_max: int = 2,
        **kwargs
    ):
        """Initialize composite filter.
        
        Args:
            use_lipinski: Whether to use Lipinski filter
            use_pains: Whether to use PAINS filter
            use_structural: Whether to use structural filter
            lipinski_violations_max: Maximum allowed Lipinski violations
            **kwargs: Additional arguments for individual filters
        """
        self.use_lipinski = use_lipinski
        self.use_pains = use_pains
        self.use_structural = use_structural
        self.lipinski_violations_max = lipinski_violations_max
        
        if use_lipinski:
            self.lipinski_filter = LipinskiFilter(**kwargs.get('lipinski_kwargs', {}))
        
        if use_pains:
            self.pains_filter = PAINSFilter()
        
        if use_structural:
            self.structural_filter = StructuralFilter(**kwargs.get('structural_kwargs', {}))
    
    def filter_molecule(self, smiles: str) -> Tuple[bool, Dict[str, any]]:
        """Apply all filters to a molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Tuple of (passes_all_filters, detailed_results)
        """
        results = {
            'smiles': smiles,
            'valid': True,
            'passes_all': True,
            'lipinski': {},
            'pains': {},
            'structural': {}
        }
        
        try:
            # Check if molecule is valid
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results['valid'] = False
                results['passes_all'] = False
                return False, results
            
            # Lipinski filter
            if self.use_lipinski:
                properties = self.lipinski_filter.compute_properties(smiles)
                violations = self.lipinski_filter.lipinski_violations(smiles)
                
                results['lipinski'] = {
                    'properties': properties,
                    'violations': violations,
                    'passes': violations <= self.lipinski_violations_max,
                    'extended_passes': self.lipinski_filter.passes_extended_lipinski(smiles)
                }
                
                if not results['lipinski']['passes']:
                    results['passes_all'] = False
            
            # PAINS filter
            if self.use_pains:
                has_pains = self.pains_filter.has_pains_substructure(smiles)
                pains_matches = self.pains_filter.get_pains_matches(smiles)
                
                results['pains'] = {
                    'has_pains': has_pains,
                    'matches': pains_matches,
                    'passes': not has_pains
                }
                
                if has_pains:
                    results['passes_all'] = False
            
            # Structural filter
            if self.use_structural:
                passes_structural = self.structural_filter.passes_structural_filters(smiles)
                
                results['structural'] = {
                    'passes': passes_structural,
                    'large_rings': self.structural_filter.has_large_rings(smiles),
                    'problematic_atoms': self.structural_filter.has_problematic_atoms(smiles),
                    'atom_count_ok': self.structural_filter.check_atom_count(smiles)
                }
                
                if not passes_structural:
                    results['passes_all'] = False
            
        except Exception as e:
            results['error'] = str(e)
            results['passes_all'] = False
            return False, results
        
        return results['passes_all'], results
    
    def filter_molecules(self, smiles_list: List[str]) -> Tuple[List[str], pd.DataFrame]:
        """Filter a list of molecules.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (filtered_smiles, detailed_results_df)
        """
        filtered_smiles = []
        detailed_results = []
        
        for smiles in smiles_list:
            passes, results = self.filter_molecule(smiles)
            detailed_results.append(results)
            
            if passes:
                filtered_smiles.append(smiles)
        
        results_df = pd.DataFrame(detailed_results)
        
        return filtered_smiles, results_df
    
    def get_statistics(self, results_df: pd.DataFrame) -> Dict[str, any]:
        """Get filtering statistics.
        
        Args:
            results_df: Results DataFrame from filter_molecules
            
        Returns:
            Dictionary of statistics
        """
        total = len(results_df)
        passed = results_df['passes_all'].sum()
        
        stats = {
            'total_molecules': total,
            'passed_all_filters': passed,
            'pass_rate': passed / total if total > 0 else 0,
            'invalid_molecules': (~results_df['valid']).sum(),
        }
        
        if self.use_lipinski:
            stats['lipinski_pass_rate'] = results_df['lipinski'].apply(
                lambda x: x.get('passes', False) if isinstance(x, dict) else False
            ).mean()
        
        if self.use_pains:
            stats['pains_pass_rate'] = results_df['pains'].apply(
                lambda x: x.get('passes', False) if isinstance(x, dict) else False
            ).mean()
        
        if self.use_structural:
            stats['structural_pass_rate'] = results_df['structural'].apply(
                lambda x: x.get('passes', False) if isinstance(x, dict) else False
            ).mean()
        
        return stats


# Convenience functions
def quick_lipinski_filter(smiles_list: List[str], max_violations: int = 2) -> List[str]:
    """Quick Lipinski filtering of SMILES list.
    
    Args:
        smiles_list: List of SMILES strings
        max_violations: Maximum allowed violations
        
    Returns:
        Filtered SMILES list
    """
    lipinski_filter = LipinskiFilter()
    filtered = []
    
    for smiles in smiles_list:
        violations = lipinski_filter.lipinski_violations(smiles)
        if violations <= max_violations:
            filtered.append(smiles)
    
    return filtered


def quick_pains_filter(smiles_list: List[str]) -> List[str]:
    """Quick PAINS filtering of SMILES list.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Filtered SMILES list
    """
    pains_filter = PAINSFilter()
    filtered = []
    
    for smiles in smiles_list:
        if not pains_filter.has_pains_substructure(smiles):
            filtered.append(smiles)
    
    return filtered


def apply_all_filters(smiles_list: List[str], **kwargs) -> List[str]:
    """Apply all filters to SMILES list.
    
    Args:
        smiles_list: List of SMILES strings
        **kwargs: Arguments for CompositeFilter
        
    Returns:
        Filtered SMILES list
    """
    composite_filter = CompositeFilter(**kwargs)
    filtered_smiles, _ = composite_filter.filter_molecules(smiles_list)
    return filtered_smiles
