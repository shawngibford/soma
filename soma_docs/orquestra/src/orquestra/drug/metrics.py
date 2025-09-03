import os
import sys
from typing import Callable, List, Sequence, Set

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem, DataStructs, RDConfig

# Add the path to the SA_Score module
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

# Now you can import sascorer
import sascorer


def get_diversity(smiles_ls):

    pred_mols = [Chem.MolFromSmiles(s) for s in smiles_ls]
    pred_mols = [x for x in pred_mols if x is not None]
    pred_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in pred_mols]

    similarity = 0
    for i in range(len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], pred_fps[:i])
        similarity += sum(sims)

    n = len(pred_fps)
    n_pairs = n * (n - 1) / 2
    diversity = 1 - (similarity / n_pairs)

    return diversity * 100


class MoleculeNovelty:
    """
    A class to calculate the novelty of a molecule based on a defined dataset of reference molecules.

    Attributes:
        reference_fingerprints (list): A list of fingerprints of the reference molecules.

    Methods:
        smiles_to_fingerprint(smiles, radius=2, nBits=2048): Converts a SMILES string to a molecular fingerprint.
        calculate_similarity(fingerprint, reference_fingerprint): Calculates the Tanimoto similarity between two fingerprints.
        assess_novelty(smiles, threshold=0.3): Assesses the novelty of a molecule based on similarity scores.
    Example:
        # Initialize with a reference dataset of SMILES strings
        reference_smiles = ["CCO", "Nc1ccccc1", "CCN(CC)CC"]
        novelty_calculator = MoleculeNovelty(reference_smiles)

        # Assess the novelty of a molecule
        smiles_to_test = "CCO"
        is_novel = novelty_calculator.assess_novelty(smiles_to_test)
        print(f"Is the molecule novel? {'Yes' if is_novel else 'No'}")
    """

    def __init__(self, reference_smiles, threshold=0.3):
        """
        Initializes the MoleculeNovelty with a list of reference SMILES strings.

        Parameters:
            reference_smiles (list): A list of SMILES strings for the reference molecules.
        """
        self.threshold = threshold
        self.reference_fingerprints = [
            self.smiles_to_fingerprint(smiles) for smiles in reference_smiles
        ]

    def smiles_to_fingerprint(self, smiles, radius=2, nBits=2048):
        """
        Converts a SMILES string to a molecular fingerprint using the Morgan algorithm.

        Parameters:
            smiles (str): The SMILES string of the molecule.
            radius (int): The radius parameter for the Morgan algorithm. Default is 2.
            nBits (int): The size of the fingerprint. Default is 2048.

        Returns:
            RDKit ExplicitBitVect: The computed fingerprint of the molecule.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:  # Invalid SMILES
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)

    def calculate_similarity(self, fingerprint, reference_fingerprint):
        """
        Calculates the Tanimoto similarity between two fingerprints.

        Parameters:
            fingerprint (RDKit ExplicitBitVect): The fingerprint of the target molecule.
            reference_fingerprint (RDKit ExplicitBitVect): The fingerprint of a reference molecule.

        Returns:
            float: The Tanimoto similarity score between the two fingerprints.
        """
        return DataStructs.TanimotoSimilarity(fingerprint, reference_fingerprint)

    def assess_novelty(self, smiles, threshold=0.3):
        """
        Assesses the novelty of a molecule based on the maximum similarity score to the reference dataset.

        Parameters:
            smiles (str): The SMILES string of the molecule to assess.
            threshold (float): The similarity threshold below which a molecule is considered novel. Default is 0.3.

        Returns:
            bool: True if the molecule is considered novel, False otherwise.
        """
        target_fp = self.smiles_to_fingerprint(smiles)
        if target_fp is None:
            return np.nan  # Invalid input SMILES
        similarities = [
            self.calculate_similarity(target_fp, ref_fp)
            for ref_fp in self.reference_fingerprints
        ]
        max_similarity = max(similarities)
        return max_similarity < threshold

    def __call__(self, smiles):
        return self.assess_novelty(smiles, self.threshold)

    def get_novelity_smiles(self, smiles: List, threshold=0.3):
        novelity_fraction = (
            100
            * np.nansum([self.assess_novelty(x, threshold) for x in smiles])
            / len(smiles)
        )
        return novelity_fraction


def calculate_sa_score(smiles):
    """
    Calculate the Synthetic Accessibility (SA) score for a given molecule.

    Parameters:
    smiles (str): The SMILES string representing the molecule.

    Returns:
    float value:
        - sa_score (float): The SA score, which estimates the ease of synthesis of the molecule. Lower values indicate easier synthesis.

    If the SMILES string is invalid, the function returns (None, None).

    Example:
    >>> smiles = "NC(=O)c1ccccc1"
    >>> sa_score calculate_sa_score(smiles)
    >>> print(f"SA Score: {sa_score}")

    Notes:
    - The SA score is computed using RDKit's contributed SA_Score module.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        sa_score = sascorer.calculateScore(mol)
        return sa_score
    else:
        return None


def calculate_qed_score(smiles):
    """
    Calculate the Synthetic Accessibility (SA) score and the Quantitative Estimation of Drug-likeness (QED) score for a given molecule.

    Parameters:
    smiles (str): The SMILES string representing the molecule.

    Returns:
    float value:
        - qed_score (float): The QED score, which estimates the drug-likeness of the molecule. The score ranges from 0 (not drug-like) to 1 (very drug-like).

    If the SMILES string is invalid, the function returns (None, None).

    Example:
    >>> smiles = "NC(=O)c1ccccc1"
    >>> qed_score = calculate_qed_score(smiles)
    >>> print(f"QED Score: {qed_score}")

    Notes:
    - The QED score is computed using RDKit's built-in QED functionality.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        qed_score = QED.qed(mol)
        return qed_score
    else:
        return None


def load_aromatic_substructures(file_path):
    """
    Loads aromatic substructure SMARTS patterns from a text file.

    Parameters:
    file_path (str): Path to the text file containing SMARTS patterns.

    Returns:
    list: A list of RDKit molecule objects representing the substructures.

    EXAMPLE:
    substructures = load_aromatic_substructures('aromatic_substructures.txt')
    ```
    c1ccccc1  # Benzene
    c1ccncc1  # Pyridine
    c1ccc2ccccc2c1  # Naphthalene
    ```
    """
    substructures = []
    with open(file_path, "r") as file:
        for line in file:
            # Remove comments and strip whitespace
            pattern = line.split("#")[0].strip()
            if pattern:
                # Convert the SMARTS pattern to an RDKit molecule
                substructure = Chem.MolFromSmarts(pattern)
                if substructure:
                    substructures.append(substructure)
    return substructures


def filter_aromatic_molecules(smiles, substructures):
    """
    Filters a molecule based on the presence of specified aromatic substructures.

    Parameters:
    smiles (str): The SMILES string representing the molecule.
    substructures (list): A list of RDKit molecule objects representing the substructures.

    Returns:
    bool: True if the molecule matches any of the specified substructures, False otherwise.

    EXAMPLE:
    substructures = load_aromatic_substructures('aromatic_substructures.txt')

    3Define your SMILES string to be filtered
    smiles = "c1ccccc1"  # Example: Benzene

    #Check if the molecule contains any of the defined aromatic substructures
    if filter_aromatic_molecules(smiles, substructures):
        print(f"The molecule {smiles} contains one of the specified aromatic substructures.")
    else:
        print(f"The molecule {smiles} does NOT contain any of the specified aromatic substructures.")

    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        for substructure in substructures:
            if molecule.HasSubstructMatch(substructure):
                return True
    return False
