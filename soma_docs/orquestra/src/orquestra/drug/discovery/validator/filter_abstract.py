from abc import ABC, abstractmethod

from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol


class FilterAbstract(ABC):
    def __init__(self) -> None:
        super().__init__()

    def check_mol(self, input):
        if isinstance(input, Chem.Mol):
            return input
        elif isinstance(input, str):
            return smi2mol(input)
        else:
            raise TypeError("It is not Mol|Smile")

    @abstractmethod
    def apply(self, smile: str):
        """Applies custom filter to a smile

        Args:
            smile: a string of smile.
        """
        raise NotImplementedError
