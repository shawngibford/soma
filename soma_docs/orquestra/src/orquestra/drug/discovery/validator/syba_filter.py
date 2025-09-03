# SYBA
from syba.syba import SybaClassifier

from .filter_abstract import FilterAbstract


class SybaFilter(FilterAbstract):
    def __init__(self) -> None:
        self.syba = SybaClassifier()
        self.syba.fitDefaultScore()

    def apply(self, smile: str):
        """
        Predicts the value for the given SMILES string using the SYBA model and returns
        a boolean indciating whether the predicted value is positive.

        Argument:
            - SMILES (str)

        Returns:
            - bool
        """
        value = self.syba.predict(smile)

        # return whether the predicted value is positive
        return value > 0
