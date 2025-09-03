import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit.Chem as rdc
import rdkit.Chem.Descriptors as rdcd
import rdkit.Chem.rdMolDescriptors as rdcmd
import rdkit.Chem.rdmolops as rdcmo
import tqdm
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors

from .filter_abstract import FilterAbstract

current_script_path = Path(__file__).resolve().parent
database_path = current_script_path.parent.parent.parent

try:
    _mcf = pd.read_csv(os.path.join(f"{database_path}/database/druglikeness/mcf.csv"))
    _pains = pd.read_csv(
        os.path.join(f"{database_path}/database/druglikeness/wehi_pains.csv"),
        names=["smarts", "names"],
    )
    _filters = [
        Chem.MolFromSmarts(x)
        for x in pd.concat([_mcf, _pains], sort=True)["smarts"].values
    ]
    inf = open(f"{database_path}/database/druglikeness/pains.txt", "r")
    sub_strct = [line.rstrip().split(" ") for line in inf]
    smarts = [line[0] for line in sub_strct]
    desc = [line[1] for line in sub_strct]
    dic = dict(zip(smarts, desc))

except FileNotFoundError as e:
    logging.warning(
        f"Unable to locate one or more of the filter files. Continuing without filters, this may lead to unexpected errors. Exception: {e}"
    )


class GeneralFilter(FilterAbstract):
    def __init__(self, max_mol_weight: int = 800) -> None:
        super().__init__()
        self.maximum_ring_size = MaximumRingFilter(maximum_ring_size=8)
        self.filter_phosphorus = PhosphorusFilter()
        self.substructure_violations = SubstructureViolationsFilter()
        self.lipinski_filter = LipinskiFilter(max_mol_weight=max_mol_weight)

    def apply(self, smile: str):
        try:
            if (
                "C-" in smile
                or "N+" in smile
                or "C+" in smile
                or "S+" in smile
                or "S-" in smile
                or "O+" in smile
            ):
                return False
            mol = smi2mol(smile)
            if mol == None:
                return False
            # Added after GDB-13 was filtered to get rid charged molecules
            if rdcmo.GetFormalCharge(mol) != 0:
                # print('Formal charge failed! Value: ', rdcmo.GetFormalCharge(mol))
                return False
            # Added after GDB-13 was filtered to get rid radicals
            elif rdcd.NumRadicalElectrons(mol) != 0:
                # print('rdcd.NumRadicalElectrons(mol) failed! Value: ', rdcd.NumRadicalElectrons(mol))
                return False
            # Filter by bridgehead atoms
            elif rdcmd.CalcNumBridgeheadAtoms(mol) > 2:
                return False
            # Filter by ring size
            elif self.maximum_ring_size.apply(mol) > 8:
                return False
            # Filter by proper phosphorus
            elif self.filter_phosphorus.apply(mol):
                return False
            elif self.substructure_violations.apply(mol):
                return False
            elif self.lipinski_filter.apply(mol) == False:
                return False
            elif rdcmd.CalcNumRotatableBonds(mol) >= 10:
                return False
            else:
                return True
        except FileNotFoundError as e:
            logging.warning(f"unable to filter in apply_filter function: {e}")


class MaximumRingFilter(FilterAbstract):
    def __init__(self, maximum_ring_size: int = 8) -> None:
        super().__init__()
        self.maximum_ring_size = maximum_ring_size

    def apply(self, mol: Chem.Mol):
        """
        Calculate maximum ring size of molecule
        """
        cycles = mol.GetRingInfo().AtomRings()
        if len(cycles) == 0:
            maximum_ring_size = 0
        else:
            maximum_ring_size = max([len(ci) for ci in cycles])

        if maximum_ring_size < self.maximum_ring_size:
            return True
        else:
            return False


class PhosphorusFilter(FilterAbstract):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, mol: Chem.Mol):
        """
        Check for presence of phopshorus fragment
        Return True: contains proper phosphorus
        Return False: contains improper phosphorus
        """
        mol = self.check_mol(mol)
        violation = False

        if mol.HasSubstructMatch(rdc.MolFromSmarts("[P,p]")) == True:
            if mol.HasSubstructMatch(rdc.MolFromSmarts("*~[P,p](=O)~*")) == False:
                violation = True

        return violation


class SubstructureViolationsFilter(FilterAbstract):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, mol: Chem.Mol):
        """
        Check for substructure violates
        Return True: contains a substructure violation
        Return False: No substructure violation
        """
        mol = self.check_mol(mol)
        violation = False
        forbidden_fragments = [
            "[S&X3]",
            "[S&X4]",
            "[S&X6]",
            "[S&X2]",
            "[S&X1]",
            "*1=**=*1",
            "*1*=*=*1",
            "*1~*=*1",
            "[F,Cl,Br]C=[O,S,N]",
            "[Br]-C-C=[O,S,N]",
            "[N,n,S,s,O,o]C[F,Cl,Br]",
            "[I]",
            "[S&X3]",
            "[S&X5]",
            "[S&X6]",
            "[B,N,n,O,S]~[F,Cl,Br,I]",
            "*=*=*=*",
            "*=[NH]",
            "[P,p]~[F,Cl,Br]",
            "SS",
            "C#C",
            "C=C=C",
            "*=*=*",
            "NNN",
            "[R3R]",
            "[R4R]",
        ]

        for ni in range(len(forbidden_fragments)):
            if (
                mol.HasSubstructMatch(rdc.MolFromSmarts(forbidden_fragments[ni]))
                == True
            ):
                violation = True
                # print('Substruct violation is: ', forbidden_fragments[ni])
                break
            else:
                continue

        return violation


class LipinskiFilter(FilterAbstract):
    def __init__(self, max_mol_weight: int = 800) -> None:
        super().__init__()
        self.max_mol_weight = max_mol_weight

    def apply(self, mol: Chem.Mol):
        try:
            mol = self.check_mol(mol)
            return (
                MolLogP(mol) <= 5
                and NumHAcceptors(mol) <= 10
                and NumHDonors(mol) <= 5
                and 300 <= ExactMolWt(mol) <= self.max_mol_weight
            )
        except:
            return False


class PainFilter(FilterAbstract):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, mol: Chem.Mol):
        mol = self.check_mol(mol)
        for k, v in dic.items():
            subs = Chem.MolFromSmarts(k)
            if subs != None:
                if mol.HasSubstructMatch(subs):
                    mol.SetProp(v, k)
        if len([prop for prop in mol.GetPropNames()]) == 0:
            return True
        else:
            return False


class WehiMCFilter(FilterAbstract):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, smile: str):
        mol = self.check_mol(smile)
        h_mol = Chem.AddHs(mol)
        if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
            return False
        else:
            return True


# def combine_filter(
#     smiles_compound, max_mol_weight: float = 800, filter_fc=filters
# ):
#     # syba imports take a while move them here to only import when needed

#     pass_all = []
#     i = 0

#     with tqdm.tqdm(total=len(smiles_compound)) as pbar:
#         for smile_ in smiles_compound:
#             pbar.set_description(
#                 f"Filtered {i} / {len(smiles_compound)}. passed={len(pass_all)},frac={len(pass_all)/len(smiles_compound)}"
#             )
#             try:
#                 if (
#                     filter_fc(smile_, max_mol_weight)
#                     and smile_ not in pass_all
#                     and (syba.apply(smile_))
#                     and passes_wehi_mcf(smile_)
#                     and (len(pains_filt(Chem.MolFromSmiles(smile_))) == 0)
#                 ):
#                     pass_all.append(smile_)
#             except Exception as e:
#                 print(
#                     f"The following error occurred during the `combine_filter` step: {e}"
#                 )

#             i += 1
#             pbar.update()
#     return pass_all
