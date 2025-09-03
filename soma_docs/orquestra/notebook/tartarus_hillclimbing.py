#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:23:31 2023

@author: akshat
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import selfies
import tqdm
from pathos.pools import ProcessPool
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Mol
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from selfies import decoder, encoder

from orquestra.drug.tartarus_filter import process_molecule

# from syba.syba import SybaClassifier

# start_time = time.time()
# syba = SybaClassifier()
# syba.fitDefaultScore()
# print('Syba fitting time: ', time.time()-start_time)

__this_file__ = Path(__file__).resolve()

RDLogger.DisableLog("rdApp.*")

# construct absolute base path
base_path = __this_file__.parent.parent.resolve()

# defining the paths to the following requried csv files.
mcf_path = os.path.join(base_path, "src/orquestra/database/druglikeness/mcf.csv")
pains_csv_path = os.path.join(
    base_path, "src/orquestra/database/druglikeness/wehi_pains.csv"
)
pains_txt_path = os.path.join(
    base_path, "src/orquestra/database/druglikeness/pains.txt"
)

# read the files ino a pandas df
_mcf = pd.read_csv(mcf_path)
_pains = pd.read_csv(pains_csv_path, names=["smarts", "names"])


_filters = [
    Chem.MolFromSmarts(x) for x in pd.concat([_mcf, _pains], sort=True)["smarts"].values
]


def passes_wehi_mcf(smi: str):
    """
    Check if a given SMILES string passes the EHI and MCF filters

    Arguments:
        - smi (str): The SMILES string of the molecule that is passed

    Returns:
        - bool
    """
    mol = Chem.MolFromSmiles(smi)
    h_mol = Chem.AddHs(mol)
    # returning True if the molecules does not match any of the SMARTS filters, otherwise False
    return not any(h_mol.HasSubstructMatch(smarts) for smarts in _filters)


# Opening the pains.txt file and processing its contents.
with open(pains_txt_path, "r") as inf:
    # Read the file line by line, stripping whitespace and splitting by space
    sub_strct = [line.rstrip().split(" ") for line in inf]
    # Extract the first/second elements from each line to form the 'smarts' and 'desc' lists
    smarts = [line[0] for line in sub_strct]
    desc = [line[1] for line in sub_strct]
    dic = dict(zip(smarts, desc))


def pains_filt(mol):

    for k, v in dic.items():
        subs = Chem.MolFromSmarts(k)
        if subs != None:
            if mol.HasSubstructMatch(subs):
                mol.SetProp(v, k)
    return [prop for prop in mol.GetPropNames()]


def benchmark_filter(smiles_compound, max_mol_weight: float = 800):
    pass_all = []
    for smile_ in smiles_compound:
        try:
            mol_cal = process_molecule(smile_)
            if mol_cal[1] == "PASS":
                pass_all.append(smile_)
        except:
            pass
    return pass_all


def apply_filter(smiles_compound, max_mol_weight: float = 800):

    try:
        mol_cal = process_molecule(smiles_compound)
        if mol_cal[1] == "PASS":
            return True
        else:
            return False
    except:
        return False


class _FingerprintCalculator:
    """Calculate the fingerprint for a molecule, given the fingerprint type
    Parameters:
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object (None if invalid smile string smi)
        fp_type (string)            :Fingerprint type  (choices: AP/PHCO/BPF,BTF,PAT,ECFP4,ECFP6,FCFP4,FCFP6)
    Returns:
        RDKit fingerprint object
    """

    def get_fingerprint(self, mol: Mol, fp_type: str):
        method_name = "get_" + fp_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f"{fp_type} is not a supported fingerprint type.")
        return method(mol)

    def get_AP(self, mol: Mol):
        return AllChem.GetAtomPairFingerprint(mol, maxLength=10)

    def get_PHCO(self, mol: Mol):
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

    def get_BPF(self, mol: Mol):
        return GetBPFingerprint(mol)

    def get_BTF(self, mol: Mol):
        return GetBTFingerprint(mol)

    def get_PATH(self, mol: Mol):
        return AllChem.RDKFingerprint(mol)

    def get_ECFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2)

    def get_ECFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3)

    def get_FCFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)

    def get_FCFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)


def get_fingerprint(mol: Mol, fp_type: str):
    """Fingerprint getter method. Fingerprint is returned after using object of
        class '_FingerprintCalculator'

    Parameters:
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object (None if invalid smile string smi)
        fp_type (string)            :Fingerprint type  (choices: AP/PHCO/BPF,BTF,PAT,ECFP4,ECFP6,FCFP4,FCFP6)
    Returns:
        RDKit fingerprint object

    """
    return _FingerprintCalculator().get_fingerprint(mol=mol, fp_type=fp_type)


def randomize_smiles(mol):
    """Returns a random (dearomatized) SMILES given an rdkit mol object of a molecule.
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) :  RdKit mol object (None if invalid smile string smi)

    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object  (None if invalid smile string smi)
    """
    if not mol:
        return None

    Chem.Kekulize(mol)
    return rdkit.Chem.MolToSmiles(
        mol, canonical=False, doRandom=True, isomericSmiles=False, kekuleSmiles=True
    )


def get_fp_scores(smiles_back, target_smi, fp_type):
    """Calculate the Tanimoto fingerprint (using fp_type fingerint) similarity between a list
       of SMILES and a known target structure (target_smi).

    Parameters:
    smiles_back   (list) : A list of valid SMILES strings
    target_smi (string)  : A valid SMILES string. Each smile in 'smiles_back' will be compared to this stucture
    fp_type (string)     : Type of fingerprint  (choices: AP/PHCO/BPF,BTF,PAT,ECFP4,ECFP6,FCFP4,FCFP6)

    Returns:
    smiles_back_scores (list of floats) : List of fingerprint similarities
    """
    smiles_back_scores = []
    target = Chem.MolFromSmiles(target_smi)

    fp_target = get_fingerprint(target, fp_type)

    for item in smiles_back:
        mol = Chem.MolFromSmiles(item)
        fp_mol = get_fingerprint(mol, fp_type)
        score = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores


def get_selfie_chars(selfie):
    """Obtain a list of all selfie characters in string selfie

    Parameters:
    selfie (string) : A selfie string - representing a molecule

    Example:
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']

    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    """
    chars_selfie = []  # A list of all SELFIE sybols from string selfie
    while selfie != "":
        chars_selfie.append(selfie[selfie.find("[") : selfie.find("]") + 1])
        selfie = selfie[selfie.find("]") + 1 :]
    return chars_selfie


def mutate_selfie(selfie, max_molecules_len, write_fail_cases=False):
    """Return a mutated selfie string (only one mutation on slefie is performed)

    Mutations are done until a valid molecule is obtained
    Rules of mutation: With a 33.3% propbabily, either:
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another
        3. Delete a random character

    Parameters:
    selfie            (string)  : SELFIE string to be mutated
    max_molecules_len (int)     : Mutations of SELFIE string are allowed up to this length
    write_fail_cases  (bool)    : If true, failed mutations are recorded in "selfie_failure_cases.txt"

    Returns:
    selfie_mutated    (string)  : Mutated SELFIE string
    smiles_canon      (string)  : canonical smile of mutated SELFIE string
    """
    valid = False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)

    while not valid:
        fail_counter += 1

        alphabet = list(selfies.get_semantic_robust_alphabet())

        choice_ls = [1, 2, 3]  # 1=Insert; 2=Replace; 3=Delete
        random_choice = np.random.choice(choice_ls, 1)[0]

        # Insert a character in a Random Location
        if random_choice == 1:
            random_index = np.random.randint(len(chars_selfie) + 1)
            random_character = np.random.choice(alphabet, size=1)[0]

            selfie_mutated_chars = (
                chars_selfie[:random_index]
                + [random_character]
                + chars_selfie[random_index:]
            )

        # Replace a random character
        elif random_choice == 2:
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[
                    random_index + 1 :
                ]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index]
                    + [random_character]
                    + chars_selfie[random_index + 1 :]
                )

        # Delete a random character
        elif random_choice == 3:
            random_index = np.random.randint(len(chars_selfie))
            if random_index == 0:
                selfie_mutated_chars = chars_selfie[random_index + 1 :]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index] + chars_selfie[random_index + 1 :]
                )

        else:
            raise Exception("Invalid Operation trying to be performed")

        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        sf = "".join(x for x in chars_selfie)

        try:
            smiles = decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(selfie_mutated_chars) > max_molecules_len or smiles_canon == "":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except:
            valid = False
            if fail_counter > 1 and write_fail_cases == True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write(
                    "Tried to mutate SELFIE: "
                    + str(sf)
                    + " To Obtain: "
                    + str(selfie_mutated)
                    + "\n"
                )
                f.close()

    return (selfie_mutated, smiles_canon)


def get_mutated_SELFIES(selfies_ls, num_mutations):
    """Mutate all the SELFIES in 'selfies_ls' 'num_mutations' number of times.

    Parameters:
    selfies_ls   (list)  : A list of SELFIES
    num_mutations (int)  : number of mutations to perform on each SELFIES within 'selfies_ls'

    Returns:
    selfies_ls   (list)  : A list of mutated SELFIES

    """
    for _ in range(num_mutations):
        selfie_ls_mut_ls = []
        for str_ in selfies_ls:

            str_chars = get_selfie_chars(str_)
            max_molecules_len = len(str_chars) + num_mutations

            selfie_mutated, _ = mutate_selfie(str_, max_molecules_len)
            selfie_ls_mut_ls.append(selfie_mutated)

        selfies_ls = selfie_ls_mut_ls.copy()
    return selfies_ls


def sanitize_smiles(smi):
    """Return a canonical smile representation of smi
    Parameters:
    smi (string) : smile string to be canonicalized
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful
    """
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)


def get_frags(smi, radius):
    """Create fragments from smi with some radius. Remove duplicates and any
    fragments that are blank molecules.
    """
    mol = smi2mol(smi, sanitize=True)
    frags = []
    for ai in range(mol.GetNumAtoms()):
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, ai)
        amap = {}
        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
        frag = mol2smi(submol, isomericSmiles=False, canonical=True)
        frags.append(frag)
    return list(filter(None, list(set(frags))))


def form_fragments(smi):
    """Create fragments of certain radius. Returns a list of fragments
    using SELFIES characters.
    """
    selfies_frags = []
    unique_frags = get_frags(smi, radius=3)
    for item in unique_frags:
        sf = encoder(item)
        if sf is None:
            continue
        dec_ = decoder(sf)

        try:
            m = Chem.MolFromSmiles(dec_)
            Chem.Kekulize(m)
            dearom_smiles = Chem.MolToSmiles(
                m, canonical=False, isomericSmiles=False, kekuleSmiles=True
            )
            dearom_mol = Chem.MolFromSmiles(dearom_smiles)
        except:
            continue

        if dearom_mol == None:
            raise Exception("mol dearom failes")

        selfies_frags.append(encoder(dearom_smiles))

    return selfies_frags


# df.to_csv(f".tmp/ml_export_{protein_name}.csv")


def hill_climbing(
    similarity_limit,
    num_random_samples,
    docking_filename,
    training_filename,
    parallel_cpus=1,
    top_molecules=30,
    training_dataset_size_limit: int | None = None,
):
    import os
    import pathlib
    import subprocess
    import tempfile

    from more_itertools import chunked_even

    current_path = pathlib.Path(__file__).parent.absolute()
    finger_print_path = os.path.join(current_path, "fingerprint_smiles_batch.py")

    df = pd.read_csv(docking_filename)
    smiles_all = (
        df.nsmallest(top_molecules, "scores").reset_index(drop=True).smiles.to_list()
    )
    # Repeat each SMILES num_random_samples times:
    smiles_all = [item for item in smiles_all for _ in range(num_random_samples)]
    np.random.shuffle(smiles_all)

    # parallelization process:
    d, m = divmod(len(smiles_all), parallel_cpus)
    if m != 0:
        d += 1
    chunks = list(chunked_even(smiles_all, d))

    chunks_with_cpu_id = list(zip(chunks, range(parallel_cpus)))
    print(f"Processing {len(chunks_with_cpu_id)} chunks with {parallel_cpus} cpus")

    def launch_chunk(chunk: tuple[list[str], int, str, str]):
        actual_chunk, cpu_id, input_smiles_path, output_smiles_path = chunk
        df = pd.DataFrame({"smiles": actual_chunk})
        df.to_csv(input_smiles_path, index=False)
        subprocess.run(
            [
                "taskset",
                "-c",
                str(cpu_id),
                "python",
                finger_print_path,
                "--smiles_path",
                input_smiles_path,
                "--cpu_id",
                str(cpu_id),
                "--similarity_limit",
                str(similarity_limit),
                "--training_filename",
                output_smiles_path,
            ]
        )

    resulting_smiles = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        chunks_with_cpu_id_and_paths = []
        for chunk, cpu_id in chunks_with_cpu_id:
            input_smiles_path = os.path.join(tmpdirname, f"input_{cpu_id}.csv")
            output_smiles_path = os.path.join(tmpdirname, f"output_{cpu_id}.csv")
            chunks_with_cpu_id_and_paths.append(
                (chunk, cpu_id, input_smiles_path, output_smiles_path)
            )
        with ProcessPool(parallel_cpus) as pool:
            pool.map(launch_chunk, chunks_with_cpu_id_and_paths)
            for filename in os.listdir(tmpdirname):
                if filename.startswith("output_"):
                    df = pd.read_csv(os.path.join(tmpdirname, filename))
                    resulting_smiles.extend(df.smiles.to_list())

    all_smiles = list(set(resulting_smiles))
    np.random.shuffle(all_smiles)
    if training_dataset_size_limit is not None:
        all_smiles = all_smiles[:training_dataset_size_limit]
    df = pd.DataFrame({"smiles": all_smiles})
    df.to_csv(training_filename, index=False)


def hill_climbing_(
    protein_name,
    similarity_limit,
    num_random_samples,
    docking_filename,
    training_filename,
):
    df_smiles = pd.read_csv(docking_filename)

    new_data = df_smiles.nsmallest(30, "scores").reset_index(drop=True)
    fp_type = "ECFP4"  # Extended-Connectivity Fingerprints with a diameter of 4
    smiles_all = new_data.smiles.to_list()
    pass_all = []
    with tqdm.tqdm(total=len(smiles_all)) as pbar:
        for i, smi in enumerate(smiles_all):
            pbar.set_description(
                f"HillClimbing: Filtered {i} / {len(smiles_all)}. passed= {len(pass_all)}"
            )
            print(smi)
            pass_filter = apply_filter(smi)  # This should be always true
            if pass_filter:
                print(i, smi)
                total_time = time.time()
                # num_random_samples = 20000 # TODO
                num_mutation_ls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise Exception("Invalid starting structure encountered")

                start_time = time.time()
                randomized_smile_orderings = [
                    randomize_smiles(mol) for _ in range(num_random_samples)
                ]

                # Convert all the molecules to SELFIES
                selfies_ls = [encoder(x) for x in randomized_smile_orderings]

                all_smiles_collect = []

                for num_mutations in num_mutation_ls:
                    # Mutate the SELFIES:
                    total_time = time.time()
                    selfies_mut = get_mutated_SELFIES(
                        selfies_ls.copy(), num_mutations=num_mutations
                    )

                    # Convert back to SMILES:
                    smiles_back = [decoder(x) for x in selfies_mut]
                    all_smiles_collect = all_smiles_collect + smiles_back

                all_smiles_collect = list(set(all_smiles_collect))

                # Convert all molecules to canonical smiles:
                all_smiles_collect = [
                    rdkit.Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
                    for smi in all_smiles_collect
                ]
                all_smiles_collect = list(set(all_smiles_collect))

                for smi_check in all_smiles_collect:

                    if apply_filter(smi_check) and smi_check not in pass_all:
                        fp_score = get_fp_scores(
                            [smi_check], target_smi=smi, fp_type=fp_type
                        )[0]
                        if fp_score >= similarity_limit:
                            pass_all.append(smi_check)

            i += 1
            pbar.update()

    pass_all = list(set(pass_all))

    df = pd.DataFrame({"smiles": pass_all})
    df.to_csv(training_filename, index=False)
    return pass_all
