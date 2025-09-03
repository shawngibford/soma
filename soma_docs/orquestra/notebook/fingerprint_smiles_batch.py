import sys

import numpy as np
import pandas as pd
import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from selfies import decoder, encoder, get_semantic_robust_alphabet

from orquestra.drug.tartarus_filter import process_molecule


def get_fingerprint(mol: Chem.Mol):
    return AllChem.GetMorganFingerprint(mol, 2)


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
    return Chem.MolToSmiles(
        mol, canonical=False, doRandom=True, isomericSmiles=False, kekuleSmiles=True
    )


def sanitize_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)


def get_fp_score(smiles_back, target_smi):
    target = Chem.MolFromSmiles(target_smi)
    fp_target = get_fingerprint(target)
    mol = Chem.MolFromSmiles(smiles_back)
    fp_mol = get_fingerprint(mol)
    score = TanimotoSimilarity(fp_mol, fp_target)
    return score


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
    N_MAX_TRIES = 5000

    while not valid:
        fail_counter += 1

        alphabet = list(get_semantic_robust_alphabet())

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
        if fail_counter > N_MAX_TRIES:
            return None, None
    return (selfie_mutated, smiles_canon)


def apply_filter(smiles_compound):

    try:
        mol_cal = process_molecule(smiles_compound)
        if mol_cal[1] == "PASS":
            return True
        else:
            return False
    except:
        return False


def main(
    smiles_path: str, cpu_id: int, similarity_limit: float, training_filename: str
):
    smiles = pd.read_csv(smiles_path)["smiles"].tolist()

    # Input smiles can contain repeated smiles.

    smiles_to_mol = {}
    num_mutation_ls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    mutations = []
    for smi in tqdm.tqdm(
        smiles, desc=f"Fingerprinting on cpu {cpu_id}", total=len(smiles)
    ):
        if smi not in smiles_to_mol:
            mol = Chem.MolFromSmiles(smi)
            smiles_to_mol[smi] = mol
        else:
            mol = smiles_to_mol[smi]
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        random_molecule = randomize_smiles(mol)
        selfies = encoder(random_molecule)
        for n_mut in num_mutation_ls:
            selfies_chars = get_selfie_chars(selfies)
            max_molecules_len = len(selfies_chars) + n_mut
            mutated_selfie, _ = mutate_selfie(selfies, max_molecules_len)
            if mutated_selfie is None:
                continue
            mutated_smile = Chem.MolToSmiles(
                Chem.MolFromSmiles(decoder(mutated_selfie)), canonical=True
            )
            if mutated_smile not in smiles and apply_filter(mutated_smile):
                fp_score = get_fp_score(mutated_smile, target_smi=smi)
                if fp_score >= similarity_limit:
                    mutations.append(mutated_smile)
    print(f"On CPU {cpu_id} attempted {len(smiles) * len(num_mutation_ls)} mutations.")
    print(f"Found {len(mutations)} mutations.")
    df = pd.DataFrame({"smiles": mutations})
    df.to_csv(training_filename, index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
