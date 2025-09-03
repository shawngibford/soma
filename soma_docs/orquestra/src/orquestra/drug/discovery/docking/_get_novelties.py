import os

import cloudpickle
from rdkit.Chem import DataStructs


def calculate_similarity(fingerprint, reference_fingerprint):
    return DataStructs.TanimotoSimilarity(fingerprint, reference_fingerprint)


def _get_novelty(fingerprint, reference_fingerprints: list, threshold: float):
    similarities = [
        calculate_similarity(fingerprint, ref_fp) for ref_fp in reference_fingerprints
    ]
    return max(similarities) < threshold


def get_novelties_with_file(
    fingerprints_filepath: str, reference_filepath: str, threshold: float, chunk_id: int
):
    with open(reference_filepath, "rb") as f:
        reference_fingerprints = cloudpickle.load(f)
    with open(fingerprints_filepath, "rb") as f:
        fingerprints = cloudpickle.load(f)
    results = []
    for fingerprint in fingerprints:
        if fingerprint is None:
            results.append(None)
        else:
            results.append(_get_novelty(fingerprint, reference_fingerprints, threshold))
    # Get parent folder of filepath and save results in a file indexed with the chunk_id
    parent_folder = os.path.dirname(fingerprints_filepath)
    with open(os.path.join(parent_folder, f"novelties_{chunk_id}.pickle"), "wb") as f:
        cloudpickle.dump(results, f)


if __name__ == "__main__":
    from fire import Fire

    Fire(get_novelties_with_file)
