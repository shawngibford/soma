import os
import sys
from collections import OrderedDict

print(sys.path)

import cloudpickle
import pandas as pd
import tqdm
from more_itertools import chunked_even
from pathos.pools import ProcessPool

from orquestra.drug.discovery import docking


def single_smi_docking(smi: str, protein_name: str, cpu_id: int) -> tuple[str, float]:
    score = docking.perform_calc_single(
        smi, protein_name, docking_program="qvina", cpu_id=cpu_id
    )
    return smi, score


def chunk_smi_docking(
    chunk: tuple[list[str], int], protein_name: str
) -> list[tuple[str, float]]:
    result = []
    cpu_id = chunk[1]
    for smi in tqdm.tqdm(
        chunk[0], desc=f"Docking scores on cpu {cpu_id}", total=len(chunk[0])
    ):
        result.append(single_smi_docking(smi, protein_name, cpu_id))
    return result


def docking_results(
    protein_name="1syh",
    tmp_file="/root/orquestra-drug-discovery/notebook/tmp/ml_export_tmp.csv",
    docking_scores_path="/root/orquestra-drug-discovery/notebook/tmp/"
    "docking_scores.pkl",
    parallel_cpus=1,
):

    if os.path.exists(docking_scores_path):
        delete_keys = []
        with open(docking_scores_path, "rb") as f:
            docking_scores = cloudpickle.load(f)
        for smi, score in tqdm.tqdm(
            docking_scores.items(), desc="Reading docking scores..."
        ):
            if score == 10_000:
                delete_keys.append(smi)
        for key in delete_keys:
            del docking_scores[key]
    else:
        docking_scores = OrderedDict()

    df = pd.read_csv(tmp_file)
    all_scores = []
    unprocessed_smiles = []
    pre_computed_scores = 0
    pbar = tqdm.tqdm(
        df.smiles, desc="Docking score preprocessing...", total=len(df.smiles)
    )
    for smi in pbar:
        if smi in docking_scores:
            pre_computed_scores += 1
            score = docking_scores[smi]
            pbar.set_description(
                f"Pre-computed docking scores: {pre_computed_scores}/{len(df.smiles)}"
            )
            all_scores.append((smi, score))
        else:
            unprocessed_smiles.append(smi)
    pbar.close()

    # parallization process:
    d, m = divmod(len(unprocessed_smiles), parallel_cpus)
    if m != 0:
        d += 1
    chunks = list(chunked_even(unprocessed_smiles, d))

    chunks_with_cpu_id = list(zip(chunks, range(parallel_cpus)))
    print(f"Processing {len(chunks_with_cpu_id)} chunks with {parallel_cpus} cpus")

    def chunk_smi_docking_with_fixed_protein(chunk):
        return chunk_smi_docking(chunk, protein_name)

    with ProcessPool(nodes=len(chunks_with_cpu_id)) as pool:
        results = pool.map(chunk_smi_docking_with_fixed_protein, chunks_with_cpu_id)

    for result in results:
        all_scores.extend(result)
        for smi, score in result:
            docking_scores[smi] = score

    df = pd.DataFrame(all_scores, columns=["smiles", "scores"])
    df = df.sample(frac=1, ignore_index=True)
    # median = df.scores.median()
    # df = df[df.scores < median]
    df.to_csv(tmp_file, index=False)

    with open(docking_scores_path, "wb") as f:
        cloudpickle.dump(docking_scores, f)
