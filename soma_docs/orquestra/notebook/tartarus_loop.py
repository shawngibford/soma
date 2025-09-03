import os
import tempfile

import cloudpickle
import mlflow
import pandas as pd
import torch
from tartarus_docking import docking_results
from tartarus_hillclimbing import hill_climbing
from train_lstm import codecarbon_lstm_model

from orquestra.drug.discovery.docking.utils import process_molecule
from orquestra.drug.discovery.validator.filter_abstract import FilterAbstract
from orquestra.drug.sdk import configure_mlflow

protein_to_dataset_name = {"1syh": "alzheimer", "6y2f": "covid"}


class TartarusFilters(FilterAbstract):
    def apply(self, smile: str):
        _, status = process_molecule(smile)
        if status == "PASS":
            return True
        return False


def run_tartarus_loop(
    N_QUBITS: int,
    GPU_ID: int,
    STATS_SAMPLES: int,
    N_CPUS: int,
    MOLECULE: str,
    MODEL: str,
    prior_args: dict,
    run_id: str,
    similarity_limit: float = 0.7,
    run_name_suffix: str = "",
    skip_loading_from_run_id: bool = False,
    training_dataset_size_limit: int | None = None,
    random_seed: int = 687350,
    training_path="/root/orquestra-drug-discovery/notebook/tmp/16/"
    "docking_hill_climbing_0.csv",
    cluster="lambda02",
    n_epochs=500,
    experiment_name="tartarus_lstm",
    use_vectorized_cost_fn=False,
    use_linear_projection=True,
    merge_prior_fc=None,
    use_kaiming_init=False,
    embedding_dim=224,
    sampling_temperature=None,
    prior_n_epochs=5,
    use_fusednaive_lstm=False,
    n_tartarus_iterations=10,
):
    filter_lists = [TartarusFilters()]
    weight_lists = [1.0]
    tmp_actual_base = "/root/orquestra-drug-discovery/notebook/tmp/"
    tmp_base = f"{tmp_actual_base}{N_QUBITS}/"
    if run_name_suffix != "":
        if not run_name_suffix.startswith("_"):
            run_name_suffix = "_" + run_name_suffix

    for i in range(n_tartarus_iterations):
        docking_path = (
            tmp_base + f"docking_{i}_{MODEL}_{MOLECULE}_60{run_name_suffix}.csv"
        )
        if i == 0 and not skip_loading_from_run_id:
            if not os.path.exists(docking_path):
                configure_mlflow(f"syba_lstm__prior_{MODEL}_16__dataset__{MOLECULE}")
                client = mlflow.tracking.MlflowClient()
                artifacts = client.list_artifacts(run_id=run_id)
                for artifact in artifacts:
                    if "result_and_model" in artifact.path:
                        # Download the artifact to a temporary file:
                        with tempfile.TemporaryDirectory() as tmpdirname:
                            file_name = client.download_artifacts(
                                run_id,
                                artifact.path,
                                tmpdirname,
                            )
                            with open(file_name, "rb") as file_name:
                                result_and_model = cloudpickle.load(file_name)
                            if "as_dict" in file_name.name:
                                lstm = result_and_model["lstm"]
                            else:
                                lstm = result_and_model[1]
                        lstm.to_device(f"cuda:{GPU_ID}")
                        break
                if N_QUBITS == 128:
                    with open(
                        "alzheimer_selfies_smiles_novelty_and_filter.pickle", "rb"
                    ) as f:
                        selfies, _, _, filter = cloudpickle.load(f)
                elif N_QUBITS == 16:
                    with open(
                        "alzheimer selfies_smiles_novelty_and_filter_16_qubits.pickle",
                        "rb",
                    ) as f:
                        selfies, _, _, filter = cloudpickle.load(f)
                elif N_QUBITS == 32:
                    raise ValueError(
                        "N_QUBITS 32 not supported for default hyperparameters. Set "
                        "skip_loading_from_run_id=True and provide the prior_args"
                    )
                else:
                    raise ValueError("N_QUBITS must be 16 or 128")
                samples = lstm.generate(STATS_SAMPLES, random_seed=random_seed).cpu()
                mols = samples.numpy().astype(int)

                ligands = selfies.selfie_to_smiles(selfies.decode(mols))
                sr_rate = filter.get_validity_smiles(ligands)
                passed = list(set(filter.all_passed_smile))
                df = pd.DataFrame({"smiles": passed})
                df.to_csv(docking_path, index=False)
                print("Saving filtered ligands with shape", df.shape)
                del samples, mols, ligands, sr_rate, lstm, selfies
                torch.cuda.empty_cache()
            else:
                print("Skipping first iteration, file already exists")

        else:
            if os.path.exists(training_path):
                df = pd.read_csv(training_path)
                is_empty = df.shape[0] < 100
            else:
                is_empty = True
            if not os.path.exists(docking_path) or is_empty:
                codecarbon_lstm_model(
                    MODEL,
                    prior_bits=N_QUBITS,
                    prior_args=prior_args.copy(),
                    dataset_name=protein_to_dataset_name[MOLECULE],
                    filter_lists=filter_lists,
                    weight_lists=weight_lists,
                    mlflow_experiment_name_prefix=experiment_name,
                    mlflow_run_name_prefix=f"tartarus_lstm_{MOLECULE}_{i}"
                    f"{run_name_suffix}",
                    n_epochs=n_epochs,
                    prior_n_epochs=prior_n_epochs,
                    random_seed=random_seed,
                    save_quality_metrics=False,
                    gpu_id=GPU_ID,
                    compute_final_sr_rate=True,  # NEEDS TO BE TRUE!
                    compute_final_diversity_rate=True,
                    compute_final_novelity_rate=True,
                    cluster=cluster,
                    input_dataset_path=training_path,
                    dataset_path=docking_path,
                    save_filtered_ligands_in_dataset_path=True,
                    stats_samples=STATS_SAMPLES,
                    use_vectorized_cost_fn=use_vectorized_cost_fn,
                    parallel_cpus=N_CPUS,
                    use_linear_projection=use_linear_projection,
                    merge_prior_fc=merge_prior_fc,
                    use_kaiming_init=use_kaiming_init,
                    embedding_dim=embedding_dim,
                    sampling_temperature=sampling_temperature,
                    use_fusednaive_lstm=use_fusednaive_lstm,
                )
            else:
                print("Skipping training at iteration", i, ". File already exists")
        training_path = (
            tmp_base + f"training_{i+1}_{MODEL}_{MOLECULE}_60{run_name_suffix}.csv"
        )
        random_seed += 1
        df = pd.read_csv(docking_path)
        if "scores" not in df.columns:
            docking_results(
                protein_name=MOLECULE,
                tmp_file=docking_path,
                parallel_cpus=N_CPUS,
                docking_scores_path=f"{tmp_actual_base}docking_scores_{MOLECULE}.pkl",
            )
        else:
            print("Skipping docking at iteration", i, ". File already exists")
        if os.path.exists(training_path):
            df = pd.read_csv(training_path)
            is_empty = df.shape[0] < 100
        else:
            is_empty = True
        if not os.path.exists(training_path) or is_empty:
            hill_climbing(
                similarity_limit=similarity_limit,
                num_random_samples=20000,
                docking_filename=docking_path,
                training_filename=training_path,
                parallel_cpus=N_CPUS,
                top_molecules=60,
                training_dataset_size_limit=training_dataset_size_limit,
            )
        else:
            print("Skipping hill climbing at iteration", i, ". File already exists")
