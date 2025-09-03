import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, List, Set

import cloudpickle
import mlflow
import numpy as np
import torch
import tqdm
from orquestra.qml.api import (
    Callback,
    GenerativeModel,
    Tensor,
    TrainCache,
    TrainerBehaviourControl,
    TrainingSchedule,
    convert_to_numpy,
    convert_to_torch,
)
from pathos.pools import ThreadPool

from orquestra.drug.discovery import docking
from orquestra.drug.discovery.validator import ConditionFilters

from .metrics import MoleculeNovelty


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


@dataclass
class CompoundsStatistics:
    unique_compounds: Set[str]  # generated compounds that are unique
    valid_compounds: Set[str]  # generated, unique compounds that are also valid
    unseen_compounds: Set[
        str
    ]  # generated, unique, valid compounds that are also not present in train data
    all_compounds: List[str]
    label_compounds: List[str]
    novelity_fraction: List[str]
    diversity_fraction: float
    filter_fraction: float
    unique_fraction: float
    # Diversity %
    # Fraction of molecules that pass the filter
    # Fraction of unique molecules

    @property
    def n_unique(self) -> int:
        return len(self.unique_compounds)

    @property
    def n_valid(self) -> int:
        return len(self.valid_compounds)

    @property
    def n_unseen(self) -> int:
        return len(self.unseen_compounds)

    @property
    def total_compounds(self) -> int:
        return len(self.all_compounds)


# save in file:
def save_pickle(obj, file_path):
    with open(file_path, "wb") as f:
        r = cloudpickle.dump(obj, f)
    return r


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        obj = cloudpickle.load(f)
    return obj


def log_mlflow_artifact(artifact, name):
    import tempfile

    with tempfile.NamedTemporaryFile(prefix=f"{name}_", suffix=".pickle") as tmp:
        cloudpickle.dump(artifact, tmp)
        tmp.seek(0)
        mlflow.log_artifact(tmp.name)


class SaveQBAGParamsToMLFlow(Callback):
    def __init__(self, qbag):
        self.qbag = qbag

    def on_epoch_end(
        self, epoch: int, cache: TrainCache, control: TrainerBehaviourControl
    ):
        import tempfile

        params = {
            "J_logical": self.qbag._J_logical,
            "flux_biases_logical": self.qbag._flux_biases_logical,
            "embedding": self.qbag._single_embedding,
            "epoch": epoch,
        }
        log_mlflow_artifact(params, f"qbag_params__epoch_{epoch}")


class ModifyTargetsCallback(Callback):
    """
    This is a callback that modifies the targets of the training data by replacing
    them with samples from a prior.
    """

    def __init__(self, prior: GenerativeModel, seed: int = 0):
        self.prior = prior
        self._seed = seed

    def on_train_begin(self, cache: TrainCache):
        cache.remove_reserved_key("_train_batch")
        cache._store_train_data_flag = True

    def on_train_step_begin(self, step: int, cache: TrainCache):
        latest_batch = cache.latest_train_batch
        assert latest_batch is not None
        num_samples = latest_batch.data.shape[0]
        prior_samples = convert_to_torch(
            self.prior.generate(num_samples, random_seed=self._seed)
        ).float()
        self._seed += 1
        latest_batch.targets = prior_samples


class MLFlowCallback(Callback):
    def __init__(
        self,
        lstm_loss_key: str = "loss",
        prior_loss_keys: Sequence[str] = ("loss",),
        save_quality_metrics: bool = True,
    ):
        self.lstm_loss_key = lstm_loss_key
        self.prior_loss_keys = prior_loss_keys
        self.save_quality_metrics = save_quality_metrics

    def on_train_step_end(self, step: int, cache: TrainCache):
        mlflow.log_metric(
            self.lstm_loss_key, cache["history"][self.lstm_loss_key][-1], step
        )

    def on_epoch_end(
        self, epoch: int, cache: TrainCache, control: TrainerBehaviourControl
    ):
        for metric in ["cost_fn", "median_cost_fn", "min_cost_fn", "max_cost_fn"]:
            mlflow.log_metric(metric, cache["history"][metric][-1], epoch)
        if self.save_quality_metrics:
            mlflow.log_metric(
                "valid_samples_percentage",
                cache["history"]["valid_samples_percentage"][-1],
                epoch,
            )
            mlflow.log_metric(
                "SR", cache["history"]["valid_samples_percentage"][-1], epoch
            )
            mlflow.log_metric(
                "diversity", cache["history"]["diversity_rate"][-1], epoch
            )
            mlflow.log_metric("novelity", cache["history"]["novelity_rate"][-1], epoch)
        if "prior_history" in cache.history.keys():
            for ploss_key in self.prior_loss_keys:
                for i, loss in enumerate(
                    cache.history["prior_history"][-1].history[ploss_key]
                ):
                    mlflow.log_metric(f"prior_{ploss_key}_{epoch}", loss, i)


class SamplingMLFlowCallback(Callback):
    """
    Callback to log sampling metrics and generated ligands to MLflow.
    """

    def __init__(
        self,
        model,
        n_samples: int,
        selfies,
        seed: int,
        eos_index: int,
        sos_index: int,
        loss_key: str = "loss",
    ):
        import math

        assert math.log2(
            n_samples
        ).is_integer(), "Number of samples must be a power of 2"
        self.loss_key = loss_key
        self.model = model
        self.n_samples = n_samples
        self.selfies = selfies
        self.seed = seed
        self.eos_index = eos_index
        self.sos_index = sos_index

    def on_train_step_end(self, step: int, cache: TrainCache):
        """
        Log loss metric at the end of each training step.

        Arguments:
            step (int): Current training step.
            cache (TrainCache): Training cache.
        """
        mlflow.log_metric(
            self.loss_key,
            cache["history"][self.loss_key][-1],
            step,
            synchronous=False,
        )

    def on_epoch_end(
        self, epoch: int, cache: TrainCache, control: TrainerBehaviourControl
    ):
        """
        Generate samples, process them, and log to MLflow at epoch end.

        Arguments:
            epoch (int): Current epoch number.
            cache (TrainCache): Training cache.
            control (TrainerBehaviourControl): control object
        """
        x = []
        for i in range(5):
            try:
                for j in range(2**i):
                    samples = self.model.generate(
                        self.n_samples // (2**i), self.seed + j
                    )
                    x.append(samples)
                break
            except Exception as e:
                print(f"Sample generation error: {e}")
                print("Retrying with fewer samples.")
                continue
        else:
            raise RuntimeError("Failed to generate samples after multiple retries.")

        if isinstance(x[0], torch.Tensor):
            x = torch.cat(x, dim=0).cpu().numpy()
        else:
            x = np.concatenate(x, axis=0)

        tokens = x.astype(int)
        with tempfile.NamedTemporaryFile(
            prefix=f"tokens_epoch{epoch}_", suffix=".npy"
        ) as f:
            np.save(f, tokens)
            mlflow.log_artifact(f.name)

        # Updating seed for next epoch
        self.seed += 1

        # We replace the sos_index and eos_index with the pad_index:
        tokens[tokens == self.sos_index] = self.selfies.pad_index
        tokens[tokens == self.eos_index] = self.selfies.pad_index

        # Decoding the tokens to SMILES strings
        ligands = self.selfies.selfie_to_smiles(self.selfies.decode(tokens))
        # Logging the ligands as a text artifact
        ligands_text = "\n".join(ligands)
        mlflow.log_text(ligands_text, f"ligands_epoch{epoch}.txt")


class FinalBatchSchedule(TrainingSchedule):
    def __init__(self, final_batch: int) -> None:
        """
        A custom training schedule that only trains the prior at the last batch of
        the main training loop.
        """
        super().__init__()
        self.final_batch = final_batch

    def is_enabled(
        self,
        *,
        epoch: int,
        step: int,
        batch_index: int,
    ) -> bool:
        if batch_index != self.final_batch:
            return False
        return True


class RecordVectorizedCostFn(Callback):
    _priority = 1  # This callback gets executed first.

    def __init__(
        self,
        filter_lists,
        weight_lists,
        selfies,
        parallel_cpus,
        protein,
    ):
        self._current_costs = []
        self.filter = ConditionFilters(
            filter_lists=filter_lists, weight_lists=weight_lists
        )
        self.selfies = selfies
        self.paralell_cpus = parallel_cpus
        self.protein = protein
        self.docking_scores_cache = {}

    def __call__(self, tokenstrings: Sequence):

        from more_itertools import chunked_even

        # First, convert the tokenstrings to smiles:
        tokenstrings = [
            convert_to_numpy(ts).astype(int).reshape(1, -1) for ts in tokenstrings
        ]
        selfies_strings = [
            self.selfies.selfie_to_smiles(self.selfies.decode(ts))[0]
            for ts in tokenstrings
        ]

        uncomputed_selfies_strings = []
        for selfies_string in selfies_strings:
            if selfies_string not in self.docking_scores_cache:
                uncomputed_selfies_strings.append(selfies_string)

        d, m = divmod(len(uncomputed_selfies_strings), self.paralell_cpus)
        if m != 0:
            d += 1
        chunks = list(chunked_even(uncomputed_selfies_strings, d))
        chunks_with_cpu_id = list(zip(chunks, range(self.paralell_cpus)))

        def chunk_smi_docking_with_fixed_protein(chunk):
            return chunk_smi_docking(chunk, self.protein)

        with ThreadPool(nodes=len(chunks_with_cpu_id)) as pool:
            results = pool.map(chunk_smi_docking_with_fixed_protein, chunks_with_cpu_id)

        for result in results:
            for smi, score in result:
                self.docking_scores_cache[smi] = score
        self._current_costs = [
            self.docking_scores_cache[selfie_string]
            for selfie_string in selfies_strings
        ]
        return self._current_costs

    def on_epoch_end(
        self, epoch: int, cache: TrainCache, control: TrainerBehaviourControl
    ):
        metrics = {
            "cost_fn": np.mean(self._current_costs).item(),
            "median_cost_fn": np.median(self._current_costs).item(),
            "min_cost_fn": np.min(self._current_costs).item(),
            "max_cost_fn": np.max(self._current_costs).item(),
        }
        cache.update_history(metrics)
        # Logging current costs with mlflow
        log_mlflow_artifact(self._current_costs, f"costs_epoch_{epoch}")
        self._current_costs = []


class RecordCostFn(Callback):
    _priority = 1  # This callback gets executed first.

    def __init__(
        self,
        filter_lists,
        weight_lists,
        selfies,
        novelity,
        diversity,
        save_quality_metrics: bool = True,
    ):
        self._current_costs = []
        self.filter = ConditionFilters(
            filter_lists=filter_lists, weight_lists=weight_lists
        )
        self.selfies = selfies
        self.novelity = novelity
        self.diversity = diversity
        self.save_quality_metrics = save_quality_metrics

    def __call__(self, tokenstring):
        tokenstring = convert_to_numpy(tokenstring).astype(int).reshape(1, -1)
        # Enforces the last two bits to be 1 because having zeros costs more.
        selfies_string = self.selfies.selfie_to_smiles(self.selfies.decode(tokenstring))
        cost = -self.filter.compute_reward(selfies_string[0])[1]
        self._current_costs.append(cost)
        return cost

    def on_epoch_end(
        self, epoch: int, cache: TrainCache, control: TrainerBehaviourControl
    ):
        metrics = {}
        if self.save_quality_metrics:
            # For other metrics:
            samples = self._model.generate(5000, random_seed=23).cpu()
            mols = samples.numpy().astype(int)

            ligands = self.selfies.selfie_to_smiles(self.selfies.decode(mols))
            novelity_rate = self.novelity.get_novelity_smiles(ligands)
            sr_rate = self.filter.get_validity_smiles(ligands)
            diversity_rate = self.diversity(ligands)

            # there is a bug here! we need to take a binary value here not cost value
            # invalid_strings = np.sum(np.asarray(self._current_costs) == 0)
            # total_strings = len(self._current_costs)
            # TODO: add other metrics to the cache

            metrics["valid_samples_percentage"] = sr_rate
            metrics["novelity_rate"] = novelity_rate
            metrics["diversity_rate"] = diversity_rate
        metrics["cost_fn"] = np.mean(self._current_costs).item()
        metrics["median_cost_fn"] = np.median(self._current_costs).item()
        metrics["min_cost_fn"] = np.min(self._current_costs).item()
        metrics["max_cost_fn"] = np.max(self._current_costs).item()
        cache.update_history(metrics)
        # Logging current costs with mlflow
        log_mlflow_artifact(self._current_costs, f"costs_epoch_{epoch}")
        self._current_costs = []


def compute_compound_stats(
    compounds: Tensor,
    diversity_fn: Callable,
    validity_fn: Callable[[List[str]], List[str]],
    novelity_fn: Callable,
    train_compounds: List[str],
) -> CompoundsStatistics:

    # truncate samples by removing anything that comes after the `pad_char`
    # generated_compounds = truncate_fn(generated_compounds)
    diversity_fraction = diversity_fn(compounds)

    unqiue_generated_compounds = set(compounds)

    # gives us only valid unique compounds
    filtered_set = validity_fn(compounds)
    unique_valid_compounds = set(filtered_set)

    # valid unique compounds that are also not present in the training data
    unique_train_compounds = set(train_compounds)
    unique_unseen_valid_compounds = unique_valid_compounds.difference(
        unique_train_compounds
    )
    # fraction of unique valid compounds that are unseen
    novelity_fraction = (
        100
        * np.sum([novelity_fn(x) for x in list(unique_valid_compounds)])
        / len(compounds)
    )
    unique_fraction = 100 * len(unqiue_generated_compounds) / len(compounds)
    filter_fraction = 100 * len(filtered_set) / len(compounds)

    stats = CompoundsStatistics(
        unqiue_generated_compounds,
        unique_valid_compounds,
        unique_unseen_valid_compounds,
        compounds,
        [1] * len(compounds),
        novelity_fraction,
        diversity_fraction,
        filter_fraction,
        unique_fraction,
    )

    return stats
