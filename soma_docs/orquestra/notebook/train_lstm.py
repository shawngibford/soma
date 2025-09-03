import os
import random
import tempfile
from datetime import datetime

import cloudpickle
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from orquestra.qml.api import convert_to_torch
from orquestra.qml.data_loaders import new_data_loader
from orquestra.qml.experimental.models.dwave.qbag import QBAG
from orquestra.qml.experimental.models.qcbm import FastSU4QCBM
from orquestra.qml.experimental.models.recurrent.lstm.th import NoisyLSTM
from orquestra.qml.models.rbm.jx import RBM
from orquestra.qml.trainers import AdversarialTrainer, SimpleTrainer

from orquestra.drug.discovery.encoding import Selfies
from orquestra.drug.discovery.validator import GeneralFilter  # SybaFilter,
from orquestra.drug.discovery.validator import PainFilter, WehiMCFilter
from orquestra.drug.metrics import MoleculeNovelty, get_diversity
from orquestra.drug.sdk import configure_mlflow
from orquestra.drug.utils import (
    ConditionFilters,
    FinalBatchSchedule,
    MLFlowCallback,
    ModifyTargetsCallback,
    RecordCostFn,
    RecordVectorizedCostFn,
    SamplingMLFlowCallback,
    SaveQBAGParamsToMLFlow,
    save_pickle,
)

dataset_dict = {
    "alzheimer": "/root/orquestra-drug-discovery/notebook/data/docking.csv",
    "covid": "/root/orquestra-drug-discovery/notebook/data/docking.csv",
}
protein_dict = {"alzheimer": "1syh", "covid": "6y2f"}


def generate_datetime_string():
    """Get the current date and time"""
    now = datetime.now()

    # Format the date and time as YYYYMMDD_HHMMSS
    datetime_string = now.strftime("%Y%m%d_%H%M%S")

    return datetime_string


def codecarbon_lstm_model(
    prior_model: str,
    prior_bits: int,
    prior_args: dict,
    dataset_name="alzheimer",
    filter_lists=[GeneralFilter(), PainFilter(), WehiMCFilter()],  # SybaFilter()],
    weight_lists=[5.0, 3.0, 3.0],  # 5.0],
    novelity_threshold=0.6,
    mlflow_experiment_name_prefix: str = "lstm",
    batch_size=4096,
    mlflow_run_name_prefix: str = "run",
    n_epochs=100,
    prior_n_epochs=10,
    random_seed=0,
    save_quality_metrics: bool = False,
    gpu_id: int = 0,
    compute_final_sr_rate: bool = True,
    compute_final_diversity_rate: bool = True,
    compute_final_novelity_rate: bool = True,
    cluster="lambda01",
    input_dataset_path=None,
    dataset_path=None,
    save_filtered_ligands_in_dataset_path=False,
    stats_samples=5000,
    use_vectorized_cost_fn=False,
    parallel_cpus=1,
    use_linear_projection=True,
    merge_prior_fc=None,
    use_kaiming_init=False,
    embedding_dim=224,
    sampling_temperature=None,
    use_fusednaive_lstm=False,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    datetime_string = generate_datetime_string()
    mlflow_experiment_name = (
        f"{mlflow_experiment_name_prefix}__prior_{prior_model}_"
        f"{prior_bits}__dataset_{dataset_name}"
    )
    configure_mlflow(mlflow_experiment_name)
    # Create a temporary file where codecarbon will save the emissions data:
    with tempfile.NamedTemporaryFile(prefix="emissions", suffix=".pickle") as tmp:
        mlflow.start_run(
            run_name=f"{mlflow_run_name_prefix}__lstm_batchsize_{batch_size}__"
            f"{datetime_string}"
        )
        # Track the emissions of the function "train_lstm_model":
        with EmissionsTracker(save_to_file=False, gpu_ids=[gpu_id]) as tracker:
            result = train_lstm_model(
                prior_model,
                prior_bits,
                prior_args,
                dataset_name,
                filter_lists,
                weight_lists,
                novelity_threshold,
                batch_size,
                n_epochs,
                prior_n_epochs,
                random_seed,
                save_quality_metrics,
                compute_final_sr_rate,
                compute_final_diversity_rate,
                compute_final_novelity_rate,
                cluster,
                input_dataset_path,
                dataset_path,
                save_filtered_ligands_in_dataset_path,
                stats_samples,
                use_vectorized_cost_fn,
                parallel_cpus,
                use_linear_projection,
                merge_prior_fc,
                use_kaiming_init,
                embedding_dim,
                sampling_temperature,
                use_naive_lstm=use_fusednaive_lstm,
            )
        cloudpickle.dump(tracker.final_emissions_data, tmp)
        tmp.seek(0)
        mlflow.log_artifact(tmp.name)
        mlflow.end_run()
        return result


def _load_dataset(dataset_path, novelity_threshold, filter_lists, weight_lists):
    selfies = Selfies.from_smiles_csv(dataset_path)
    smiles_dataset_df = pd.read_csv(dataset_path)
    smiles_dataset = smiles_dataset_df.smiles.to_list()
    novelity = MoleculeNovelty(smiles_dataset, threshold=novelity_threshold)
    filter = ConditionFilters(filter_lists=filter_lists, weight_lists=weight_lists)
    return selfies, smiles_dataset_df, smiles_dataset, novelity, filter


def train_lstm_model(
    prior_model: str,
    prior_bits: int,
    prior_args: dict,
    dataset_name="alzheimer",
    filter_lists=[GeneralFilter(), PainFilter(), WehiMCFilter()],  # SybaFilter()],
    weight_lists=[5.0, 3.0, 3.0],  # 5.0],
    novelity_threshold=0.6,
    batch_size=4096,
    n_epochs=100,
    prior_n_epochs=10,
    random_seed=0,
    save_quality_metrics: bool = False,
    compute_final_sr_rate: bool = True,
    compute_final_diversity_rate: bool = True,
    compute_final_novelity_rate: bool = True,
    cluster="lambda01",
    input_dataset_path=None,
    dataset_path=None,
    save_filtered_ligands_in_dataset_path=False,
    stats_samples=5000,
    use_vectorized_cost_fn=False,
    parallel_cpus=1,
    use_linear_projection=True,
    merge_prior_fc=None,
    use_kaiming_init=False,
    embedding_dim=224,
    sampling_temperature=None,
    use_naive_lstm=False,
):
    if save_filtered_ligands_in_dataset_path:
        assert dataset_path is not None
    mlflow.log_param("cluster", cluster)
    if input_dataset_path is not None:
        selfies, smiles_dataset_df, smiles_dataset, novelity, filter = _load_dataset(
            input_dataset_path, novelity_threshold, filter_lists, weight_lists
        )
        print("Loaded dataset from path", input_dataset_path)
    else:
        try:

            with open(
                f"{dataset_name} selfies_smiles_novelty_and_filter.pickle", "rb"
            ) as f:
                selfies, smiles_dataset, novelity, filter = cloudpickle.load(f)
                print("Loaded dataset from pickle")
        except FileNotFoundError:
            selfies = Selfies.from_smiles_csv(dataset_dict[dataset_name])
            smiles_dataset_df = pd.read_csv(dataset_dict[dataset_name])
            smiles_dataset = smiles_dataset_df.smiles.to_list()

            novelity = MoleculeNovelty(smiles_dataset, threshold=novelity_threshold)
            filter = ConditionFilters(
                filter_lists=filter_lists, weight_lists=weight_lists
            )
            with open(
                f"{dataset_name} selfies_smiles_novelty_and_filter.pickle", "wb"
            ) as f:
                cloudpickle.dump([selfies, smiles_dataset, novelity, filter], f)
    lstm_loss_key = "nll"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Will be using the following device", device)
    if use_vectorized_cost_fn:
        record_cost_fn = RecordVectorizedCostFn(
            filter_lists,
            weight_lists,
            selfies,
            parallel_cpus,
            protein_dict[dataset_name],
        )
    else:
        record_cost_fn = RecordCostFn(
            filter_lists,
            weight_lists,
            selfies,
            novelity,
            get_diversity,
            save_quality_metrics,
        )
    rng = random.Random(random_seed)
    model_callback = []
    if prior_model == "qbag":
        prior_args.update(
            {
                "edges": "all",
                "sampling_mode": "parallel",
                "n_shots": 50_000,
                "random_seed": rng.randint(0, 2**20),
                "n_visible": prior_bits,
            }
        )
        prior = QBAG(
            token_secret_name="dwave-api-token",
            config_name="research",
            solver_name="Advantage_system4.1",
            generation_max_retries=10,
            **prior_args,
        )
        prior_loss_keys = (
            "loss",
            "mean_magnetisation_difference",
            "mean_correlation_difference",
        )
        mlflow.log_params(prior_args)
        # We are using parallel, so we can access how many embeddings are being used
        mlflow.log_param("n_dwave_embeddings", len(prior._all_embeddings_list))
        model_callback.append(SaveQBAGParamsToMLFlow(prior))
    elif prior_model == "rbm":
        import optax

        sparsity = (3 / 4 * prior_bits * prior_bits + prior_bits) / (
            prior_bits * prior_bits
        )
        if "n_hidden" in prior_args.keys():
            n_hidden = prior_args["n_hidden"]
        else:
            n_hidden = prior_bits
        prior_args.update(
            {
                "n_visible": prior_bits,
                "n_hidden": n_hidden,
                "sparsity": sparsity,
                "random_seed": rng.randint(0, 2**20),
            }
        )
        mlflow.log_params(prior_args)
        optimizer = optax.adam(prior_args.pop("learning_rate"))
        prior = RBM(**prior_args, optimizer=optimizer)
        prior_loss_keys = ("loss",)
    elif prior_model == "qcbm":
        import optax

        params_qbag = prior_bits**2 / 4 + prior_bits
        # sparsity must be set so that the total number of parameters is params_qbag.
        total_qcbm_params = (prior_bits - 1) * prior_args["qcbm_n_layers"] * 15
        sparsity = 1 - params_qbag / total_qcbm_params
        prior_args.update({"n_qubits": prior_bits, "sparsity": sparsity})
        mlflow.log_params(prior_args)
        prior_args["n_layers"] = prior_args.pop("qcbm_n_layers")
        optimizer = optax.adam(prior_args.pop("learning_rate"))
        prior = FastSU4QCBM(**prior_args, optimizer=optimizer)
        prior_loss_keys = ("loss",)
    elif prior_model == "random":
        from orquestra.qml.models.samplers.th import RandomChoiceSampler

        prior_loss_keys = tuple()

        prior = RandomChoiceSampler(sampler_dimension=prior_bits, choices=(0, 1))
    elif prior_model == "zeros":
        from orquestra.qml.api import GenerativeModel

        class Zeros(GenerativeModel):
            _repr_fields: list[str] = ["sampler_dimension"]

            def __init__(self, sampler_dimension: int):
                self.sampler_dimension = sampler_dimension

            @property
            def sample_size(self):
                return (self.sampler_dimension,)

            def train_on_batch(self, batch):
                return {"loss": 0.0}

            def generate(self, n_samples: int, random_seed: int | None = None):
                return np.zeros((n_samples, self.sampler_dimension))

        prior_loss_keys = tuple()
        prior = Zeros(sampler_dimension=prior_bits)
    else:
        raise ValueError(f"Unknown prior model: {prior_model}")

    vocab_size = selfies.n_tokens
    max_seq_len = selfies.max_length
    sos_token_index = 0
    eos_index = selfies.eos_index
    padding_token_index = selfies.pad_index
    prior_batch_size = -1
    n_prior_samples = 1024
    cost_fn_temperature = 1.0
    n_layers = 5
    hidden_dim = 128
    sampling_temperature = (
        0.62 if sampling_temperature is None else sampling_temperature
    )

    lstm = NoisyLSTM(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        sos_token_index=sos_token_index,
        padding_token_index=padding_token_index,
        prior_sample_dim=prior_bits,
        prior=prior,
        prior_trainer=SimpleTrainer(),
        prior_batch_size=prior_batch_size,
        n_prior_samples=n_prior_samples,
        prior_n_epochs=prior_n_epochs,
        cost_fn=record_cost_fn,
        cost_fn_temperature=cost_fn_temperature,
        random_seed=rng.randint(0, 2**20),
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        sampling_temperature=sampling_temperature,
        loss_key=lstm_loss_key,
        is_cost_fn_vectorized=use_vectorized_cost_fn,
        use_linear_projection=use_linear_projection,
        merge_prior_fc=merge_prior_fc,
        use_kaiming_init=use_kaiming_init,
        use_naive_lstm=use_naive_lstm,
    )
    lstm.to_device(device)
    mlflow.log_param("vocab_size", vocab_size)
    mlflow.log_param("max_seq_len", max_seq_len)
    mlflow.log_param("sos_token_index", sos_token_index)
    mlflow.log_param("padding_token_index", padding_token_index)
    mlflow.log_param("prior_batch_size", prior_batch_size)
    mlflow.log_param("n_prior_samples", n_prior_samples)
    mlflow.log_param("prior_n_epochs", prior_n_epochs)
    mlflow.log_param("n_epochs", n_epochs)
    mlflow.log_param("cost_fn_temperature", cost_fn_temperature)
    mlflow.log_param("n_layers", n_layers)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("embedding_dim", embedding_dim)
    mlflow.log_param("sampling_temperature", sampling_temperature)
    mlflow.log_param(
        "cost_function_filters", ", ".join([f.__class__.__name__ for f in filter_lists])
    )
    mlflow.log_param("cost_function_weights", ", ".join([str(w) for w in weight_lists]))

    dataset = selfies.as_tensor()
    dummy_targets = convert_to_torch(np.ones((len(dataset), prior_bits)))
    dataloader = new_data_loader(
        dataset, targets=dummy_targets, batch_size=batch_size
    ).shuffle(seed=rng.randint(0, 2**20))

    schedule = FinalBatchSchedule(final_batch=dataloader.n_batches - 1)
    callback = ModifyTargetsCallback(prior=prior, seed=rng.randint(0, 2**20))

    sampling_mlflow_callback = SamplingMLFlowCallback(
        model=lstm,
        n_samples=512,
        selfies=selfies,
        seed=rng.randint(0, 2**20),
        eos_index=eos_index,
        sos_index=sos_token_index,
        loss_key=lstm_loss_key,
    )

    trainer = AdversarialTrainer(prior_training_schedule=schedule)
    result = trainer.train(
        lstm,
        dataloader,
        n_epochs=n_epochs,
        callbacks=[
            callback,
            record_cost_fn,
            MLFlowCallback(lstm_loss_key, prior_loss_keys, save_quality_metrics),
            sampling_mlflow_callback,
        ]
        + model_callback,
    )

    metrics = []
    final_costs = [0]
    return_dict = {"result": result}
    # TODO: compute success rate with process_molecule in docking.utils
    if (
        compute_final_sr_rate
        or compute_final_diversity_rate
        or compute_final_novelity_rate
    ):
        # After training, for generation, we want to increase the temperature to the
        # one that was found by Mohammad G. in his paper.
        lstm.sampling_temperature = 0.62
        samples = lstm.generate(stats_samples, random_seed=rng.randint(0, 2**20)).cpu()
        mols = samples.numpy().astype(int)

        ligands = selfies.selfie_to_smiles(selfies.decode(mols))
        if use_vectorized_cost_fn:
            final_costs = record_cost_fn(samples)
        else:
            final_costs = [record_cost_fn(sample) for sample in samples]
        return_dict["final_costs"] = final_costs
        mlflow.log_metric("final_cost", np.mean(final_costs))
        if compute_final_novelity_rate:
            from orquestra.drug.discovery.docking.utils import get_novelties

            novelties = get_novelties(
                ligands,
                novelity.reference_fingerprints,
                novelity.threshold,
                parallel_cpus,
            )
            novelty_rate = sum([1 for n in novelties if n]) / len(novelties)
            return_dict["novelity_rate"] = novelty_rate
            metrics.append(novelty_rate)
            print(f"novelty_rate:{novelty_rate}")
            mlflow.log_metric("final_novelity_rate", novelty_rate)
        if compute_final_sr_rate:
            sr_rate = filter.get_validity_smiles(ligands)
            return_dict["sr_rate"] = sr_rate
            metrics.append(sr_rate)
            print(f"sr_rate:{sr_rate}")
            mlflow.log_metric("final_sr_rate", sr_rate)
        if compute_final_diversity_rate:
            diversity_rate = get_diversity(ligands)
            return_dict["diversity_rate"] = diversity_rate
            metrics.append(diversity_rate)
            print(f"diversity_rate:{diversity_rate}")
            mlflow.log_metric("final_diversity_rate", diversity_rate)

    with tempfile.NamedTemporaryFile(prefix="result_as_dict_", suffix=".pickle") as tmp:
        save_pickle(return_dict, tmp.name)
        tmp.seek(0)
        mlflow.log_artifact(tmp.name)
    with tempfile.NamedTemporaryFile(prefix="model_", suffix=".pickle") as tmp:
        lstm.to_device(torch.device("cpu"))
        save_pickle(lstm, tmp.name)
        tmp.seek(0)
        mlflow.log_artifact(tmp.name)
    with tempfile.NamedTemporaryFile(
        prefix="smiles_and_filter", suffix=".pickle"
    ) as tmp:
        save_pickle({"selfes": selfies, "filter": filter}, tmp.name)
        tmp.seek(0)
        mlflow.log_artifact(tmp.name)
    if save_filtered_ligands_in_dataset_path:
        if hasattr(filter, "all_passed_smile"):
            passed = list(set(filter.all_passed_smile))
            df = pd.DataFrame({"smiles": passed})
            df.to_csv(dataset_path, index=False)
            print("Saving filtered ligands with shape", df.shape)
        else:
            print("No filtered ligands to save, saving the original dataset")
            smiles_dataset_df.to_csv(dataset_path, index=False)
    return np.mean(final_costs), *metrics


def objective(
    trial: optuna.Trial,
    suggest_fn,
    prior_model,
    prior_bits,
    gpu_id,
    compute_final_sr_rate=True,
    compute_final_diversity_rate=True,
    compute_final_novelity_rate=True,
    epochs=50,
    prior_n_epochs=10,
    use_syba=False,
    cluster="lambda01",
):
    if use_syba:
        from orquestra.drug.discovery.validator import SybaFilter

        filter_lists = [GeneralFilter(), PainFilter(), WehiMCFilter(), SybaFilter()]
        weight_lists = [5.0, 3.0, 3.0, 5.0]
        experiment_name = "syba_lstm"
    else:
        filter_lists = [GeneralFilter(), PainFilter(), WehiMCFilter()]
        weight_lists = [5.0, 3.0, 3.0]
        experiment_name = "lstm"
    suggestions = suggest_fn(trial)
    result = codecarbon_lstm_model(
        prior_model,
        prior_bits,
        suggestions,
        dataset_name="alzheimer",
        filter_lists=filter_lists,
        weight_lists=weight_lists,
        novelity_threshold=0.6,
        mlflow_experiment_name_prefix=experiment_name,
        batch_size=4096,
        mlflow_run_name_prefix=f"trial_{trial.number}",
        n_epochs=epochs,
        prior_n_epochs=prior_n_epochs,
        random_seed=0,
        save_quality_metrics=False,
        gpu_id=gpu_id,
        compute_final_sr_rate=compute_final_sr_rate,
        compute_final_diversity_rate=compute_final_diversity_rate,
        compute_final_novelity_rate=compute_final_novelity_rate,
        cluster=cluster,
    )
    return result


if __name__ == "__main__":
    codecarbon_lstm_model(
        prior_model="qbag",
        prior_bits=16,
        prior_args={"phi": 0.00001, "J": 0.05},
        dataset_name="alzheimer",
        filter_lists=[GeneralFilter(), PainFilter(), WehiMCFilter()],
        weight_lists=[5.0, 3.0, 3.0],
        novelity_threshold=0.6,
        mlflow_experiment_name_prefix="lstm",
        batch_size=4096,
        mlflow_run_name_prefix="run",
        n_epochs=1000,
        prior_n_epochs=5,
        random_seed=0,
        save_quality_metrics=False,
        gpu_id=1,
        compute_final_sr_rate=True,
        compute_final_diversity_rate=True,
        compute_final_novelity_rate=True,
    )
