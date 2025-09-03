import os

# Set cuda device 7 to be visible:
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import torch
from orquestra.qml.api import convert_to_torch
from orquestra.qml.data_loaders import new_data_loader
from orquestra.qml.experimental.models.dwave.qbag import QBAG
from orquestra.qml.experimental.models.recurrent.lstm.th import NoisyLSTM
from orquestra.qml.trainers import AdversarialTrainer, SimpleTrainer

from orquestra.drug.discovery.encoding import Selfies
from orquestra.drug.discovery.validator import (
    GeneralFilter,
    PainFilter,
    SybaFilter,
    WehiMCFilter,
)
from orquestra.drug.metrics import MoleculeNovelty, get_diversity
from orquestra.drug.sdk import configure_mlflow
from orquestra.drug.utils import (
    ConditionFilters,
    FinalBatchSchedule,
    MLFlowCallback,
    ModifyTargetsCallback,
    RecordCostFn,
    save_pickle,
)


def generate_datetime_string():
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as YYYYMMDD_HHMMSS
    datetime_string = now.strftime("%Y%m%d_%H%M%S")

    return datetime_string


# Example usage:


dataset_dict = {"alzheimer": "/root/orquestra-drug-discovery/notebook/data/docking.csv"}

dataset_name = "alzheimer"

selfies = Selfies.from_smiles_csv(dataset_dict[dataset_name])
smiles_dataset_df = pd.read_csv(dataset_dict[dataset_name])
smiles_dataset = smiles_dataset_df.smiles.to_list()

# selfies = Selfies.from_smiles_csv(
# #"data/1Mstoned_vsc_initial_dataset_insilico_chemistry42_filtered.csv"
# #"data/data.csv"
# #"data/initial_data_set_with_100k_hits.csv"
# "data/docking.csv"
# )

filter_lists = [GeneralFilter(), PainFilter(), WehiMCFilter(), SybaFilter()]
weight_lists = [5.0, 3.0, 3.0, 5.0]
# filter_lists=[GeneralFilter(), PainFilter(),WehiMCFilter()]
# weight_lists=[5.0, 3.0, 3.0]
# filter_lists=[GeneralFilter(), PainFilter(),WehiMCFilter()]
# weight_lists=[5.0, 2.0, 1.0]

novelity = MoleculeNovelty(smiles_dataset)
filter = ConditionFilters(filter_lists=filter_lists, weight_lists=weight_lists)

lstm_loss_key = "nll"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Will be using the following device", device)

prior_bits = 16
record_cost_fn = RecordCostFn(
    filter_lists, weight_lists, selfies, novelity, get_diversity
)
edges = "all"
sampling_mode = "parallel"
qbag = QBAG(
    n_visible=prior_bits,
    token_secret_name="dwave-api-token",
    config_name="prod-d",
    edges=edges,
    sampling_mode=sampling_mode,
    n_shots=50_000,
    random_seed=0,
)

vocab_size = selfies.n_tokens
max_seq_len = selfies.max_length
sos_token_index = 0
padding_token_index = selfies.pad_index
prior_batch_size = -1
n_prior_samples = 1024
prior_n_epochs = 10
n_epochs = 200
cost_fn_temperature = 1.0
random_seed = 0
n_layers = 5
hidden_dim = 128
embedding_dim = 224
sampling_temperature = 0.62

lstm = NoisyLSTM(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    sos_token_index=sos_token_index,
    padding_token_index=padding_token_index,
    prior_sample_dim=prior_bits,
    prior=qbag,
    prior_trainer=SimpleTrainer(),
    prior_batch_size=prior_batch_size,
    n_prior_samples=n_prior_samples,
    prior_n_epochs=prior_n_epochs,
    cost_fn=record_cost_fn,
    cost_fn_temperature=cost_fn_temperature,
    random_seed=random_seed,
    n_layers=n_layers,
    hidden_dim=hidden_dim,
    embedding_dim=embedding_dim,
    sampling_temperature=sampling_temperature,
    loss_key=lstm_loss_key,
)
lstm.to_device(device)

batch_size = 4096
mlflow_experiment_name = f"priorsize_{prior_bits}_lstm_{dataset_name}_syba"
configure_mlflow(mlflow_experiment_name)
# calculate metrics

datetime_string = generate_datetime_string()
print(datetime_string)
with mlflow.start_run(
    run_name=f"qbag_{edges}_{sampling_mode}_prior_lstm_batchsize_{batch_size}"
):
    mlflow.log_param("prior_bits", prior_bits)
    mlflow.log_param("vocab_size", vocab_size)
    mlflow.log_param("max_seq_len", max_seq_len)
    mlflow.log_param("sos_token_index", sos_token_index)
    mlflow.log_param("padding_token_index", padding_token_index)
    mlflow.log_param("prior_batch_size", prior_batch_size)
    mlflow.log_param("n_prior_samples", n_prior_samples)
    mlflow.log_param("prior_n_epochs", prior_n_epochs)
    mlflow.log_param("n_epochs", n_epochs)
    mlflow.log_param("cost_fn_temperature", cost_fn_temperature)
    mlflow.log_param("random_seed", random_seed)
    mlflow.log_param("n_layers", n_layers)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("embedding_dim", embedding_dim)
    mlflow.log_param("sampling_temperature", sampling_temperature)
    mlflow.log_param(
        "cost_function_conditions",
        "[GeneralFilter(), PainFilter(),WehiMCFilter(),SybaFilter()]",
    )
    mlflow.log_param("cost_function_weights", "[5.0, 3.0, 3.0, 7.0]")
    qbag_fields = ["_J0", "_phi0", "_J_logical", "_h_logical", "_parameters"]

    def save_qbag_mlflow_model(model):
        # Save pickled model to temporary file:
        import pickle
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(suffix=".pkl") as tmp:
            if isinstance(model, dict) and all(field in model for field in qbag_fields):
                pickle.dump(
                    {field: model[field] for field in qbag_fields},
                    tmp,
                )
            mlflow.log_artifact(tmp.name)

    dataset = selfies.as_tensor()
    dummy_targets = convert_to_torch(np.ones((len(dataset), prior_bits)))
    dataloader = new_data_loader(
        dataset, targets=dummy_targets, batch_size=batch_size
    ).shuffle(seed=33)

    schedule = FinalBatchSchedule(final_batch=dataloader.n_batches - 1)
    callback = ModifyTargetsCallback(prior=qbag, seed=98735)

    trainer = AdversarialTrainer(prior_training_schedule=schedule)
    result = trainer.train(
        lstm,
        dataloader,
        n_epochs=n_epochs,
        callbacks=[callback, record_cost_fn, MLFlowCallback(lstm_loss_key)],
    )

    samples = lstm.generate(5000, random_seed=23).cpu()
    mols = samples.numpy().astype(int)

    ligands = selfies.selfie_to_smiles(selfies.decode(mols))
    novelity_rate = novelity.get_novelity_smiles(ligands)
    sr_rate = filter.get_validity_smiles(ligands)
    diversity_rate = get_diversity(ligands)

    print(
        f"sr_rate:{sr_rate},diversity_rate:{diversity_rate},novelity_rate:{novelity_rate}"
    )

    # TODO: Do not save the QBAG because it is unpickable. Use the function below to save its attributes.
    save_qbag_mlflow_model(qbag)
    save_pickle(
        [
            lstm._model,
            selfies,
            mols,
            samples,
            result,
            ligands,
            diversity_rate,
            sr_rate,
            novelity_rate,
        ],
        f"result_{datetime_string}.pkl",
    )
    mlflow.log_artifact(f"results_{datetime_string}.pkl")
    # Compute Metrics
