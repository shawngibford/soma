import os

# Set cuda device 7 to be visible:
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

import sys
from datetime import datetime

import mlflow
import numpy as np
import optax
import pandas as pd
import torch
from orquestra.qml.api import convert_to_numpy, convert_to_torch
from orquestra.qml.api.trainer import TrainCache, TrainerBehaviourControl
from orquestra.qml.data_loaders import new_data_loader
from orquestra.qml.experimental.models.dwave.qbag import QBAG
from orquestra.qml.experimental.models.recurrent.lstm.th import NoisyLSTM
from orquestra.qml.models.rbm.jx import RBM
from orquestra.qml.models.samplers.th import RandomChoiceSampler
from orquestra.qml.trainers import AdversarialTrainer, SimpleTrainer
from tartarus_docking import docking_results
from torch import nn

from orquestra.drug.discovery.docking.utils import process_molecule
from orquestra.drug.discovery.encoding import Selfies
from orquestra.drug.discovery.validator import (
    GeneralFilter,
    PainFilter,
    SybaFilter,
    WehiMCFilter,
)
from orquestra.drug.discovery.validator.filter_abstract import FilterAbstract
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


class AddXY(nn.Module):
    """Add two tensors. Tensor shapes must be broadcastable."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class MultiplyXY(nn.Module):
    """sumary_line

    Keyword arguments:
    argument -- description
    Return: return_description
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, y)


class TartarusFilters(FilterAbstract):
    def apply(self, smile: str):
        _, status = process_molecule(smile)
        if status == "PASS":
            return True
        return False


def generate_datetime_string():
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as YYYYMMDD_HHMMSS
    datetime_string = now.strftime("%Y%m%d_%H%M%S")

    return datetime_string


# Example usage:

MOLECULE = "6y2f"
N_CPUS = 200
tmp_actual_base = "/root/orquestra-drug-discovery/notebook/tmp/"
dataset_dict = {
    "covid": "/root/orquestra-drug-discovery/notebook/data/docking_5_qcbm_6y2f_60_lower_similarity.csv"
}

dataset_name = "covid"
# python nLSTM_multi_prior_size.py rbm 2048 3 1 multi
if len(sys.argv) > 2:

    prior_name = sys.argv[1]
    prior_bits = int(sys.argv[2])
    random_seed = int(sys.argv[3])
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]
    print(prior_name, prior_bits, random_seed, sys.argv[4])
    func_prior = str(sys.argv[5])
else:
    prior_name = "random"
    prior_bits = 16
    random_seed = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    func_prior = "add"


if func_prior == "add":
    func_prior_class = AddXY()
elif func_prior == "multi":
    func_prior_class = MultiplyXY()


print(f"prior_name = {prior_name} and prior_bits = {prior_bits}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("Using", torch.cuda.device_count(), "GPUs!")
# exit()
selfies = Selfies.from_smiles_csv(dataset_dict[dataset_name])
smiles_dataset_df = pd.read_csv(dataset_dict[dataset_name])
smiles_dataset = smiles_dataset_df.smiles.to_list()

# selfies = Selfies.from_smiles_csv(
# #"data/1Mstoned_vsc_initial_dataset_insilico_chemistry42_filtered.csv"
# #"data/data.csv"
# #"data/initial_data_set_with_100k_hits.csv"
# "data/docking.csv"
# )

# filter_lists=[GeneralFilter()]
weight_lists = [5.0]
filter_lists = [TartarusFilters()]

novelity = MoleculeNovelty(smiles_dataset)
filter = ConditionFilters(filter_lists=filter_lists, weight_lists=weight_lists)

lstm_loss_key = "nll"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Will be using the following device", device)


record_cost_fn = RecordCostFn(
    filter_lists, weight_lists, selfies, novelity, get_diversity
)
# edges = "all"
# sampling_mode = "parallel"
# qbag = QBAG(n_visible=prior_bits, token_secret_name="dwave-api-token",
#             config_name="prod-d", # research
#             edges=edges, sampling_mode=sampling_mode, n_shots=50_000, random_seed=0)

if prior_name == "rbm":
    prior_loss_keys = ("loss",)
    prior = RBM(
        n_visible=prior_bits,
        n_hidden=prior_bits,
        random_seed=random_seed,
        optimizer=optax.sgd(learning_rate=1e-6),
    )
elif prior_name == "random":
    prior_loss_keys = tuple()
    prior = RandomChoiceSampler(sampler_dimension=prior_bits, choices=(0, 1))
else:
    raise ValueError("prior name must be random or rbm or qabm")

vocab_size = selfies.n_tokens
max_seq_len = selfies.max_length
sos_token_index = 0
padding_token_index = selfies.pad_index
prior_batch_size = -1
n_prior_samples = 1024
prior_n_epochs = 40
n_epochs = 500
cost_fn_temperature = 1.0
# random_seed = 0
n_layers = 5
hidden_dim = 128
embedding_dim = prior_bits
sampling_temperature = 0.62
batch_size = 2048  # 4096

lstm = NoisyLSTM(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    sos_token_index=sos_token_index,
    padding_token_index=padding_token_index,
    prior_sample_dim=prior_bits,
    embedding_dim=prior_bits,
    prior=prior,
    prior_trainer=SimpleTrainer(),
    prior_batch_size=prior_batch_size,
    n_prior_samples=n_prior_samples,
    prior_n_epochs=prior_n_epochs,
    cost_fn=record_cost_fn,
    cost_fn_temperature=cost_fn_temperature,
    random_seed=random_seed,
    n_layers=n_layers,
    hidden_dim=hidden_dim,
    sampling_temperature=sampling_temperature,
    loss_key=lstm_loss_key,
    use_linear_projection=False,
    merge_prior_fc=func_prior_class,
    use_kaiming_init=False,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lstm.to_device(device)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    lstm._model = nn.DataParallel(lstm._model)
    batch_size = batch_size * torch.cuda.device_count()


print(device)
print("Using", torch.cuda.device_count(), "GPUs!")


mlflow_experiment_name = f"nLSTM_Tartarus_filter_{dataset_name}"
configure_mlflow(mlflow_experiment_name)
# calculate metrics

datetime_string = generate_datetime_string()
print(datetime_string)
docking_path = f"tmp/{prior_name}_{prior_bits}_nlstm_batchsize_{batch_size}_ligands_{datetime_string}.csv"
tmp_actual_base = "/root/orquestra-drug-discovery/notebook/tmp/"
with mlflow.start_run(
    run_name=f"{prior_name}_{prior_bits}_nlstm_batchsize_{batch_size}_seed_{random_seed}_prior_n_epochs_{prior_n_epochs}_fc_{func_prior}"
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
    mlflow.log_param("sampling_temperature", sampling_temperature)
    mlflow.log_param("cost_function_conditions", "[GeneralFilter()]")
    mlflow.log_param("cost_function_weights", "[5.0]")
    mlflow.log_param("use_linear_projection", "False")
    mlflow.log_param("merge_prior_fc", func_prior)
    # qbag_fields = ["_J0", "_phi0", "_J_logical", "_h_logical", "_parameters"]
    # def save_qbag_mlflow_model(model):
    #     # Save pickled model to temporary file:
    #     import pickle
    #     from tempfile import NamedTemporaryFile

    #     with NamedTemporaryFile(suffix=".pkl") as tmp:
    #         if isinstance(model, dict) and all(field in model for field in qbag_fields):
    #             pickle.dump(
    #                 {field: model[field] for field in qbag_fields},
    #                 tmp,
    #             )
    #         mlflow.log_artifact(tmp.name)

    dataset = selfies.as_tensor()
    dummy_targets = convert_to_torch(np.ones((len(dataset), prior_bits)))
    dataloader = new_data_loader(
        dataset, targets=dummy_targets, batch_size=batch_size
    ).shuffle(seed=33)

    schedule = FinalBatchSchedule(final_batch=dataloader.n_batches - 1)
    callback = ModifyTargetsCallback(prior=prior, seed=98735)

    trainer = AdversarialTrainer(prior_training_schedule=schedule)
    result = trainer.train(
        lstm,
        dataloader,
        n_epochs=n_epochs,
        callbacks=[
            callback,
            record_cost_fn,
            MLFlowCallback(lstm_loss_key, prior_loss_keys),
        ],
    )

    samples = lstm.generate(5000, random_seed=23).cpu()
    mols = samples.numpy().astype(int)

    ligands = selfies.selfie_to_smiles(selfies.decode(mols))
    # novelity_rate = novelity.get_novelity_smiles(ligands)
    sr_rate = filter.get_validity_smiles(ligands)
    # diversity_rate = get_diversity(ligands)

    # print(f"sr_rate:{sr_rate},diversity_rate:{diversity_rate},novelity_rate:{novelity_rate}")
    print(f"sr_rate:{sr_rate}")

    # save_qbag_mlflow_model(prior)
    # save_pickle([lstm._model,selfies,mols,samples,result,ligands,diversity_rate,sr_rate,novelity_rate], f"result_{datetime_string}.pkl")
    save_pickle(
        [
            lstm._model,
            selfies,
            mols,
            samples,
            result,
            ligands,
            sr_rate,
            prior.generate(5000),
        ],
        f"result_{datetime_string}.pkl",
    )

    mlflow.log_artifact(f"result_{datetime_string}.pkl")
    N_CPUS = 200
    df = pd.DataFrame({"smiles": ligands})
    df.to_csv(docking_path)
    docking_results(
        protein_name=MOLECULE,
        tmp_file=docking_path,
        parallel_cpus=N_CPUS,
        docking_scores_path=f"{tmp_actual_base}docking_scores_{MOLECULE}.pkl",
    )
