import os

# Set cuda device 7 to be visible:
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import mlflow
import numpy as np
import torch
from orquestra.qml.api import convert_to_torch
from orquestra.qml.data_loaders import new_data_loader
from orquestra.qml.experimental.models.recurrent.lstm.th import NoisyLSTM
from orquestra.qml.models.rbm.jx import RBM
from orquestra.qml.trainers import AdversarialTrainer, SimpleTrainer

from orquestra.drug.discovery.encoding import Selfies
from orquestra.drug.discovery.validator import GeneralFilter, PainFilter, WehiMCFilter
from orquestra.drug.sdk import configure_mlflow
from orquestra.drug.utils import (
    ConditionFilters,
    FinalBatchSchedule,
    MLFlowCallback,
    ModifyTargetsCallback,
    RecordCostFn,
    save_pickle,
)

dataset_dict = {"alzheimer": "data/docking.csv"}

dataset_name = "alzheimer"

selfies = Selfies.from_smiles_csv(dataset_dict[dataset_name])


# selfies = Selfies.from_smiles_csv(
# #"data/1Mstoned_vsc_initial_dataset_insilico_chemistry42_filtered.csv"
# #"data/data.csv"
# #"data/initial_data_set_with_100k_hits.csv"
# "data/docking.csv"
# )

filter_lists = [GeneralFilter(), PainFilter(), WehiMCFilter()]
weight_lists = [5.0, 2.0, 1.0]
lstm_loss_key = "nll"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Will be using the following device", device)

prior_bits = 32
record_cost_fn = RecordCostFn(filter_lists, weight_lists, selfies)

# We deduce the sparsity for the RBM:

sparsity = (3 / 4 * prior_bits * prior_bits + prior_bits) / (prior_bits * prior_bits)
rbm = RBM(n_visible=prior_bits, n_hidden=prior_bits, random_seed=0, sparsity=sparsity)
print("Going for sparsity", sparsity)

vocab_size = selfies.n_tokens
max_seq_len = selfies.max_length
sos_token_index = 0
padding_token_index = selfies.pad_index
prior_batch_size = -1
n_prior_samples = 1024
prior_n_epochs = 10
n_epochs = 100
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
    prior=rbm,
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
mlflow_experiment_name = f"priorsize_{prior_bits}_lstm_{dataset_name}"
# for this to work: orq login -s https://prod-d.orquestra.io
configure_mlflow(mlflow_experiment_name)

with mlflow.start_run(run_name=f"jax_sparse_rbm_prior_lstm_batchsize_{batch_size}"):
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

    dataset = selfies.as_tensor()
    dummy_targets = convert_to_torch(np.ones((len(dataset), prior_bits)))
    dataloader = new_data_loader(
        dataset, targets=dummy_targets, batch_size=batch_size
    ).shuffle(seed=33)

    schedule = FinalBatchSchedule(final_batch=dataloader.n_batches - 1)
    callback = ModifyTargetsCallback(prior=rbm, seed=98735)

    trainer = AdversarialTrainer(prior_training_schedule=schedule)
    result = trainer.train(
        lstm,
        dataloader,
        n_epochs=n_epochs,
        callbacks=[callback, record_cost_fn, MLFlowCallback(lstm_loss_key)],
    )

    samples = lstm.generate(10, random_seed=23).cpu()

    mols = samples.cpu().numpy().astype(int)

    ligands = selfies.selfie_to_smiles(selfies.decode(mols))

    save_pickle([lstm, selfies, mols, samples, result, ligands], "result.pkl")

    filter = ConditionFilters(filter_lists=filter_lists, weight_lists=weight_lists)

    filter.apply_all(ligands)
