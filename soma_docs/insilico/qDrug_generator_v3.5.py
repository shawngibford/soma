from typing import Dict, Optional, List, Any, Sequence, Callable, Set
from inspect import signature
from dataclasses import dataclass
import os, json, random, string, math, sys
from datetime import datetime
from pathlib import Path
from functools import partial
import torch
from torch import nn
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from rdkit import Chem
from rdkit.Chem import Draw
from orquestra.qml.api import (
    TorchGenerativeModel,
    GenerativeModel,
    Callback,
    TrainCache,
    convert_to_numpy,
    GenerativeModel,
)
from orquestra.qml.optimizers.th import AdamConfig
from orquestra.qml.models.rbm.th import RBM, TrainingParameters as RBMParams
from orquestra.qml.models.samplers.th import RandomChoiceSampler
from orquestra.qml.data_loaders import new_data_loader
from orquestra.qml.trainers import SimpleTrainer

from utils import (
    SmilesEncoding,
    SelfiesEncoding,
    generate_bulk_samples,
    DisplaySmilesCallback,
    truncate_smiles,
    Experiment,
    LegacyExperiment,
    lipinski_filter,
    lipinski_hard_filter,
    compute_compound_stats,
)
from utils.lipinski_utils import (
    compute_qed,
    compute_lipinski,
    compute_logp,
    draw_compounds,
)
from models.recurrent import NoisyLSTMv3
from utils.docking import compute_array_value
from models.priors.qcbm import QCBMSamplingFunction_v2, QCBMSamplingFunction_v3
from utils.data import compund_to_csv
import optuna
from optuna.trial import TrialState

# nicer plots
seaborn.set()

# allows us to ignore warnings
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
# QCBM imports
from orquestra.qml.models.qcbm.layer_builders import LineEntanglingLayerBuilder
from orquestra.qml.models.qcbm.ansatze.alternating_entangling_layers import (
    EntanglingLayerAnsatz,
)
from orquestra.qml.models.qcbm import WavefunctionQCBM
from orquestra.integrations.qulacs.simulator import QulacsSimulator
from orquestra.opt.optimizers.scipy_optimizer import ScipyOptimizer

# multi bases
from orquestra.qml.models.qcbm import MultiBasisWavefunctionQCBM
from orquestra.quantum.circuits import X, create_layer_of_gates
import time
from syba.syba import SybaClassifier
from utils.filter import (
    apply_filters,
    filter_phosphorus,
    substructure_violations,
    maximum_ring_size,
    # lipinski_filter,
    get_diversity,
    passes_wehi_mcf,
    pains_filt,
    legacy_apply_filters,
)
import dill
import cloudpickle


diversity_fn = get_diversity
start_time = time.time()
syba = SybaClassifier()
syba.fitDefaultScore()
print("Syba fitting time: ", time.time() - start_time)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 512
NEPOCH = 20
BATCHSIZE_GENERETATION = 100000
ACTIVE_FILTER = False
DISABLE_PROGRESS_BAR_PRIOR = False


def wma(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Returns Weighted Moving Average.

    Args:
        arr (np.ndarray): data array.
        window_size (int): window_size for computing average.
    """
    weights = np.arange(window_size)
    return np.convolve(arr, weights, "valid") / np.sum(weights)


def compute_stats_encode(data):
    new_data = []
    for data_ in data:
        try:
            reward_1 = compute_array_value(data_)
            lip = compute_lipinski(data_, mol_weight_ref=600)
            reward = np.append(lip[2], reward_1)
            new_data.append(reward)

        except:
            reward = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            new_data.append(reward)

    return torch.Tensor(new_data).sum(dim=1)


def filter_legacy(smiles_compound, max_mol_weight=800):
    pass_all = []
    for smile_ in smiles_compound:
        try:
            if apply_filters(smile_, max_mol_weight=max_mol_weight):
                pass_all.append(smiles_compound)
        except:
            pass
    return pass_all


def reward_fc_legacy(data):
    new_data = []
    for data_ in data:
        try:
            if validity_fn(data_):
                reward_1 = compute_array_value(data_)
                lip = compute_lipinski(data_, mol_weight_ref=600)
                reward = np.append(lip[2], reward_1)
                new_data.append(reward)
            else:
                reward = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                new_data.append(reward)
        except:
            reward = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            new_data.append(reward)

    return torch.Tensor(new_data).sum(dim=1)


def combine_filter(smiles_compound, max_mol_weight=800, filter_fc=apply_filters):
    pass_all = []
    i = 0

    with tqdm.tqdm(total=len(smiles_compound)) as pbar:
        for smile_ in smiles_compound:
            pbar.set_description(
                f"Filtered {i} / {len(smiles_compound)}. passed={len(pass_all)}"
            )
            try:
                if (
                    filter_fc(smile_, max_mol_weight)
                    and smile_ not in pass_all
                    and (syba.predict(smile_) > 0)
                    and passes_wehi_mcf(smile_)
                    and (len(pains_filt(Chem.MolFromSmiles(smile_))) == 0)
                ):

                    pass_all.append(smile_)
            except:
                pass
            i += 1
            pbar.update()
    return pass_all


def reward_fc(smiles_ls, max_mol_weight=800, filter_fc=apply_filters):
    rewards = []
    for smiles_compound in smiles_ls:
        #: TODO: add wieghts for filter
        try:
            reward = 0
            if smiles_compound not in smiles_ls:
                reward += 5
                if filter_fc(smiles_compound, max_mol_weight=max_mol_weight):
                    reward += 5
                    if passes_wehi_mcf(smiles_compound):
                        reward += 5
                        if len(pains_filt(Chem.MolFromSmiles(smiles_compound))) == 0:
                            reward += 5
                            if syba.predict(smiles_compound) > 0:
                                reward += 10

            rewards.append(reward)
        except:
            rewards.append(0)

    return torch.Tensor(rewards)


# save in file:
def save_obj(obj, file_path):
    with open(file_path, "wb") as f:
        r = cloudpickle.dump(obj, f)
    return r


def load_obj(file_path):
    with open(file_path, "rb") as f:
        obj = cloudpickle.load(f)
    return obj


# inputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available")
if len(sys.argv) > 1:
    lstm_n_epochs = int(sys.argv[1])
    prior_n_epochs = int(sys.argv[2])
    n_compound_generation = int(sys.argv[3])
    n_generation_steps = int(sys.argv[4])
    prior_model = str(sys.argv[5])
    filter_constraint = str(sys.argv[6])
    n_layes_lstm = int(sys.argv[7])
    embedding_dim = int(sys.argv[8])
    hidden_dim = int(sys.argv[9])
    prior_size = int(sys.argv[10])
    n_qcbm_layers = int(sys.argv[11])
    data_set = int(sys.argv[12])
    parallelize = bool(int(sys.argv[13]))
else:
    lstm_n_epochs = int(input("Enter lstm_n_epochs!\n"))
    prior_n_epochs = int(input("Enter prior_n_epochs!\n"))
    n_compound_generation = int(input("Enter n_compound_generation!\n"))
    n_generation_steps = int(input("Enter n_generation_steps!\n"))
    prior_model = str(input("Enter prior_model(QCBM/mQCBM/RBM/random)!\n"))
    print("******** Please enter model configurations ********\n")
    filter_constraint = str(input("Enter filter_constraint(hard|soft)!\n"))
    n_layes_lstm = int(input("Enter n_layes_lstm(1->5))!\n"))
    embedding_dim = int(input("Enter embedding_dim(256))!\n"))
    hidden_dim = int(input("Enter hidden_dim(128))!\n"))
    prior_size = int(input("Enter prior_sample_size (10> and <20)default=10!\n"))
    n_qcbm_layers = int(input("Enter QCBM layer (2~10) default=5)!\n"))
    data_set = int(input("Enter dataset code (0 =700, 1=11K, 2=1M)!\n"))

print("parallelize!!",parallelize)
#### inputs ####
n_test_samples = 20000
max_mol_weight = 800
prior_sample_size = prior_size
prior_hidden_layer = 10
lstm_n_batch_size = BATCHSIZE
# path_to_dataset = "data/KRAS_G12D/KRAS_G12D_inhibitors_update2023.csv"
if data_set == 0:
    # 700
    path_to_dataset = "data/KRAS_G12D/KRAS_G12D_inhibitors_update2023.csv"
elif data_set == 1:
    # 12k
    path_to_dataset = "data/KRAS_G12D/initial_dataset.csv"
elif data_set == 2:
    # 1M
    path_to_dataset = "data/KRAS_G12D/initial_data_with_chemistry42_syba_merged_v2.csv"
elif data_set == 3:
    # top(100K)
    path_to_dataset = "data/KRAS_G12D/initial_data_set_with_100k_hits.csv"
elif data_set == 4:
    # 1M+12K+top(100K)+initial dataset
    path_to_dataset = (
        "data/merged_dataset/1Mstoned_vsc_initial_dataset_insilico_chemistry42_filtered.csv"
    )

path_to_model_weights = None


if filter_constraint == "hard":
    filter_fc = partial(combine_filter, max_mol_weight=max_mol_weight)
    rew_fc = reward_fc
else:
    filter_fc = partial(
        combine_filter,
        max_mol_weight=max_mol_weight,
        filter_fc=legacy_apply_filters,
    )
    rew_fc = partial(
        reward_fc, max_mol_weight=max_mol_weight, filter_fc=legacy_apply_filters
    )

run_date_time = datetime.today().strftime("%Y_%d_%mT%H_%M_%S.%f")
experiment = LegacyExperiment(run_id=f"noisy-lstm-v3-{run_date_time}")
print(f"Experiment ID: {experiment.run_id}")

if data_set == 2:
    object_loaded = load_obj("data/initial_data.pkl")
    selfies = object_loaded[1]
else:
    selfies = SelfiesEncoding(path_to_dataset, dataset_identifier="insilico_KRAS")
print(f"Using file: {selfies._filepath}.")
print(f"Dataset identifier: {selfies.dataset_identifier}")

optimizer = ScipyOptimizer(method="Powell", options={"maxiter": 1})
if prior_model == "QCBM":
    # QCBM
    entangling_layer_builder = LineEntanglingLayerBuilder(n_qubits=prior_sample_size)
    qcbm_ansatz = EntanglingLayerAnsatz(
        n_qubits=prior_sample_size,
        n_layers=n_qcbm_layers,
        entangling_layer_builder=entangling_layer_builder,
    )
    prior = WavefunctionQCBM(
        ansatz=qcbm_ansatz,
        optimizer=optimizer,
        backend=QulacsSimulator(),
        choices=(0, 1),
        use_efficient_training=False,
    )
elif prior_model == "mQCBM":
    # multi bases QCBM
    entangling_layer_builder = LineEntanglingLayerBuilder(
        n_qubits=prior_sample_size // 2
    )
    multiqcbm_ansatz = EntanglingLayerAnsatz(
        n_qubits=prior_sample_size // 2,
        n_layers=n_qcbm_layers,
        entangling_layer_builder=entangling_layer_builder,
    )
    # We create a circuit that rotates the basis of the qubits at the end of the circuit
    rotate_basis_circuit = create_layer_of_gates(
        number_of_qubits=prior_sample_size // 2, gate_factory=X
    )
    prior = MultiBasisWavefunctionQCBM(
        ansatz=multiqcbm_ansatz,
        optimizer=optimizer,
        backend=QulacsSimulator(),
        choices=(0, 1),
        use_efficient_training=False,
        train_basis=False,
        basis_rotations=rotate_basis_circuit,
    )
elif prior_model == "mrQCBM":
    # multi bases QCBM
    entangling_layer_builder = LineEntanglingLayerBuilder(
        n_qubits=prior_sample_size // 2
    )
    multiqcbm_ansatz = EntanglingLayerAnsatz(
        n_qubits=prior_sample_size // 2,
        n_layers=n_qcbm_layers,
        entangling_layer_builder=entangling_layer_builder,
    )
    # We create a circuit that rotates the basis of the qubits at the end of the circuit
    rotate_basis_circuit = create_layer_of_gates(
        number_of_qubits=prior_sample_size // 2, gate_factory=X
    )
    prior = MultiBasisWavefunctionQCBM(
        ansatz=multiqcbm_ansatz,
        optimizer=optimizer,
        backend=QulacsSimulator(),
        choices=(0, 1),
        use_efficient_training=False,
        train_basis=False,
        basis_rotations=rotate_basis_circuit,
    )
elif prior_model == "RBM":
    prior = RBM(
        n_visible_units=prior_sample_size,
        n_hidden_units=prior_hidden_layer,
        training_parameters=RBMParams(),
    )
else:
    prior = RandomChoiceSampler(prior_sample_size, [0.0, 1.0])

print(f"Prior identifier: {prior.__str__()}")

# TODO: optuna for
# hidden_dim=8,  # next: 64
# embedding_dim=256
# hidden_dim=128,  # next: 64 # best resulst is with hidden_dim 128
model = NoisyLSTMv3(
    vocab_size=selfies.num_emd,
    seq_len=selfies.max_length,
    sos_token_index=selfies.start_char_index,
    prior_sample_dim=prior_sample_size,
    padding_token_index=selfies.pad_char_index,
    hidden_dim=hidden_dim,
    embedding_dim=embedding_dim,
    n_layers=n_layes_lstm,
)


if parallelize==1:
    print("parallelize!!",parallelize)
    model._model = torch.nn.DataParallel(model._model)

# experiment.model_configurations.append(model.config.as_dict())


if path_to_model_weights is not None:
    print(f"Loading model weights from {path_to_model_weights}")
    model.load_weights(path_to_model_weights)

training_parameters = {
    "n_epochs": lstm_n_epochs,
    "batch_size": lstm_n_batch_size,
}

if torch.cuda.is_available():
    model.to_device(DEVICE)


n_epochs = training_parameters["n_epochs"]

# encoded_samples_th = torch.tensor(selfies.encoded_samples)
# data = encoded_samples_th.float()
# save_obj([data,selfies],"data/initial_data.pkl")
if data_set == 2:
    data = object_loaded[0]
else:
    encoded_samples_th = torch.tensor(selfies.encoded_samples)
    data = encoded_samples_th.float()
decoder_fn = selfies.decode_fn
truncate_fn = truncate_smiles
validity_fn = filter_fc
train_compounds = selfies.train_samples


training_parameters.update(
    dict(
        n_test_samples=n_test_samples,
        decoder_fn_signature=str(signature(decoder_fn)),
        truncate_fn_signature=str(signature(truncate_fn)),
        validity_fn_signature=str(signature(validity_fn)),
    )
)

epoch_plot_dir = Path("experiment_results") / "epoch_plots" / experiment.run_id
epoch_plot_dir = epoch_plot_dir.resolve()

if epoch_plot_dir.exists() is False:
    os.makedirs(str(epoch_plot_dir))


dataloader = new_data_loader(
    data=data, batch_size=training_parameters["batch_size"],drop_last = True
).shuffle(12345).truncate(0.01)
train_cache = TrainCache()

if prior_model == "QCBM" or prior_model == "mQCBM" or prior_model == "mrQCBM":
    batch_size = -1
else:
    batch_size = training_parameters["batch_size"]

# start training
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
generated_compunds = {}
live_model_loss = []
for epoch in range(1, n_epochs + 1):
    with tqdm.tqdm(total=dataloader.n_batches) as pbar:
        pbar.set_description(f"Epoch {epoch} / {n_epochs}.")
        concat_prior_samples = []
        for batch_idx, batch in enumerate(dataloader):
            prior_samples = torch.tensor(prior.generate(batch.batch_size)).float()
            batch.targets = prior_samples
            batch_result = model.train_on_batch(batch)
            train_cache.update_history(batch_result)
            concat_prior_samples = concat_prior_samples + prior_samples.tolist()

            pbar.set_postfix(dict(Loss=batch_result["loss"]))
            pbar.update()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
        th_prior_samples = torch.tensor(concat_prior_samples)
        # put model in evaluation mode such that layers like Dropout, Batchnorm don't affect results
        model.set_eval_state()

        # prior training
        if epoch == 1:
            prior_samples_current = torch.tensor(prior.generate(n_test_samples)).float()
            encoded_compounds = model.generate(prior_samples_current)

            compound_stats = compute_compound_stats(
                encoded_compounds,
                decoder_fn,
                diversity_fn,
                validity_fn,
                train_compounds,
            )

        else:
            datanew = rew_fc(list(compound_stats.all_compounds)).cpu()
            soft = torch.nn.Softmax(dim=0)
            probs = soft(datanew)
            prior_train_data = new_data_loader(
                data=prior_samples_current,
                probs=probs,
                batch_size=batch_size,
            )

            # TODO track prior training cache
            if prior_model != "mrQCBM":
                prior_train_cache = SimpleTrainer().train(
                    prior,
                    prior_train_data,
                    n_epochs=prior_n_epochs,
                    disable_progress_bar=DISABLE_PROGRESS_BAR_PRIOR,
                )

            # generate compounds and then decode them such that we are working with sequences of str
            prior_samples_current = torch.tensor(prior.generate(n_test_samples)).float()
            encoded_compounds = model.generate(prior_samples_current)

            compound_stats = compute_compound_stats(
                encoded_compounds,
                decoder_fn,
                diversity_fn,
                validity_fn,
                train_compounds,
            )

        # train rbm or ...
        # new_data_loader(data=data,probs=stats, batch_size=training_parameters['batch_size'], shuffle=True)
        pbar.set_postfix(
            dict(
                Loss=batch_result["loss"],
                NumUniqueGenerated=compound_stats.n_unique,
                NumValidGenerated=compound_stats.n_valid,
                NumUnseenGenerated=compound_stats.n_unseen,
                unique_fraction=compound_stats.unique_fraction,
                filter_fraction=compound_stats.filter_fraction,
                diversity_fraction=compound_stats.diversity_fraction,
            )
        )  # type: ignore

        # update train result so we have a history of the samples
        train_cache[str(epoch)] = dict(
            samples={
                "unique": list(compound_stats.unique_compounds),
                "valid": list(compound_stats.valid_compounds),
                "unseen": list(compound_stats.unseen_compounds),
                "unique_fraction": compound_stats.unique_fraction,
                "filter_fraction": compound_stats.filter_fraction,
                "diversity_fraction": compound_stats.diversity_fraction,
            }
        )
        generated_compunds[str(epoch)] = dict(
            samples={
                "all": list(compound_stats.all_compounds),
                "unique": list(compound_stats.unique_compounds),
                "valid": list(compound_stats.valid_compounds),
                "unseen": list(compound_stats.unseen_compounds),
                "prior": list(prior_samples_current.tolist()),
                "unique_fraction": compound_stats.unique_fraction,
                "filter_fraction": compound_stats.filter_fraction,
                "diversity_fraction": compound_stats.diversity_fraction,
            }
        )

        # return model to train state
        model.set_train_state()

        # display randomly selected smiles
        rng = np.random.default_rng()

        try:
            selected_smiles = rng.choice(
                list(compound_stats.unseen_compounds), 20, replace=False
            )
            mols = [Chem.MolFromSmiles(smile_) for smile_ in selected_smiles]
            img = Draw.MolsToGridImage(mols, molsPerRow=20, returnPNG=False)
            img.save(f"{epoch_plot_dir}/epoch_{epoch}.png")

        except Exception as e:
            print(f"Unable to draw molecules: {e}")
        live_model_loss.append(np.mean(train_cache.history["loss"]))

        try:
            plt.figure()
            figure_path = epoch_plot_dir / f"prior_cost_{epoch}.png"
            plt.plot(prior_train_cache["history"]["opt_value"])
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig(figure_path, bbox_inches="tight", format="png")
        except Exception as e:
            print(f"Unable to draw prior loss fc in epoch {epoch}: {e}")

        try:
            plt.figure()
            figure_path = epoch_plot_dir / f"model_losses.png"
            plt.scatter(range(0, len(live_model_loss)), live_model_loss)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig(figure_path, bbox_inches="tight", format="png")
        except Exception as e:
            print(f"Unable to draw model loss fn in epoch {epoch}: {e}")
        try:
            data_ = {
                "prior_samples": prior_samples_current,
                "mode_samples": encoded_compounds,
                "selfies": selfies,
                "prior": prior,
                "model": model,
            }
            file_name = f"mode_prior_{epoch}"
            save_obj(data_, f"{epoch_plot_dir}/{file_name}.pkl")
        except Exception as e:
            print(f"Unable to save model and prior in epoch {epoch}: {e}")

# export analysis


unique_generated_samples = compound_stats.unique_compounds
valid_generated_samples = compound_stats.valid_compounds
unseen_generated_samples = compound_stats.unseen_compounds
unique_train_compounds = set(train_compounds)

print(f"Filter is {filter_constraint}, Prior model is: {prior_model}")
print(f"Number of UNIQUE compounds in TRAINING set is {len(unique_train_compounds)}.")
print(
    f"Number of UNIQUE compounds in GENERATED set is {len(unique_generated_samples)}."
)
print(
    f"Number of UNIQUE VALID samples in GENERATED set is {len(valid_generated_samples)}."
)
print(
    f"Number of UNIQUE VALID UNSEEN samples in GENERATED set is: {len(unseen_generated_samples)}."
)
print(
    f"Number of diversity fraction samples in GENERATED set is: {compound_stats.diversity_fraction}."
)
print(
    f"Number of filter fraction samples in GENERATED set is: {compound_stats.filter_fraction}."
)
print(
    f"Number of UNIQUE fraction samples in GENERATED set is: {compound_stats.unique_fraction}."
)

results_to_save = {
    "dataset_path": path_to_dataset,
    "model_name": model.__class__.__name__,
    "model_config": model.config.__dict__,
    "training_parameters": training_parameters,
    "generated_samples": {
        "all": list(unique_generated_samples),
        "unseen": list(unseen_generated_samples),
    },
    "n_generated_samples": len(unique_generated_samples),
    "n_training_samples": len(train_compounds),
    "n_unseen_generated_samples": len(unseen_generated_samples),
    "histories": generated_compunds,
    "training_cache": train_cache.history,
}

experiment.update_results(results_to_save)

results_filename = f"{model.model_identifier}-{selfies.dataset_identifier}-{experiment.run_id}-{experiment.date}.json"
experiment.save_experiment(results_filename)


saved_models_dir = Path(experiment._experiments_directory) / "saved_models"

if saved_models_dir.exists() is False:
    saved_models_dir.mkdir()


# Trained Noisy-LSTM
weights_path = (
    saved_models_dir
    / f"{model.model_identifier}-{experiment.run_id}-{experiment.date}.pt"
)

model.save_weights(str(weights_path))

model_saved = (
    saved_models_dir
    / f"{model.model_identifier}-{experiment.run_id}-{experiment.date}.pkl"
)
data = {"selfies": selfies, "prior": prior, "model": model}
save_obj(data, f"{model_saved}")
# Trained prior
# weights_rbm = saved_models_dir / f"{prior_model}-{experiment.run_id}-{experiment.date}.pt"

# prior_model.save_weights(weights_rbm)


training_cache = train_cache
n_test_samples = experiment.results["training_parameters"]["n_test_samples"]
model_identifier = model.model_identifier
model_name = experiment.results["model_name"]
hidden_state_fn_str = prior.__str__()

figures_dir = Path(experiment._experiments_directory) / "figures"

if figures_dir.exists() is False:
    figures_dir.mkdir()

figure_path = (
    figures_dir
    / f"{model_identifier}-{selfies.dataset_identifier}-{experiment.run_id}-{experiment.date}.png"
)


n_valid_history = np.zeros(n_epochs)
n_unseen_history = np.zeros(n_epochs)
n_unique_history = np.zeros(n_epochs)
for i in range(1, n_epochs + 1):
    n_unique = len(training_cache[f"{i}"]["samples"]["unique"])
    n_valid = len(training_cache[f"{i}"]["samples"]["valid"])
    n_unseen = len(training_cache[f"{i}"]["samples"]["unseen"])

    n_unique_history[i - 1] = n_unique
    n_valid_history[i - 1] = n_valid
    n_unseen_history[i - 1] = n_unseen


fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

axs[0].plot(
    wma(n_valid_history, 15),
    label="Number of Valid Samples (Total={})".format(n_test_samples),
)
axs[0].plot(
    wma(n_unseen_history, 15),
    label="Number of Unseen Samples (Total={})".format(n_test_samples),
)
axs[1].plot(
    wma(n_unique_history, 15),
    label="Number of Unique Samples (Total={})".format(n_test_samples),
)

title = """Model: {}\nHidden State Sampling Fn: {}""".format(
    model_name, hidden_state_fn_str
)
fig.suptitle(title)
plt.xlabel("Epoch")
axs[0].legend()
axs[1].legend()
plt.savefig(figure_path, bbox_inches="tight", format="png")


# for train_cache_ in [prior_train_cache]:
if prior_model != "mrQCBM":
    figure_path = (
        figures_dir
        / f"{model_identifier}-{selfies.dataset_identifier}-{experiment.run_id}-{experiment.date}_prior_cost.png"
    )
    plt.plot(prior_train_cache["history"]["opt_value"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(figure_path, bbox_inches="tight", format="png")


# generate samples
if n_compound_generation <= BATCHSIZE_GENERETATION:
    g_samples = generate_bulk_samples(
        model,
        n_compound_generation,
        n_generation_steps,
        4,
        prior=prior,
        verbose=True,
        unique=True,
    )

    saved_compounds_dir = Path(experiment._experiments_directory) / "compound_samples"
    saved_compounds_dir = Path(experiment._experiments_directory) / "compound_samples"
    file_name = f"compound_stats-{experiment.run_id}-{n_compound_generation}-{filter_constraint}"
    if ACTIVE_FILTER == True:
        results_analysis = compute_compound_stats(
            g_samples, decoder_fn, diversity_fn, validity_fn, train_compounds
        )
        compund_to_csv(
            results_analysis, file_path=f"{saved_compounds_dir}/{file_name}.csv"
        )
    data = {"samples": g_samples, "selfies": selfies, "prior": prior, "model": model}
    save_obj(data, f"{saved_compounds_dir}/{file_name}.pkl")
else:
    n_compound_generation_steps = 0
    while n_compound_generation_steps <= n_compound_generation:

        g_samples = generate_bulk_samples(
            model,
            BATCHSIZE_GENERETATION,
            n_generation_steps,
            4,
            prior=prior,
            verbose=True,
            unique=True,
        )

        saved_compounds_dir = (
            Path(experiment._experiments_directory) / "compound_samples"
        )
        file_name = f"compound_stats-{experiment.run_id}-{n_compound_generation}-{filter_constraint}_{n_compound_generation_steps}"

        if ACTIVE_FILTER == True:
            results_analysis = compute_compound_stats(
                g_samples, decoder_fn, diversity_fn, validity_fn, train_compounds
            )
            compund_to_csv(
                results_analysis, file_path=f"{saved_compounds_dir}/{file_name}.csv"
            )
        data = {
            "samples": g_samples,
            "selfies": selfies,
            "prior": prior,
            "model": model,
        }
        save_obj(data, f"{saved_compounds_dir}/{file_name}.pkl")
        n_compound_generation_steps += BATCHSIZE_GENERETATION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"Directory: {saved_compounds_dir}/{file_name}, Experiment ID {experiment.run_id}"
)
file_name_config = f"compound_config-{experiment.run_id}.txt"
content = f"{experiment.run_id}\n"
content += f"{selfies.dataset_identifier}\n"
content += f"{model.model_identifier}\n"
content += f"{model.__str__()}\n"
content += f"file saved at: {saved_compounds_dir}/{file_name}\n"
content += f"lstm_n_epochs : {lstm_n_epochs}\n"
content += f"n_layes_lstm : {n_layes_lstm}\n"
content += f"embedding_dim : {embedding_dim}\n"
content += f"hidden_dim : {hidden_dim}\n"
content += f"prior_n_epochs : {prior_n_epochs}\n"
content += f"n_generation_steps : {n_generation_steps}\n"
content += f"n_compound_generation : {n_compound_generation}\n"
content += f"prior_model : {prior.__str__()}\n"
content += f"filter_constraint : {filter_constraint}\n"
content += f"file_name : {file_name}\n"

with open(f"{saved_compounds_dir}/{file_name_config}", "w") as file:
    file.write(content)
