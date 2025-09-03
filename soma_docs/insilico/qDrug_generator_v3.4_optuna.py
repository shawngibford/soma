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
)

diversity_fn = get_diversity
start_time = time.time()
syba = SybaClassifier()
syba.fitDefaultScore()
print("Syba fitting time: ", time.time() - start_time)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 128
NEPOCH = 20

print(f"{DEVICE} is available")
print(f"{DEVICE} is available")


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


def combine_filter(smiles_compound, max_mol_weight=800):
    pass_all = []
    for smile_ in smiles_compound:
        try:
            if (
                apply_filters(smile_, max_mol_weight)
                and smile_ not in pass_all
                and (syba.predict(smile_) > 0)
                and passes_wehi_mcf(smile_)
                and (len(pains_filt(Chem.MolFromSmiles(smile_))) == 0)
            ):

                pass_all.append(smile_)
        except:
            pass
    return pass_all


def reward_fc(smiles_ls):
    rewards = []
    for smiles_compound in smiles_ls:
        #: TODO: add wieghts for filter
        try:
            reward = 0
            if smiles_compound not in smiles_ls:
                reward += 5
                if apply_filters(smiles_compound):
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


def qDrug_main(
    trials,
    filter_constraint="hard",
    prior_model="QCBM",
    prior_size=10,
    prior_n_epochs=20,
    data_set=1,
):

    print("Optuna will be apply to optimize the model")
    print(f"filter_constraint{filter_constraint}, prior_model {prior_model}")
    n_layes_lstm = trials.suggest_int("n_layes_lstm", 2, 5)
    hidden_dim = trials.suggest_int("hidden_dim", 32, 256)
    embedding_dim = trials.suggest_int("embedding_dim", 32, 512)
    # lr = trials.suggest_int(
    #     "learning_rate",
    #     0.1 or 0.01,
    # )
    n_qcbm_layers = trials.suggest_int("n_qcbm_layers", 4, 10)
    lstm_n_epochs = NEPOCH  # trials.suggest_int("lstm_n_epochs", 50, 200)

    if filter_constraint == "hard":
        filter_fc = combine_filter
        rew_fc = reward_fc
    else:
        filter_fc = filter_legacy
        rew_fc = reward_fc_legacy

    #### inputs ####
    n_test_samples = 20000
    max_mol_weight = 800
    prior_sample_size = prior_size
    prior_hidden_layer = 10
    lstm_n_batch_size = 32
    if data_set == 0:
        # 700
        path_to_dataset = "data/KRAS_G12D/KRAS_G12D_inhibitors_update2023.csv"
    elif data_set == 1:
        # 12k
        path_to_dataset = "data/KRAS_G12D/initial_dataset.csv"
    elif data_set == 2:
        # 1M
        path_to_dataset = (
            "data/KRAS_G12D/initial_data_with_chemistry42_syba_merged_v2.csv"
        )
    path_to_model_weights = None

    run_date_time = datetime.today().strftime("%Y_%d_%mT%H_%M")
    experiment = Experiment(run_id=f"noisy-lstm-v3-{run_date_time}")
    print(f"Experiment ID: {experiment.run_id}")

    selfies = SelfiesEncoding(path_to_dataset, dataset_identifier="insilico_KRAS")

    print(f"Using file: {selfies._filepath}.")
    print(f"Dataset identifier: {selfies.dataset_identifier}")

    optimizer = ScipyOptimizer(method="Powell", options={"maxiter": 1})
    if prior_model == "QCBM":
        # QCBM
        entangling_layer_builder = LineEntanglingLayerBuilder(
            n_qubits=prior_sample_size
        )
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

    encoded_samples_th = torch.tensor(selfies.encoded_samples)
    data = encoded_samples_th.float()

    decoder_fn = selfies.decode_fn
    truncate_fn = truncate_smiles
    validity_fn = partial(filter_fc, max_mol_weight=max_mol_weight)
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
        data=data, batch_size=training_parameters["batch_size"]
    ).shuffle(12345)
    train_cache = TrainCache()

    if prior_model == "QCBM" or prior_model == "mQCBM":
        batch_size = -1
    else:
        batch_size = training_parameters["batch_size"]

    # start training
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
            th_prior_samples = torch.tensor(concat_prior_samples)
            # put model in evaluation mode such that layers like Dropout, Batchnorm don't affect results
            model.set_eval_state()

            # prior training
            if epoch == 1:
                prior_samples_current = torch.tensor(
                    prior.generate(n_test_samples)
                ).float()
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
                ).shuffle(12345)

                # TODO track prior training cache
                prior_train_cache = SimpleTrainer().train(
                    prior,
                    prior_train_data,
                    n_epochs=prior_n_epochs,
                    disable_progress_bar=True,
                )

                # generate compounds and then decode them such that we are working with sequences of str
                prior_samples_current = torch.tensor(
                    prior.generate(n_test_samples)
                ).float()
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
            )

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
        trials.report(np.mean([compound_stats.unique_fraction,compound_stats.filter_fraction,compound_stats.filter_fraction,compound_stats.diversity_fraction]), epoch)
        if trials.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(
        [compound_stats.unique_fraction,
        compound_stats.filter_fraction,
        compound_stats.filter_fraction,
        compound_stats.diversity_fraction]
    )


if __name__ == "__main__":

    if len(sys.argv) > 1:
        prior_model = str(sys.argv[1])
        filter_constraint = str(sys.argv[2])
        objective = partial(
            qDrug_main, filter_constraint=filter_constraint, prior_model=prior_model
        )
        storage = f"sqlite:///qdrug_discovery_{prior_model}.db"
        study_name = f"qDrug_{prior_model}"
    else:
        prior_model = "QCBM"
        filter_constraint = "hard"
        objective = partial(qDrug_main, filter_constraint="hard", prior_model="mQCBM")
        storage = f"sqlite:///qdrug_discovery_{prior_model}.db"
        study_name = f"qDrug_{prior_model}"

    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
