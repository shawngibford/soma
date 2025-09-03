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
    RewardAPI
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
import dill
import cloudpickle
print("started!")
diversity_fn = get_diversity
start_time = time.time()
syba = SybaClassifier()
syba.fitDefaultScore()
print("Syba fitting time: ", time.time() - start_time)



base_url = "https://rip.chemistry42.com"
username = "m.ghazivakili"
password = "hJEV0jm5bgqX"
reward_api = RewardAPI(username=username,password=password,base_url=base_url)





def combine_filter(smiles_compound, max_mol_weight=800):
    pass_all = []
    i = 0

    with tqdm.tqdm(total=len(smiles_compound)) as pbar:
        for smile_ in smiles_compound:
            pbar.set_description(
                f"Filtered {i} / {len(smiles_compound)}. passed= {len(pass_all)}"
            )
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
            i += 1
            pbar.update()
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


# save in file:
def save_obj(obj, file_path):
    with open(file_path, "wb") as f:
        r = cloudpickle.dump(obj, f)
    return r


def load_obj(file_path):
    with open(file_path, "rb") as f:
        obj = cloudpickle.load(f)
    return obj


max_mol_weight = 800
saved_compounds_dir = "experiment_results/samples"

file_path ={
    "mqcbm" : "/home/mghazi/workspace/insilico-drug-discovery-bk/experiment_results/epoch_plots/noisy-lstm-v3-2023_17_03T12_51_02.776149/mode_prior_45.pkl",
    "qcbm" : "/home/mghazi/workspace/insilico-drug-discovery-bk/experiment_results/epoch_plots/noisy-lstm-v3-2023_17_03T12_51_25.971216/mode_prior_45.pkl",
    "rbm" : "/home/mghazi/workspace/insilico-drug-discovery-bk/experiment_results/epoch_plots/noisy-lstm-v3-2023_17_03T12_52_12.423234/mode_prior_45.pkl",
    "random": "/home/mghazi/workspace/insilico-drug-discovery-bk/experiment_results/epoch_plots/noisy-lstm-v3-2023_17_03T12_52_31.383903/mode_prior_45.pkl",
    "rmqcbm" : "/home/mghazi/workspace/insilico-drug-discovery-bk/experiment_results/epoch_plots/noisy-lstm-v3-2023_17_03T16_48_42.485415/mode_prior_45.pkl"
}


if len(sys.argv) >= 2:
    model_type = str(sys.argv[1])

    n_samples = int(sys.argv[2])
    print(model_type,n_samples)

else:
    print("please give at least 4 input args.")
    # exit()
    model_type = "mqcbm"
    n_samples = 10


try:
    object_path = file_path[model_type]
    object = load_obj(object_path)
    selfies = object["selfies"]
    model = object["model"]
    prior = object["prior"]
    mode_samples = object["mode_samples"]
    prior_samples = object["prior_samples"]
except Exception as e:
    print(f"Unable to open model and prior in {sys.argv[1]}: {e}")
    exit()


print(len(mode_samples))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to_device(device)

decoder_fn = selfies.decode_fn
# validity_fn = partial(combine_filter, max_mol_weight=max_mol_weight)
train_compounds = selfies.train_samples


def send_to_chemistry42(model_type,model,prior,n_samples):
    not_submitted = []
    submitted = {}
    for i in range(n_samples//10):
        g_samples = generate_bulk_samples(
            model,
            10,
            10,
            1,
            prior=prior,
            verbose=True,
            unique=True,
        )
        print(g_samples.shape)
        smiles_list  = decoder_fn(g_samples)
        print(smiles_list)
        try:
            workflow_uuid = reward_api.post_smiles(f"{model_type}_{i}",0,smiles_list)
            print(workflow_uuid)
            submitted[workflow_uuid]=smiles_list
        except:
            not_submitted.append(smiles_list)
            print("not posted!")
        print(time.sleep(5))
    return (submitted,not_submitted)



def submit_smiles_to_chemistry42(model_type,to_submit):
    not_submitted = []
    submitted = {}
    for smiles_list in to_submit:
        print(smiles_list)
        try:
            workflow_uuid = reward_api.post_smiles(f"{model_type}_{i}",0,smiles_list)
            print(workflow_uuid)
            submitted[workflow_uuid]=smiles_list
        except:
            not_submitted.append(smiles_list)
            print("not posted!")
            # print(time.sleep(60))
        print(time.sleep(5))
    return (submitted,not_submitted)


if __name__ == "__main__":
    print(model_type,n_samples)
    submited_workflows,not_submitted = send_to_chemistry42(model_type,model,prior,n_samples)
    total = len(not_submitted)
    print(f"{model_type} not submitted {not_submitted}")
    n = 0
    while total>0:
        submited_workflows_loop,not_submitted_loop = submit_smiles_to_chemistry42(model_type,not_submitted)
        submited_workflows.update(submited_workflows_loop)
        total = len(not_submitted_loop)
        print(n,total)
        n+=1
        if total == 0:
            print("Finished")
            break
    run_date_time = datetime.today().strftime("%Y_%d_%mT%H_%M_%S.%f")
    Path(f"{saved_compounds_dir}").mkdir(parents=True, exist_ok=True)
    save_obj(submited_workflows,f"{saved_compounds_dir}/{model_type}_{n_samples}_{run_date_time}.pkl")    

