import os
import tempfile

import cloudpickle
import mlflow
import numpy as np
import torch
from more_itertools import sort_together
from orquestra.qml.api import ProbabilityDistribution, convert_to_torch
from orquestra.qml.distance_measures import KLDivergence
from orquestra.qml.experimental.models.recurrent.lstm.th import NoisyLSTM
from orquestra.sdk.mlflow import get_tracking_token

os.environ["MLFLOW_TRACKING_TOKEN"] = get_tracking_token(config_name="research")
mlflow.set_tracking_uri(
    "https://research.orquestra.io/mlflow/drug-discovery-dwave-250cee/"
)


def sample_to_str(single_sample):
    x = tuple(single_sample.astype(int).tolist())
    return "".join(map(str, x))


def molecule_to_str(single_sample, eos_token):
    x = tuple(single_sample.astype(int).tolist())
    return "-".join(map(str, x)).split(str(eos_token))[0]


def create_molecule_from_bitstrings(
    bitstrings: torch.Tensor, model: NoisyLSTM, n_samples: int
):
    model.do_greedy_sampling = True  # disables random sampling
    n_samples = bitstrings.shape[0]
    generated_molecules, _ = model._generate_w_probs(
        n_samples, prior_samples=bitstrings
    )
    generated_molecules = generated_molecules.cpu().numpy()
    return [
        molecule_to_str(molecule, model.padding_token_index)
        for molecule in generated_molecules
    ]


def load_model(run_id: str) -> tuple[NoisyLSTM, str]:
    run_name = mlflow.get_run(run_id).data.tags["mlflow.runName"]
    # List artifacts for run_id:
    artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
    for artifact in artifacts:
        if "result_and_model" in artifact.path:
            # Download the artifact in temporary directory:
            with tempfile.TemporaryDirectory() as tmpdirname:
                mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=artifact.path, dst_path=tmpdirname
                )
                # Load the artifact:
                with open(os.path.join(tmpdirname, artifact.path), "rb") as f:
                    result_and_model = cloudpickle.load(f)
                    return result_and_model["lstm"], run_name
    raise ValueError("No model found in artifacts")


# model, run_name = load_model("c48274819b8141bd85a0d40df48252c2")  # big RBM iteration 0
# model, run_name = load_model(
#     "53c1b1408280484fa938e0f0911d9008"
# )  # small RBM iteration 0
model, run_name = load_model("c6f6f5320d7c42e3aa09eaca1bd66bac")  # random iteration 0
bits = model.prior_sample_dim
n_samples = 1000
# create an array of random bits of shape (n_samples, bits)
torch.manual_seed(1260)
prior_samples = torch.randint(0, 2, (n_samples, bits)).float()
# keep unique prior samples:
prior_samples = torch.unique(prior_samples, dim=0)
# move model and samples to GPU:
device = torch.device("cuda:0")
model.to_device(device)
prior_samples = prior_samples.to(device)
# generate molecules from the prior samples
molecules = create_molecule_from_bitstrings(prior_samples, model, n_samples)
print("bits", bits)
print("unique molecules", len(set(molecules)))
print("total molecules", len(molecules))
print("diversity", len(set(molecules)) / len(molecules))
