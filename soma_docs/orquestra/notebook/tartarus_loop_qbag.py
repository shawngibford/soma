import logging
import sys

import optuna
from tartarus_loop import run_tartarus_loop

# Set logging level:
logging.basicConfig(level=logging.INFO)
# Send logging to stdout:
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


N_QUBITS = 16
GPU_ID = 7
STATS_SAMPLES = 50_000
N_CPUS = 220
MOLECULE = "6y2f"
MODEL = "qbag"
seed = 6987322

if MODEL == "qbag" and N_QUBITS in [16, 128]:
    study = optuna.load_study(
        study_name=f"optuna_{N_QUBITS}bits_{MODEL}_syba",
        storage=f"sqlite:///optuna_{N_QUBITS}bits_{MODEL}_syba.db",
    )

    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]
    best_trial = df.sort_values("values_0").iloc[0]
    J0 = best_trial["params_J0"]
    phi0 = best_trial["params_phi0"]
    x_simple_anneal_time = best_trial["params_x_simple_anneal_time"]
else:
    J0 = 1e-2
    phi0 = 2e-4
    x_simple_anneal_time = 0.038
prior_args = {"J": J0, "phi": phi0, "x_simple_anneal_time": x_simple_anneal_time}

run_id = "5e75775974e74336b784832dc10123da"
run_tartarus_loop(
    N_QUBITS,
    GPU_ID,
    STATS_SAMPLES,
    N_CPUS,
    MOLECULE,
    MODEL,
    prior_args,
    run_id,
    run_name_suffix="lower_similarity_vectorized_docking_cost_fn",
    similarity_limit=0.5,
    skip_loading_from_run_id=True,
    random_seed=seed,
    use_vectorized_cost_fn=True,
)
