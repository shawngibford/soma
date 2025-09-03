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
MODEL = "qcbm"

study = optuna.load_study(
    study_name=f"optuna_{N_QUBITS}bits_{MODEL}_syba",
    storage=f"sqlite:///optuna_{N_QUBITS}bits_{MODEL}_syba.db",
)

df = study.trials_dataframe()
df = df[df["state"] == "COMPLETE"]
best_trial = df.sort_values("values_1").iloc[-1]
print(best_trial)
learning_rate = best_trial["params_learning_rate"]
n_layers = int(best_trial["params_n_layers"])
prior_args = {"learning_rate": learning_rate, "qcbm_n_layers": n_layers}

run_tartarus_loop(
    N_QUBITS,
    GPU_ID,
    STATS_SAMPLES,
    N_CPUS,
    MOLECULE,
    MODEL,
    prior_args,
    "6f93932cb409480e80feaefe0c8a5403",
    0.5,
    "lower_similarity",
)
