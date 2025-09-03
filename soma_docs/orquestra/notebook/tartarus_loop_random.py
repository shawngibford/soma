import logging
import sys

from tartarus_loop import run_tartarus_loop

# Set logging level:
logging.basicConfig(level=logging.INFO)
# Send logging to stdout:
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


N_QUBITS = 16
GPU_ID = 5
STATS_SAMPLES = 50_000
N_CPUS = 220
MOLECULE = "6y2f"
MODEL = "random"
seed = 51009

prior_args = {}

run_id = ""
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
