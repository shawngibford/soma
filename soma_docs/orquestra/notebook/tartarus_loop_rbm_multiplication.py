import logging
import sys

from orquestra.qml.experimental.models.recurrent.lstm.th import MultiplyXY
from tartarus_loop import run_tartarus_loop

# Set logging level:
logging.basicConfig(level=logging.INFO)
# Send logging to stdout:
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


N_QUBITS = 224  # same embedding dim
GPU_ID = 0
STATS_SAMPLES = 50_000
N_CPUS = 30
MOLECULE = "1syh"
MODEL = "rbm"
seed = 245103
prior_args = {"learning_rate": 1e-3, "n_gibbs_steps": 5}

run_id = "20effd73346c4bacbd2331cb030881fd"
run_tartarus_loop(
    N_QUBITS,
    GPU_ID,
    STATS_SAMPLES,
    N_CPUS,
    MOLECULE,
    MODEL,
    prior_args,
    run_id,
    n_epochs=2,
    run_name_suffix="test_fresh_installation",
    similarity_limit=0.5,
    skip_loading_from_run_id=True,
    random_seed=seed,
    use_vectorized_cost_fn=False,
    cluster="lambda04",
    use_linear_projection=False,
    embedding_dim=N_QUBITS,
    merge_prior_fc=MultiplyXY(),
    sampling_temperature=1e-4,
)
