import logging
import sys

from orquestra.qml.experimental.models.recurrent.lstm.th import AddXY
from tartarus_loop import run_tartarus_loop

# Set logging level:
logging.basicConfig(level=logging.INFO)
# Send logging to stdout:
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


N_QUBITS = 224  # same embedding dim
GPU_ID = 7
STATS_SAMPLES = 50_000
N_CPUS = 235
MOLECULE = "1syh"
MODEL = "zeros"
seed = 161507
prior_args = {}

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
    run_name_suffix=f"lower_similarity_seed_{seed}",
    similarity_limit=0.5,
    skip_loading_from_run_id=True,
    random_seed=seed,
    use_vectorized_cost_fn=False,
    cluster="lambda04",
    use_linear_projection=False,
    merge_prior_fc=AddXY(),
    use_kaiming_init=False,
    sampling_temperature=1e-4,
)
