import optuna
from train_lstm import objective

GPU_ID = 4
N_QUBITS = 8


def suggest(trial: optuna.Trial):
    J0 = trial.suggest_float("J0", 1e-4, 0.1, log=True)
    phi0 = trial.suggest_float("phi0", 1e-6, 0.1, log=True)
    x_simple_anneal_time = trial.suggest_float(
        "x_simple_anneal_time", 0.007, 0.1, log=True  # 7 ns to 100 ns
    )
    return {
        "J": J0,
        "phi": phi0,
        "generation_max_retries": 10,
        "x_simple_anneal_time": x_simple_anneal_time,
    }


def optuna_objective(trial: optuna.Trial):
    return objective(
        trial,
        suggest,
        "qbag",
        N_QUBITS,
        GPU_ID,
        compute_final_novelity_rate=False,
        compute_final_diversity_rate=False,
        compute_final_sr_rate=True,
        prior_n_epochs=5,
    )


if __name__ == "__main__":
    import logging
    import sys

    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    # Send logging to stdout:
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    study = optuna.create_study(
        storage=f"sqlite:///optuna_{N_QUBITS}bits_qbag.db",
        study_name=f"optuna_{N_QUBITS}bits_qbag",
        directions=["minimize", "maximize"],
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        study.enqueue_trial(
            {
                "J": 0.05,
                "phi": 1e-5,
                "x_simple_anneal_time": 0.02,
            }
        )
    study.optimize(optuna_objective, n_trials=50)
