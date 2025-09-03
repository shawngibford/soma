import optuna
from train_lstm import objective

GPU_ID = 7
N_QUBITS = 16
NUM_TRIALS = 50


def suggest(trial: optuna.Trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    n_gibbs_steps = trial.suggest_int("n_gibbs_steps", 1, 10)

    return {"learning_rate": learning_rate, "n_gibbs_steps": n_gibbs_steps}


def optuna_objective(trial: optuna.Trial):
    if trial.number > NUM_TRIALS:
        raise RuntimeError("Number of trials exceeded.")
    return objective(
        trial,
        suggest,
        "rbm",
        N_QUBITS,
        GPU_ID,
        compute_final_novelity_rate=False,
        compute_final_diversity_rate=False,
        compute_final_sr_rate=True,
        prior_n_epochs=5,
        epochs=50,
        use_syba=True,
        cluster="lambda01",
    )


if __name__ == "__main__":
    import logging
    import sys

    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    # Send logging to stdout:
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    study = optuna.create_study(
        storage=f"sqlite:///optuna_{N_QUBITS}bits_rbm_syba.db",
        study_name=f"optuna_{N_QUBITS}bits_rbm_syba",
        directions=["minimize", "maximize"],
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        study.enqueue_trial(
            {
                "learning_rate": 0.001,
                "n_gibbs_steps": 5,
            }
        )
    study.optimize(optuna_objective, n_trials=NUM_TRIALS)
