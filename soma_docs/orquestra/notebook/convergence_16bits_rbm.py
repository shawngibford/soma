from train_lstm import GeneralFilter, PainFilter, WehiMCFilter, codecarbon_lstm_model

SEED = 623970
GPU_ID = 5
N_QUBITS = 16

if __name__ == "__main__":
    import logging
    import sys

    import optuna

    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    # Send logging to stdout:
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    study = optuna.load_study(
        study_name=f"optuna_{N_QUBITS}bits_rbm_syba",
        storage=f"sqlite:///optuna_{N_QUBITS}bits_rbm_syba.db",
    )

    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]
    best_trial = df.sort_values("values_1").iloc[-1]
    learning_rate = best_trial["params_learning_rate"]
    n_gibbs_steps = int(best_trial["params_n_gibbs_steps"])
    use_syba = True

    if use_syba:
        from orquestra.drug.discovery.validator import SybaFilter

        filter_lists = [GeneralFilter(), PainFilter(), WehiMCFilter(), SybaFilter()]
        weight_lists = [5.0, 3.0, 3.0, 5.0]
        experiment_name = "syba_lstm"
    else:
        filter_lists = [GeneralFilter(), PainFilter(), WehiMCFilter()]
        weight_lists = [5.0, 3.0, 3.0]
        experiment_name = "lstm"

    result = codecarbon_lstm_model(
        "rbm",
        N_QUBITS,
        {"learning_rate": learning_rate, "n_gibbs_steps": n_gibbs_steps},
        dataset_name="alzheimer",
        filter_lists=filter_lists,
        weight_lists=weight_lists,
        novelity_threshold=0.6,
        mlflow_experiment_name_prefix=experiment_name,
        batch_size=4096,
        mlflow_run_name_prefix=f"best_trial_convergence_seed_{SEED}",
        n_epochs=500,
        prior_n_epochs=5,
        random_seed=0,
        save_quality_metrics=False,
        gpu_id=GPU_ID,
        compute_final_sr_rate=True,
        compute_final_diversity_rate=True,
        compute_final_novelity_rate=True,
        cluster="lambda01",
    )
