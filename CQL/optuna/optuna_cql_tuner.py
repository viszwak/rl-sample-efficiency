import optuna
from optuna_train_cql import train_cql

def objective(trial):
    cql_alpha = trial.suggest_float("cql_alpha", 0.1, 10.0, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512])
    num_samples = trial.suggest_int("num_samples", 5, 20)

    avg_score = train_cql(
        sim=trial.number,
        cql_alpha=cql_alpha,
        batch_size=batch_size,
        num_samples=num_samples,
        max_steps=350_000
    )
    return avg_score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    print(study.best_trial.params)
