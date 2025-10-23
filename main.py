from model.finetuner import run_fine_tuning
from model.trainer import run_model_training


def run_demo() -> None:
    base_mse = run_model_training()
    tuned_mse = run_fine_tuning()
    print(f"MSE on initial data: {base_mse}")
    print(f"MSE on new data: {tuned_mse}")

if __name__ == "__main__":
    run_demo()