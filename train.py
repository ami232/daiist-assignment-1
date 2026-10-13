"""Reproduce preparation, validation experiments, three models and test evaluation.

Usage: uv run python main.py train
Model decisions were established on validation; test results do not feed training.
"""
import prepare_data
import baseline
import train_sklearn
import train_log_sklearn
import train_pytorch
import evaluate_test


def main():
    for stage in [prepare_data, baseline, train_sklearn, train_log_sklearn,
                  train_pytorch, evaluate_test]:
        print(f'\nRunning {stage.__name__}', flush=True)
        stage.main()
    print('\nTraining pipeline complete. Saved artifacts are under results/.')


if __name__ == '__main__':
    main()
