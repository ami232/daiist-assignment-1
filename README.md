# Bike demand planner — Assignment 1

Javier Corredera · BBADBA 5B

A calendar-only linear regression predicts hourly Capital Bikeshare rental counts. It is trained three ways: scikit-learn Ridge, manual PyTorch with raw tensors/autograd, and PyTorch `nn.Module` with SGD. An illustrative city-center shop scenario motivates demand planning; the Washington DC system-wide data cannot validate a shop's exact inventory requirement.

## Run

From this repository folder, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync --locked
uv run python main.py train
uv run python main.py app
```

Open **http://127.0.0.1:7860** in a browser after launching the app. Stop it with Ctrl+C in the terminal. The model artifacts are committed, so the dashboard can also launch after `uv sync --locked` without rerunning training. App startup does not fit any model.

```sh
uv run pytest -q
```

The original `main.py` and GitHub verification workflow are retained. Each stage has a script only; the unused blank notebook templates have been removed.

## Files

- `REPORT.md`: the business framing, process, measured results, limitations, and AI disclosure. **The AI-assisted interpretation is a draft for student review before submission.**
- `data/hour.csv`: original, committed UCI hourly data; `data/README.md` records provenance and checksum.
- `features.py`: calendar encoding, date splits, and shared feature transformation.
- `models.py`: the two PyTorch implementations of one linear equation.
- `train.py`: train-only preprocessing, three models, validation feature experiment, baseline, and saved artifacts.
- `app.py`: load models and display comparison plots, distributions, and historical predictions.
- `tests/test_project.py`: no count/weather leakage, chronological splits, saved-model agreement, and inference without fitting or artifact writes.
- `artifacts/`: saved preprocessing/models, hold-out predictions, learning curves, and measured results.

## Measured result

On October–December 2012 test hours, all three models have RMSE ≈134.93 rentals/hour and MAE ≈103.36. The training-mean baseline has RMSE ≈208.06. Separate hour/season features give validation RMSE ≈150.07; season–hour indicators give ≈143.46. Maximum disagreement between scikit-learn and PyTorch is under 0.001 rental/hour. See `artifacts/metrics.json` for exact values.

The model uses L2-regularized linear regression with the same loss in all three versions. `alpha = n_train × lambda` matches scikit-learn's sum-of-errors objective to PyTorch's mean-of-errors objective. Neither version penalizes the intercept.

## Submission steps

1. Review and revise the marked report analysis into your own conclusions, as required by the brief.
2. Push the finished branch to your GitHub fork.
3. Open a pull request to `ami232/daiist-assignment-1` and check the verification workflow.
4. Submit the PR link and a ZIP of the project to Blackboard.

Dataset: [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset), Fanaee-T (2013), DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894), CC BY 4.0 according to UCI. The original assignment is at [ami232/daiist-assignment-1](https://github.com/ami232/daiist-assignment-1).
