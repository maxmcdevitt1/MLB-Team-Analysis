# 04: Train and evaluate a simple model using July 1 snapshots.
# Same calculations and Parquet outputs as 04TrainModel.ipynb.
# Train: 2016-2022 excluding 2020. Validation: 2023-2024. Test: 2025.
# Final/remaining-result columns are never input features.
# MAE is average absolute error; 0.050 means five percentage points.
# Regression predictions are clipped to 0-1 before scoring.

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

DATA_DIR = Path(__file__).resolve().parent / "data"
TARGET = "remaining_win_pct"

BASE_FEATURES = ["off_xbsr_per_game", "allowed_xbsr_per_game"]
EXTENDED_FEATURES = BASE_FEATURES + ["pitch_k_minus_bb_rate"]
MODEL_FEATURES = {
    "baseruns": BASE_FEATURES,
    "baseruns_k_bb": EXTENDED_FEATURES,
}
ID_COLUMNS = ["Season", "Cutoff_Date", "Team_ID", "Team", TARGET]
BENCHMARKS = ["always_500", "current_win_pct"]


def load_july_data(data_dir):
    """Load and validate July 1 snapshots for the included seasons."""
    data = pd.read_parquet(data_dir / "training_data.parquet")
    cutoff_dates = pd.to_datetime(data["Cutoff_Date"])

    july = data[cutoff_dates.dt.strftime("%m-%d") == "07-01"].copy()
    july = july[july["Season"] != 2020]
    july = july[july["Season"].between(2016, 2025)]
    july = july.sort_values(["Season", "Team_ID"]).reset_index(drop=True)

    assert not july.duplicated(["Season", "Team_ID"]).any()
    assert not july[EXTENDED_FEATURES + [TARGET, "win_pct"]].isna().any().any()
    assert july[TARGET].between(0, 1).all()

    return july


def split_by_season(july):
    """Keep training, validation, and test seasons separate."""
    train = july[july["Season"].between(2016, 2022)].copy()
    validation = july[july["Season"].between(2023, 2024)].copy()
    test = july[july["Season"] == 2025].copy()

    # For this dataset there should be 30 teams per included season.
    assert len(train) == 180
    assert len(validation) == 60
    assert len(test) == 30

    return train, validation, test


def print_split_summary(train, validation, test):
    """Show the seasons and row counts used by each split."""
    split_summary = pd.DataFrame({
        "split": ["Training", "Validation", "Test"],
        "seasons": ["2016-2022, excluding 2020", "2023-2024", "2025"],
        "rows": [len(train), len(validation), len(test)],
    })
    print(split_summary.to_string(index=False))


def fit_regression(data, features):
    """Fit a linear regression using only the requested input features."""
    model = LinearRegression()
    model.fit(data[features], data[TARGET])
    return model


def make_predictions(data, models):
    """Predict both benchmarks and each named regression, clipped to 0–1."""
    predictions = data[ID_COLUMNS].copy()
    predictions["always_500"] = 0.5
    predictions["current_win_pct"] = data["win_pct"]
    for name, model in models.items():
        predictions[name] = model.predict(data[MODEL_FEATURES[name]]).clip(0, 1)
    return predictions


def score_predictions(predictions, model_names):
    """Calculate MAE in fractions and percentage points for each approach."""
    scores = []
    for name in model_names:
        mae = mean_absolute_error(predictions[TARGET], predictions[name])
        scores.append({"model": name, "mae": mae, "mae_percentage_points": mae * 100})
    return pd.DataFrame(scores)


def select_regression(validation_results):
    """Choose by validation MAE, preferring the base regression on a tie."""
    scores = validation_results.set_index("model")["mae"]
    if scores["baseruns_k_bb"] < scores["baseruns"]:
        return "baseruns_k_bb"
    return "baseruns"


def print_test_results(test_predictions, test_results, selected_name):
    """Report test performance against benchmarks and the largest misses."""
    print(test_results.sort_values("mae").to_string(index=False))

    scores = test_results.set_index("model")["mae"]
    model_mae = scores[selected_name]
    record_mae = scores["current_win_pct"]
    constant_mae = scores["always_500"]

    print(f"Selected regression MAE: {model_mae * 100:.2f} percentage points")
    if model_mae < min(record_mae, constant_mae):
        print("The selected regression beat both benchmarks on the 2025 test.")
    else:
        print("The selected regression did not beat both benchmarks on the 2025 test.")

    print(test_predictions.sort_values("absolute_error", ascending=False).head(6).to_string(index=False))


def save_results(data_dir, validation_predictions, validation_results,
                 test_predictions, test_results, selected_name):
    """Save all four outputs and verify the saved test predictions' MAE."""
    validation_predictions.to_parquet(data_dir / "validation_predictions.parquet", index=False)
    validation_results.to_parquet(data_dir / "validation_results.parquet", index=False)
    test_predictions.to_parquet(data_dir / "test_predictions.parquet", index=False)
    test_results.to_parquet(data_dir / "test_results.parquet", index=False)

    # Check that the saved predictions match the comparison score.
    saved_predictions = pd.read_parquet(data_dir / "test_predictions.parquet")
    saved_mae = abs(saved_predictions[TARGET] - saved_predictions[selected_name]).mean()
    model_mae = test_results.set_index("model").loc[selected_name, "mae"]
    assert abs(saved_mae - model_mae) < 0.000000001
    print("Saved four Parquet files. The saved predictions reproduce the test MAE.")


def main():
    july = load_july_data(DATA_DIR)
    print(f"July team-season rows: {len(july)}")
    print(july[["Season", "Team", "Cutoff_Date", "win_pct"] + BASE_FEATURES].head().to_string(index=False))

    train, validation, test = split_by_season(july)
    print_split_summary(train, validation, test)

    models = {
        name: fit_regression(train, features)
        for name, features in MODEL_FEATURES.items()
    }
    validation_predictions = make_predictions(validation, models)
    validation_results = score_predictions(validation_predictions, BENCHMARKS + list(models))
    print(validation_results.sort_values("mae").to_string(index=False))

    selected_name = select_regression(validation_results)
    selected_features = MODEL_FEATURES[selected_name]
    print("Selected regression:", selected_name)
    print("Selected features:", selected_features)

    train_and_validation = pd.concat([train, validation], ignore_index=True)
    final_model = fit_regression(train_and_validation, selected_features)
    test_predictions = make_predictions(test, {selected_name: final_model})
    test_results = score_predictions(test_predictions, BENCHMARKS + [selected_name])
    test_predictions["absolute_error"] = abs(test_predictions[TARGET] - test_predictions[selected_name])
    print_test_results(test_predictions, test_results, selected_name)

    save_results(DATA_DIR, validation_predictions, validation_results,
                 test_predictions, test_results, selected_name)
    return test_predictions, test_results, selected_name

if __name__ == "__main__":
    test_predictions, test_results, selected_name = main()