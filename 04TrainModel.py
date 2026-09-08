from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

DATA_DIR = Path(__file__).resolve().parent / "data"

TARGET = "remaining_win_pct"

BASE_FEATURES = [
    "off_xbsr_per_game",
    "allowed_xbsr_per_game"
]

EXTENDED_FEATURES = [
    "off_xbsr_per_game",
    "allowed_xbsr_per_game",
    "pitch_k_minus_bb_rate"
]


def load_data():
    file_path = DATA_DIR / "training_data.parquet"

    data = pd.read_parquet(file_path)

    data["Cutoff_Date"] = pd.to_datetime(data["Cutoff_Date"])

    july = data[
        data["Cutoff_Date"].dt.strftime("%m-%d") == "07-01"
    ].copy()

    july = july[july["Season"] != 2020]

    july = july[
        july["Season"].between(2016, 2025)
    ]

    july = july.sort_values(
        ["Season", "Team_ID"]
    )

    july = july.reset_index(drop=True)

    return july


def check_data(july):
    duplicates = july.duplicated(
        ["Season", "Team_ID"]
    )

    assert duplicates.any() == False

    important_columns = [
        "off_xbsr_per_game",
        "allowed_xbsr_per_game",
        "pitch_k_minus_bb_rate",
        "remaining_win_pct",
        "win_pct"
    ]

    missing_values = july[important_columns].isna()

    assert missing_values.any().any() == False

    assert july[TARGET].between(0, 1).all()


def split_data(july):
    train = july[
        july["Season"].between(2016, 2022)
    ].copy()

    validation = july[
        july["Season"].between(2023, 2024)
    ].copy()

    test = july[
        july["Season"] == 2025
    ].copy()

    return train, validation, test


def train_model(data, features):
    X = data[features]
    y = data[TARGET]

    model = LinearRegression()

    model.fit(X, y)

    return model


def make_predictions(model, data, features):
    X = data[features]

    predictions = model.predict(X)

    predictions = predictions.clip(0, 1)

    return predictions


def calculate_mae(actual, predicted):
    mae = mean_absolute_error(actual, predicted)

    return mae


def main():
    july = load_data()

    check_data(july)

    print("July rows:", len(july))

    train, validation, test = split_data(july)

    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Test rows:", len(test))

    baseruns_model = train_model(
        train,
        BASE_FEATURES
    )

    baseruns_k_bb_model = train_model(
        train,
        EXTENDED_FEATURES
    )

    validation_predictions = validation[
        ["Season", "Cutoff_Date", "Team_ID", "Team", TARGET]
    ].copy()

    validation_predictions["always_500"] = 0.5

    validation_predictions["current_win_pct"] = validation["win_pct"]

    validation_predictions["baseruns"] = make_predictions(
        baseruns_model,
        validation,
        BASE_FEATURES
    )

    validation_predictions["baseruns_k_bb"] = make_predictions(
        baseruns_k_bb_model,
        validation,
        EXTENDED_FEATURES
    )

    always_500_mae = calculate_mae(
        validation[TARGET],
        validation_predictions["always_500"]
    )

    current_record_mae = calculate_mae(
        validation[TARGET],
        validation_predictions["current_win_pct"]
    )

    baseruns_mae = calculate_mae(
        validation[TARGET],
        validation_predictions["baseruns"]
    )

    baseruns_k_bb_mae = calculate_mae(
        validation[TARGET],
        validation_predictions["baseruns_k_bb"]
    )

    validation_results = pd.DataFrame({
        "model": [
            "always_500",
            "current_win_pct",
            "baseruns",
            "baseruns_k_bb"
        ],
        "mae": [
            always_500_mae,
            current_record_mae,
            baseruns_mae,
            baseruns_k_bb_mae
        ]
    })

    validation_results["mae_percentage_points"] = (
        validation_results["mae"] * 100
    )

    print()
    print("Validation Results")
    print(validation_results.sort_values("mae").to_string(index=False))

    if baseruns_k_bb_mae < baseruns_mae:
        selected_name = "baseruns_k_bb"
        selected_features = EXTENDED_FEATURES
    else:
        selected_name = "baseruns"
        selected_features = BASE_FEATURES

    print()
    print("Selected model:", selected_name)
    print("Selected features:", selected_features)

    train_and_validation = pd.concat(
        [train, validation],
        ignore_index=True
    )

    final_model = train_model(
        train_and_validation,
        selected_features
    )

    test_predictions = test[
        ["Season", "Cutoff_Date", "Team_ID", "Team", TARGET]
    ].copy()

    test_predictions["always_500"] = 0.5

    test_predictions["current_win_pct"] = test["win_pct"]

    test_predictions[selected_name] = make_predictions(
        final_model,
        test,
        selected_features
    )

    always_500_test_mae = calculate_mae(
        test[TARGET],
        test_predictions["always_500"]
    )

    current_record_test_mae = calculate_mae(
        test[TARGET],
        test_predictions["current_win_pct"]
    )

    model_test_mae = calculate_mae(
        test[TARGET],
        test_predictions[selected_name]
    )

    test_results = pd.DataFrame({
        "model": [
            "always_500",
            "current_win_pct",
            selected_name
        ],
        "mae": [
            always_500_test_mae,
            current_record_test_mae,
            model_test_mae
        ]
    })

    test_results["mae_percentage_points"] = (
        test_results["mae"] * 100
    )

    test_predictions["absolute_error"] = abs(
        test_predictions[TARGET]
        - test_predictions[selected_name]
    )

    print()
    print("Test Results")
    print(test_results.sort_values("mae").to_string(index=False))

    print()
    print(
        "Selected regression MAE:",
        round(model_test_mae * 100, 2),
        "percentage points"
    )

    if (
        model_test_mae < current_record_test_mae
        and model_test_mae < always_500_test_mae
    ):
        print("The selected regression beat both benchmarks.")
    else:
        print("The selected regression did not beat both benchmarks.")

    print()
    print("Largest prediction errors")

    largest_errors = test_predictions.sort_values(
        "absolute_error",
        ascending=False
    ).head(6)

    print(largest_errors.to_string(index=False))

    validation_predictions.to_parquet(
        DATA_DIR / "validation_predictions.parquet",
        index=False
    )

    validation_results.to_parquet(
        DATA_DIR / "validation_results.parquet",
        index=False
    )

    test_predictions.to_parquet(
        DATA_DIR / "test_predictions.parquet",
        index=False
    )

    test_results.to_parquet(
        DATA_DIR / "test_results.parquet",
        index=False
    )

    return test_predictions, test_results, selected_name


if __name__ == "__main__":
    test_predictions, test_results, selected_name = main()