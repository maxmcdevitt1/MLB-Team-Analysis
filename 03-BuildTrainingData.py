# 03: Add the remaining-season win percentage to the saved features.

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
TARGET = "remaining_win_pct"


def build_training_data(metrics, final_records):
    training = metrics.merge(
        final_records,
        on=["Season", "Team_ID"],
        how="left",
        validate="many_to_one",
    )

    if training[["final_W", "final_L"]].isna().any().any():
        raise ValueError("Some teams are missing their final record")

    training["remaining_wins"] = training["final_W"] - training["W"]
    training["remaining_losses"] = training["final_L"] - training["L"]
    training["remaining_games"] = training["remaining_wins"] + training["remaining_losses"]

    if (training[["remaining_wins", "remaining_losses"]] < 0).any().any():
        raise ValueError("A cutoff record is larger than the final record")

    # A team needs games left to have a remaining-season win percentage.
    training = training[training["remaining_games"] > 0].copy()
    training[TARGET] = training["remaining_wins"] / training["remaining_games"]

    return training.sort_values(["Season", "Cutoff_Date", "Team_ID"]).reset_index(drop=True)


def main():
    metrics = pd.read_parquet(DATA_DIR / "model_metrics.parquet")
    final_records = pd.read_parquet(DATA_DIR / "final_records.parquet")
    training = build_training_data(metrics, final_records)
    training.to_parquet(DATA_DIR / "training_data.parquet", index=False)
    print(f"Saved {len(training)} training rows")


if __name__ == "__main__":
    main()
