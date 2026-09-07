# 02: Read the raw Parquet data and calculate features.

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

AL_TEAM_IDS = [
    108, 110, 111, 114, 116, 117, 118, 133,
    136, 139, 140, 141, 142, 145, 147,
]
NL_TEAM_IDS = [
    109, 112, 113, 115, 119, 120, 121, 134,
    135, 137, 138, 143, 144, 146, 158,
]

FEATURE_SETS = {
    "baseline": [
        "off_xbsr_per_game",
        "allowed_xbsr_per_game",
    ],
    "extended": [
        "off_xbsr_per_game",
        "allowed_xbsr_per_game",
        "pitch_k_minus_bb_rate",
    ],
    "components": [
        "bat_hr_rate",
        "pitch_k_minus_bb_rate",
        "pitch_hr_rate",
    ],
    "expanded": [
        "off_xbsr_per_game", "allowed_xbsr_per_game",
        "bat_k_rate", "bat_bb_rate", "bat_hr_rate", "bat_iso",
        "pitch_k_minus_bb_rate", "pitch_hr_rate", "pitch_iso",
    ],
    "record_baseline": ["win_pct", "run_diff_per_game"],
}


def get_league(team_id):
    if team_id in AL_TEAM_IDS:
        return "AL"
    if team_id in NL_TEAM_IDS:
        return "NL"
    raise ValueError(f"Unknown MLB team ID: {team_id}")


def raw_bsr(df, side):
    if side == "bat":
        appearances = df["plateAppearances"]
    elif side == "pitch":
        appearances = df["battersFaced"]
    else:
        raise ValueError("side must be 'bat' or 'pitch'")

    hits = df[f"hits_{side}"]
    walks = df[f"baseOnBalls_{side}"]
    hit_by_pitch = df[f"hitByPitch_{side}"]
    intentional_walks = df[f"intentionalWalks_{side}"]
    home_runs = df[f"homeRuns_{side}"]
    total_bases = df[f"totalBases_{side}"]
    stolen_bases = df[f"stolenBases_{side}"]
    caught_stealing = df[f"caughtStealing_{side}"]
    double_plays = df[f"groundIntoDoublePlay_{side}"]
    sacrifice_flies = df[f"sacFlies_{side}"]
    sacrifice_bunts = df[f"sacBunts_{side}"]

    A = hits + walks + hit_by_pitch - 0.5 * intentional_walks - home_runs
    B = 1.1 * (
        1.4 * total_bases
        - 0.6 * hits
        - 3 * home_runs
        + 0.1 * (walks + hit_by_pitch - intentional_walks)
        + 0.9 * (stolen_bases - caught_stealing - double_plays)
    )
    C = (
        appearances - walks - sacrifice_flies - sacrifice_bunts
        - hit_by_pitch - hits + caught_stealing + double_plays
    )
    D = home_runs

    return ((A * B) / (B + C)) + D


def adjusted_bsr_per_game(df, side):
    df = df.copy()
    df["raw_bsr"] = raw_bsr(df, side)

    # Compare teams from the same league at the same point in the season.
    group_columns = ["Season", "Cutoff_Date", "League"]

    groups = df.groupby(group_columns)
    league_raw_bsr = groups["raw_bsr"].transform("sum")
    league_actual_runs = groups[f"runs_{side}"].transform("sum")
    adjustment = league_actual_runs / league_raw_bsr

    return df["raw_bsr"] * adjustment / df["G"]


def build_metrics(source_df):
    df = source_df.copy()
    df["League"] = df["Team_ID"].map(get_league)

    keys = ["Season", "Cutoff_Date", "Team_ID"]
    if df.duplicated(keys).any():
        raise ValueError("There is more than one row for the same team and cutoff")

    teams_per_cutoff = df.groupby(["Season", "Cutoff_Date"]).size()
    if not (teams_per_cutoff == 30).all():
        raise ValueError("Each cutoff needs all 30 teams for the BaseRuns calculation")

    # Start with team information, then add the calculated features.
    columns = ["Season", "Team_ID", "League", "G", "Team", "Cutoff_Date", "W", "L"]
    metrics = df[columns].copy()

    # Estimated scoring and run prevention using the existing BaseRuns formula.
    metrics["off_xbsr_per_game"] = adjusted_bsr_per_game(df, "bat")
    metrics["allowed_xbsr_per_game"] = adjusted_bsr_per_game(df, "pitch")
    metrics["xrun_diff_per_game"] = (
        metrics["off_xbsr_per_game"] - metrics["allowed_xbsr_per_game"]
    )

    # Actual results through the cutoff date.
    metrics["win_pct"] = df["W"] / (df["W"] + df["L"])
    metrics["runs_scored_per_game"] = df["runs_bat"] / df["G"]
    metrics["runs_allowed_per_game"] = df["runs_pitch"] / df["G"]
    metrics["run_diff_per_game"] = (
        metrics["runs_scored_per_game"] - metrics["runs_allowed_per_game"]
    )

    # Batting rates: divide each count by plate appearances.
    metrics["bat_k_rate"] = df["strikeOuts_bat"] / df["plateAppearances"]
    metrics["bat_bb_rate"] = df["baseOnBalls_bat"] / df["plateAppearances"]
    metrics["bat_hr_rate"] = df["homeRuns_bat"] / df["plateAppearances"]

    # Pitching rates: divide each count by batters faced.
    metrics["pitch_k_rate"] = df["strikeOuts_pitch"] / df["battersFaced"]
    metrics["pitch_bb_rate"] = df["baseOnBalls_pitch"] / df["battersFaced"]
    metrics["pitch_hr_rate"] = df["homeRuns_pitch"] / df["battersFaced"]
    metrics["pitch_k_minus_bb_rate"] = metrics["pitch_k_rate"] - metrics["pitch_bb_rate"]

    # Isolated power (ISO): extra bases beyond singles per at-bat.
    metrics["bat_iso"] = (df["totalBases_bat"] - df["hits_bat"]) / df["atBats_bat"]
    metrics["pitch_iso"] = (df["totalBases_pitch"] - df["hits_pitch"]) / df["atBats_pitch"]

    # Stop if missing data or division by zero produced an invalid feature.
    metrics = metrics.replace([float("inf"), float("-inf")], float("nan"))
    if metrics.isna().any().any():
        raise ValueError("Some features are missing. Check the raw counts and denominators.")

    return metrics.sort_values(keys).reset_index(drop=True)


def main():
    raw_data = pd.read_parquet(DATA_DIR / "team_snapshots.parquet")
    metrics = build_metrics(raw_data)
    metrics.to_parquet(DATA_DIR / "model_metrics.parquet", index=False)
    print(f"Saved {len(metrics)} feature rows")


if __name__ == "__main__":
    main()
