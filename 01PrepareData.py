# 01: Download the raw data and save it as Parquet files.

from pathlib import Path
import pandas as pd
import statsapi

DATA_DIR = Path(__file__).resolve().parent / "data"
SEASONS = range(2016, 2026)
CUTOFF_MONTHS = (6, 7, 8, 9)


def get_team_stats(season, cutoff, group):
    response = statsapi.get(
        "teams_stats",
        {
            "group": group,
            "stats": "byDateRange",
            "season": season,
            "sportIds": 1,
            "gameType": "R",
            "startDate": f"{season}-01-01",
            "endDate": cutoff,
        },
    )

    rows = []
    for split in response["stats"][0]["splits"]:
        row = {
            "Season": season,
            "Cutoff_Date": cutoff,
            "Team_ID": split["team"]["id"],
            "Team": split["team"]["name"],
        }
        row.update(split["stat"])
        rows.append(row)

    if len(rows) != 30:
        raise ValueError(f"Expected 30 teams for {cutoff} {group}")

    return pd.DataFrame(rows)


def get_records(season, cutoff=None):
    standings = statsapi.standings_data(
        leagueId="103,104",
        season=season,
        standingsTypes="regularSeason",
        date=cutoff,
    )

    rows = []
    for division in standings.values():
        for team in division["teams"]:
            row = {
                "Season": season,
                "Team_ID": team["team_id"],
                "W": int(team["w"]),
                "L": int(team["l"]),
            }
            if cutoff is not None:
                row["Cutoff_Date"] = cutoff
            rows.append(row)

    if len(rows) != 30:
        raise ValueError(f"Expected 30 teams in the {season} standings")

    return pd.DataFrame(rows)


def combine_snapshot(batting, pitching, records):
    keys = ["Season", "Cutoff_Date", "Team_ID"]

    snapshot = batting.merge(
        pitching,
        on=keys,
        suffixes=("_bat", "_pitch"),
        validate="one_to_one",
    )
    snapshot = snapshot.merge(records, on=keys, validate="one_to_one")

    if len(snapshot) != 30:
        raise ValueError("Some teams are missing from the snapshot")
    if not (snapshot["gamesPlayed_bat"] == snapshot["gamesPlayed_pitch"]).all():
        raise ValueError("Batting and pitching games played do not match")

    snapshot = snapshot.rename(columns={"Team_bat": "Team", "gamesPlayed_bat": "G"})
    snapshot = snapshot.drop(columns=["Team_pitch", "gamesPlayed_pitch"])
    return snapshot


def main():
    DATA_DIR.mkdir(exist_ok=True)

    batting_tables = []
    pitching_tables = []
    record_tables = []
    snapshots = []
    final_records = []

    for season in SEASONS:
        # Final records are saved separately for the training target in 03.
        final = get_records(season)
        final = final.rename(columns={"W": "final_W", "L": "final_L"})
        final_records.append(final)

        for month in CUTOFF_MONTHS:
            # The 2020 season started late, so skip June and July.
            if season == 2020 and month < 8:
                continue

            cutoff = f"{season}-{month:02d}-01"
            batting = get_team_stats(season, cutoff, "hitting")
            pitching = get_team_stats(season, cutoff, "pitching")
            records = get_records(season, cutoff)

            batting_tables.append(batting)
            pitching_tables.append(pitching)
            record_tables.append(records)
            snapshots.append(combine_snapshot(batting, pitching, records))
            print(f"Collected {cutoff}", flush=True)

    batting = pd.concat(batting_tables, ignore_index=True)
    pitching = pd.concat(pitching_tables, ignore_index=True)
    records = pd.concat(record_tables, ignore_index=True)
    snapshots = pd.concat(snapshots, ignore_index=True)
    final_records = pd.concat(final_records, ignore_index=True)

    batting.to_parquet(DATA_DIR / "batting_snapshots.parquet", index=False)
    pitching.to_parquet(DATA_DIR / "pitching_snapshots.parquet", index=False)
    records.to_parquet(DATA_DIR / "snapshot_records.parquet", index=False)
    snapshots.to_parquet(DATA_DIR / "team_snapshots.parquet", index=False)
    final_records.to_parquet(DATA_DIR / "final_records.parquet", index=False)


if __name__ == "__main__":
    main()
