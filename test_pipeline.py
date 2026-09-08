"""Offline checks of the collection/feature/target boundaries."""

from importlib import import_module
import unittest

import numpy as np
import pandas as pd

features = import_module("02CreateMetrics")
build_training_data = import_module("03-BuildTrainingData").build_training_data

def snapshot():
    rows = []
    for team in features.AL_TEAM_IDS + features.NL_TEAM_IDS:
        row = dict(Season=2025, Cutoff_Date="2025-06-01", Team_ID=team,
                   Team=str(team), G=10, W=6, L=4, plateAppearances=400, battersFaced=400)
        for side in ("bat", "pitch"):
            counts = dict(hits=90, baseOnBalls=40, hitByPitch=3, intentionalWalks=2,
                          homeRuns=10, totalBases=140, stolenBases=7, caughtStealing=2,
                          groundIntoDoublePlay=5, sacFlies=2, sacBunts=1, runs=45,
                          strikeOuts=80, atBats=350)
            row.update({f"{name}_{side}": value for name, value in counts.items()})
        rows.append(row)
    return pd.DataFrame(rows)


class PipelineTests(unittest.TestCase):
    def test_known_rates_and_league_calibration(self):
        result = features.build_metrics(snapshot())
        for name, expected in {"win_pct": .6, "bat_hr_rate": .025,
                               "pitch_k_minus_bb_rate": .1, "bat_iso": 50 / 350,
                               "off_xbsr_per_game": 4.5, "allowed_xbsr_per_game": 4.5}.items():
            np.testing.assert_allclose(result[name], expected)

    def test_later_cutoffs_cannot_change_earlier_features(self):
        raw = snapshot()
        later = raw.copy()
        later["Cutoff_Date"] = "2025-07-01"
        later["runs_bat"] *= 3
        combined = features.build_metrics(pd.concat([raw, later], ignore_index=True))
        pd.testing.assert_frame_equal(features.build_metrics(raw),
                                      combined.loc[combined.Cutoff_Date == "2025-06-01"].reset_index(drop=True))

    def test_final_records_never_become_features(self):
        raw = snapshot()
        raw["final_W"] = 100
        raw["remaining_win_pct"] = .9
        pd.testing.assert_frame_equal(features.build_metrics(raw), features.build_metrics(snapshot()))

    def test_missing_duplicate_and_zero_inputs_fail(self):
        raw = snapshot()
        invalid = [raw.iloc[:-1], raw.loc[raw.Team_ID.isin(features.AL_TEAM_IDS)],
                   pd.concat([raw, raw.iloc[:1]]), raw.drop(columns="hits_bat"),
                   raw.assign(plateAppearances=0), raw.assign(hits_bat=np.nan)]
        for frame in invalid:
            with self.subTest(shape=frame.shape), self.assertRaises((ValueError, KeyError)):
                features.build_metrics(frame)

    def test_target_and_no_remaining_games(self):
        metrics = features.build_metrics(snapshot())
        final = metrics[["Season", "Team_ID"]].assign(final_W=96, final_L=66)
        result = build_training_data(metrics, final)
        np.testing.assert_allclose(result.remaining_win_pct, 90 / 152)
        self.assertTrue(build_training_data(metrics, final.assign(final_W=6, final_L=4)).empty)

    def test_missing_duplicate_and_invalid_targets_fail(self):
        metrics = features.build_metrics(snapshot())
        final = metrics[["Season", "Team_ID"]].assign(final_W=96, final_L=66)
        for frame in [final.iloc[:-1], pd.concat([final, final.iloc[:1]]),
                      final.assign(final_W=5)]:
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                build_training_data(metrics, frame)


if __name__ == "__main__":
    unittest.main()
