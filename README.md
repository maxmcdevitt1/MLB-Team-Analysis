# Team analysis

For a detailed beginner walkthrough, read the [full project guide](docs/PROJECT_GUIDE.md).
It explains each pipeline stage, Python and library concepts, the feature formulas,
model training, current results, checks, and next steps.

Run these files in order from the project folder:

```bash
python 01PrepareData.py
python 02CreateMetrics.py
python 03-BuildTrainingData.py
python 04TrainModel.py
```

1. **01 collects raw data.** It downloads team batting, pitching, and standings
   through each cutoff date. It also saves final season records separately.
2. **02 creates features.** It reads the saved raw data and calculates BaseRuns,
   runs per game, win percentage, strikeout/walk/home-run rates, and isolated power.
3. **03 adds the target.** It reads the saved features and final records to calculate
   the win percentage over the remaining games.
4. **04 trains and evaluates a model.** Open `04TrainModel.ipynb` in the project
   folder and run the cells from top to bottom using the project's `.venv` kernel.
   It contains the modeling code in small steps, with explanations.
   Alternatively, run `python 04TrainModel.py` for the same calculations and
   Parquet outputs, with results printed in the terminal.

All data is saved as **Parquet files** in `data/`:

| File | Contents |
| --- | --- |
| `batting_snapshots.parquet` | Raw batting statistics |
| `pitching_snapshots.parquet` | Raw pitching statistics |
| `snapshot_records.parquet` | Wins and losses at each cutoff |
| `team_snapshots.parquet` | The three raw tables joined together; input to 02 |
| `final_records.parquet` | Final wins and losses; input to 03 |
| `model_metrics.parquet` | Features created by 02 |
| `training_data.parquet` | Features and the remaining-season target |
| `validation_predictions.parquet` | Predictions for 2023–2024 from each approach |
| `validation_results.parquet` | Validation mean absolute error for each approach |
| `test_predictions.parquet` | 2025 predictions from the chosen regression and benchmarks |
| `test_results.parquet` | 2025 mean absolute error for the chosen regression and benchmarks |

To change the years or cutoff months, edit `SEASONS` and `CUTOFF_MONTHS` at the top
of 01. Use completed seasons for training. The defaults are 2016–2025 and June 1,
July 1, August 1, and September 1. The 2020 season skips June and July.

Running 01 downloads the data again and replaces its Parquet files. To change
features without downloading anything, just rerun 02 and 03.

The 01–03 notebooks run the same scripts and show a small preview of their Parquet
outputs. The 04 notebook contains the modeling steps directly. Open the notebooks
from this project folder.

04 uses July 1 snapshots only, excluding 2020. It trains on 2016–2022 and uses
2023–2024 to choose between BaseRuns regression and BaseRuns plus pitching K–BB
rate. It then refits the selected version on 2016–2024 and evaluates it on 2025,
alongside two benchmarks: always predict .500 and predict current win percentage.
Lower mean absolute error (MAE) is better; 0.050 means five percentage points.
All regression predictions are clipped to 0–1 before scoring. Every team-season
has equal weight. Once 2025 has been evaluated, do not use it to tune features and
then call it an untouched test again.

04 needs scikit-learn in addition to pandas and pyarrow. If needed, run
`%pip install scikit-learn` in the notebook's Python environment. Rerunning 04
replaces its four output Parquet files; it does not download data.

Rates are fractions: 0.20 means 20%. The BaseRuns formula is unchanged and compares
teams within the same league, season, and cutoff. `FEATURE_SETS` in 02 lists
candidate model inputs. Final records and remaining results are used for the
target, not as input features.

The older `data.parquet` and `records.parquet` files are not used by this pipeline.
# MLB-Team-Analysis
