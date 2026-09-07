# Research and verification record

Internal companion to the beginner project guide. The user-facing artifact is `docs/PROJECT_GUIDE.md`; `report-source.md` is its canonical manuscript.

Review/access date: September 6, 2026, Pacific/Honolulu. This is a review of the present local project, not an independent audit of MLB event data.

## Scope and approach

The user requested a full beginner explanation of the project, its library use, design choices, and ML concepts, with Deep Research and Plugin Management selected. Markdown in the repository was selected as the durable format so the guide can be read beside the scripts and linked from README. No external application connection or additional plugin was needed. Application code and Parquet outputs were not changed by this documentation task.

The research plan was: inspect the current implementation and artifacts; verify unfamiliar or version-sensitive behavior against primary documentation; write a complete teaching guide grounded in the code; verify the examples, counts, scores, and Markdown structure. An update_plan tool was not available, so the plan was tracked internally. Two bounded source-research delegations requested by the Deep Research skill could not finish because of usage limits. The primary agent completed the needed primary-source research directly; no unreturned research was treated as evidence.

## Local primary evidence

- Read all four current numbered Python scripts, the four notebooks, README, and the test file.
- Inspected installed statsapi endpoint metadata and standings_data implementation to check request parameters and omitted-date behavior.
- Inspected package versions using the project WSL virtual environment. Python 3.12.3, pandas 3.0.5, NumPy 2.5.2, PyArrow 25.0.1, MLB-StatsAPI 1.9.0, scikit-learn 1.9.0. Online Python docs may target a newer interpreter; the APIs discussed were also checked against the installed code/environment.
- Read all eleven current Parquet outputs. The raw snapshot, feature, and training tables have 1,140 rows, with unique Season/Cutoff_Date/Team_ID keys and no missing cells. Final records contain 300 rows.
- Loaded current 04 functions to verify July filtering and the 180/60/30 train/validation/test split. Refitted only in memory; did not invoke its output-writing main function. Predictions matched the stored selected-model test column within numerical tolerance.
- Independently recomputed all seven saved MAEs and percentage-point values from prediction files.
- Traced Cleveland, Team_ID 114, July 1, 2025 through the data. Its 40-43 cutoff record and 88-74 final record give remaining performance 48/79. The stored regression prediction is approximately 0.441789.
- Ran the existing unittest command. It fails while importing missing BuildTrainingData; its six actual test methods do not run. The guide records this accurately and labels the suggested import replacement as unapplied.

## External source ledger

All sources below are primary documentation or a primary publisher's definition. Browser-native references are internal retrieval identifiers, not public citations. The guide cites the canonical URLs near the relevant explanations. Sources without a retained native ID are still identified by their exact consulted URL. Most teaching material is original explanation of the user's code and local results; source definitions are paraphrased briefly. MLB glossary paraphrases are kept brief; the ISO algebra is independently explained from the project formula.

| Source / supported topic | Publisher | Publication or update information | Native retrieval reference |
| --- | --- | --- | --- |
| [evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html) | scikit-learn project | Undated / maintained documentation; no publication date asserted | turn8view1 |
| [IPython `%run` documentation](https://ipython.readthedocs.io/en/stable/interactive/magics.html) | IPython project | Undated / maintained documentation; no publication date asserted | turn11view1 |
| [repository](https://github.com/toddrob99/MLB-StatsAPI) | MLB-StatsAPI maintainers (community wrapper) | Undated / maintained documentation; no publication date asserted | turn8view2 |
| [Apache Parquet's overview](https://parquet.apache.org/docs/overview/) | Apache Software Foundation | Undated / maintained documentation; no publication date asserted | turn7view0 |
| [pandas `to_parquet`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html) | pandas project | Undated / maintained documentation; no publication date asserted | turn7view1 |
| [virtual environment guide](https://docs.python.org/3/tutorial/venv.html) | Python Software Foundation | Undated / maintained documentation; no publication date asserted | turn6view0 |
| [data structures tutorial](https://docs.python.org/3/tutorial/datastructures.html) | Python Software Foundation | Undated / maintained documentation; no publication date asserted | turn6view2 |
| [Python `pathlib`](https://docs.python.org/3/library/pathlib.html) | Python Software Foundation | Undated / maintained documentation; no publication date asserted | turn6view1 |
| [endpoint reference](https://github.com/toddrob99/MLB-StatsAPI/wiki/Endpoints) | MLB-StatsAPI maintainers (community wrapper) | 2025-03-27 (wiki revision) | turn12view0 |
| [`standings_data` documentation](https://github.com/toddrob99/MLB-StatsAPI/wiki/Function:-standings_data) | MLB-StatsAPI maintainers (community wrapper) | Undated / maintained documentation; no publication date asserted | turn8view3 |
| [documented pandas merge behaviors](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html) | pandas project | Undated / maintained documentation; no publication date asserted | turn9view0 |
| [pandas `concat`](https://pandas.pydata.org/docs/reference/api/pandas.concat.html) | pandas project | Undated / maintained documentation; no publication date asserted | Earlier batched primary-documentation read; URL is canonical |
| [plate appearance](https://www.mlb.com/glossary/standard-stats/plate-appearance) | MLB | Undated / maintained documentation; no publication date asserted | turn5search2 |
| [at-bat](https://www.mlb.com/glossary/standard-stats/at-bat) | MLB | Undated / maintained documentation; no publication date asserted | turn5search4 |
| [MLB's ISO definition](https://www.mlb.com/glossary/advanced-stats/isolated-power) | MLB | Undated / maintained documentation; no publication date asserted | turn5search1 |
| [pandas `Series.map`](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html) | pandas project | Undated / maintained documentation; no publication date asserted | turn10view1 |
| [FanGraphs BaseRuns](https://library.fangraphs.com/features/baseruns/) | FanGraphs / Neil Weinberg | 2016-08-08 | turn5search0 |
| [pandas group transformations](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.transform.html) | pandas project | Undated / maintained documentation; no publication date asserted | Earlier batched primary-documentation read; URL is canonical |
| [common pitfalls guide](https://scikit-learn.org/stable/common_pitfalls.html) | scikit-learn project | Undated / maintained documentation; no publication date asserted | turn8view0 |
| [pandas datetime conversion](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html) | pandas project | Undated / maintained documentation; no publication date asserted | turn10view2 |
| [the official estimator reference](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) | scikit-learn project | Undated / maintained documentation; no publication date asserted | turn7view2 |
| [scikit-learn's ordinary-least-squares discussion](https://scikit-learn.org/stable/modules/linear_model.html) | scikit-learn project | Undated / maintained documentation; no publication date asserted | turn11view2 |
| [NumPy's clipping operation](https://numpy.org/doc/stable/reference/generated/numpy.clip.html) | NumPy project | Undated / maintained documentation; no publication date asserted | turn10view3 |
| [scikit-learn MAE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) | scikit-learn project | Undated / maintained documentation; no publication date asserted | turn7view3 |
| [pandas' introductory guide](https://pandas.pydata.org/docs/user_guide/10min.html) | pandas project | Undated / maintained documentation; no publication date asserted | turn10view0 |
| [pandas indexing and selection](https://pandas.pydata.org/docs/user_guide/indexing.html) | pandas project | Undated / maintained documentation; no publication date asserted | Earlier batched primary-documentation read; URL is canonical |
| [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html) | pandas project | Undated / maintained documentation; no publication date asserted | turn12view1 |
| [`unittest` guide](https://docs.python.org/3/library/unittest.html) | Python Software Foundation | Undated / maintained documentation; no publication date asserted | turn11view0 |
| [assert statement reference](https://docs.python.org/3/reference/simple_stmts.html) | Python Software Foundation | Undated / maintained documentation; no publication date asserted | turn6view3 |

## Evidence and gap matrix

| Question | Evidence | Status / boundary |
| --- | --- | --- |
| What problem is predicted? | 03 target arithmetic and 04 filters | Resolved: remaining-season win fraction, July cutoff. |
| What does each file do? | Current code, functions, notebook cells | Resolved; notebook/script duplication is documented. |
| What are the libraries doing? | Local imports/calls plus official references | Resolved for APIs used in this project. |
| Why these features? | Exact local formulas, MLB definitions, FanGraphs formula | Resolved as code explanation; predictive benefit is judged separately. |
| Is the split and scoring description correct? | Function inspection and independent in-memory recomputation | Resolved. |
| How strong is the result? | Saved validation/test comparisons | Small observed improvement; one season does not establish reliable future superiority. |
| Why did the original author choose every setting? | No complete decision history | Rationales explicitly identified as practical interpretations where intent is not recorded. |
| Is source data exactly what was knowable at each historic cutoff? | Present historical API outputs, no vintage archive | Unresolved, disclosed; no claim of audited point-in-time reconstruction. |
| Do the unit tests currently pass? | Actual unittest invocation | No: stale module import prevents collection. |
| Does K-BB specifically improve the 2025 test? | Only selected regression tested | Not established; its incremental comparison occurred on validation. |

## Research stopping decision

The local implementation answered project-specific questions, primary references covered the library and baseball definitions used, and numerical claims were reproduced. Further broad web research would not resolve the absent vintage source archive or establish author intent. Those boundaries are disclosed rather than filled with speculation. No additional predictive experiments were run to tune against the test season.

## Artifact verification

The manuscript has 17 teaching sections and a linked contents list. All 26 Python code blocks parse; illustrative fragments are explicitly distinguished from a full runnable program. The complete read-only MAE snippet was executed successfully. Heading anchors resolve, code fences balance, and every function in the four scripts is mentioned. Current model coefficients and all published rounded scores were checked. The public artifact is copied byte-for-byte from the canonical manuscript. README links to it. Markdown was structurally reviewed; no claim of PDF or Word pagination/rendering verification is made.

The separate local-evidence.json records file hashes and Parquet shapes at handoff so later edits can be distinguished from this reviewed snapshot.
