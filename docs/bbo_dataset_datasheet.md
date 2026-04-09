# Datasheet for the BBO Capstone Dataset

## Dataset Summary

This datasheet documents the dataset used in the Bayesian black-box optimisation (BBO) capstone project. In practice, the project dataset is the combination of:

- the committed `initial_data/` arrays that provide the starting observations for eight unknown functions
- the round-by-round query artifacts in `src/` that record how new candidate points were generated and submitted

The current public repository snapshot therefore preserves the baseline function evaluations plus substantial provenance about later query choices, but it does not contain a complete machine-readable archive of every post-submission function evaluation returned by the course portal.

## Motivation

I created and maintained this dataset to support a sequential optimisation task: choose one new query per function per round in order to maximise eight unknown black-box functions under a limited evaluation budget. The dataset was also intended to support:

- reproducible comparison of surrogate-model strategies
- reflection on exploration versus exploitation
- documentation of assumptions, uncertainty, and interpretability decisions
- evidence-based discussion posts and capstone deliverables

## Composition

### Machine-readable data in the public repository

The repository currently contains eight function-specific subsets in `initial_data/`. Each subset stores one input array and one output array:

- `initial_inputs.npy`: floating-point input matrix with values in `[0, 1]`
- `initial_outputs.npy`: floating-point vector of scalar function evaluations

The committed baseline contains 175 observations in total.

| Function | Dimensionality | Observations in repo snapshot | Files |
|---|---:|---:|---|
| `function_1` | 2 | 10 | `initial_data/function_1/initial_inputs.npy`, `initial_data/function_1/initial_outputs.npy` |
| `function_2` | 2 | 10 | `initial_data/function_2/initial_inputs.npy`, `initial_data/function_2/initial_outputs.npy` |
| `function_3` | 3 | 15 | `initial_data/function_3/initial_inputs.npy`, `initial_data/function_3/initial_outputs.npy` |
| `function_4` | 4 | 30 | `initial_data/function_4/initial_inputs.npy`, `initial_data/function_4/initial_outputs.npy` |
| `function_5` | 4 | 20 | `initial_data/function_5/initial_inputs.npy`, `initial_data/function_5/initial_outputs.npy` |
| `function_6` | 5 | 20 | `initial_data/function_6/initial_inputs.npy`, `initial_data/function_6/initial_outputs.npy` |
| `function_7` | 6 | 30 | `initial_data/function_7/initial_inputs.npy`, `initial_data/function_7/initial_outputs.npy` |
| `function_8` | 8 | 40 | `initial_data/function_8/initial_inputs.npy`, `initial_data/function_8/initial_outputs.npy` |

### Provenance and query-history artifacts

The `src/` directory also contains round-specific assets that document how the dataset was used and extended during the capstone, including:

- generator scripts such as `generate_round_queries.py` and `generate_week10_queries_interpretable.py`
- portal-ready query files such as `module14_week3_queries.txt` through `module21_week10_queries.txt`
- submission summaries and reflection files
- diagnostic reports such as tuning, prompt, scaling, and interpretability reports

These files are important for reproducibility because they show the strategy used to create new candidate queries even when the returned post-round outputs were not committed back into `initial_data/` in this public snapshot.

### Gaps and known omissions

The most important dataset gaps are:

- the public repository keeps the original baseline arrays rather than a fully updated post-round evaluation history
- rounds 1 and 2 are described in narrative form, but not preserved as standalone machine-readable query files
- there is no separate metadata table with timestamps, portal submission IDs, or evaluator-side notes
- the dataset is intentionally uneven across functions because the problem itself spans different dimensions and initial sample sizes

## Collection Process

### How the data was obtained

The initial arrays were provided as part of the capstone setup. After that starting point, new queries were generated sequentially: one query per function per round, submitted through the course portal, then intended to be appended locally to the relevant function arrays before the next round.

### Query-generation strategy over time

The collection process was not passive. It was shaped by the optimisation strategy used in each round:

- Round 1: simple exploratory heuristics and manual inspection
- Round 2: Gaussian Process surrogate with an upper-confidence style acquisition rule
- Round 3: bootstrap SVR ensemble with uncertainty-aware acquisition
- Round 4: MLP ensemble with bounded hyperparameter randomisation
- Round 5: deeper neural ensemble with novelty-aware scoring and local refinement
- Round 6: HEBO-inspired hybrid pipeline with input warping and blended acquisition terms
- Round 7: tuned surrogate selection using cross-validated MAE
- Round 8: LLM-inspired structured scoring with context and decoding controls
- Round 9: scale-aware comparison across small, medium, and full-context model views
- Round 10: transparency-first deterministic scoring using predicted value, uncertainty, and novelty, plus local sensitivity reporting

### Time frame

The collection process spans ten sequential capstone rounds documented across the repository modules, especially Modules 13 to 21. In practical terms, data was accumulated over repeated weekly-style submission cycles rather than through a single batch collection event.

## Preprocessing and Transformations

The stored dataset itself is lightly processed:

- inputs are already normalized to the unit hypercube `[0, 1]^d`
- outputs are stored as raw scalar values
- query strings written for the portal are formatted to six decimal places and joined with hyphens

Most heavier transformations happen inside the query-generation scripts at runtime rather than being saved back into the dataset files. Depending on the round, these transformations included:

- standardizing or normalizing targets before surrogate fitting
- bootstrap resampling for ensemble uncertainty estimates
- input warping or context-window selection
- novelty and minimum-distance filters to avoid near-duplicate points
- fixed random seeds so later rounds were more reproducible

Because those transformations are model-side rather than dataset-side, the committed arrays remain close to the original observed values.

## Intended Uses

This dataset is intended for:

- sequential black-box optimisation experiments in an educational setting
- comparing surrogate-model and acquisition-rule choices under sparse data
- tracing how optimisation strategy evolves as more observations become available
- discussing transparency, assumptions, and reproducibility in ML workflows

## Inappropriate Uses

This dataset should not be used for:

- benchmarking real-world production optimisers without additional validation
- fairness, bias, or human-impact studies, because the functions are synthetic and task-specific
- safety-critical optimisation where calibrated uncertainty and hard constraints are required
- claiming full reproducibility of post-round performance from this public snapshot alone, because not all returned portal evaluations are committed here

## Distribution

The dataset is distributed through this GitHub repository as a combination of:

- `initial_data/` for the committed NumPy arrays
- `src/` for query history artifacts, scripts, and reports
- `docs/` for supporting documentation such as this datasheet and the model card

The repository is intended to be public for course discussion and peer feedback.

## Terms of Use

No separate license file is present in the current repository snapshot. That means reuse terms are not explicitly granted beyond normal access to a public GitHub repository and any course-related expectations attached to the capstone. Until a license is added, the safest interpretation is that the material should be treated as educational/reference material rather than freely reusable project assets.

## Maintenance

The repository owner maintains the dataset and associated documentation. The expected maintenance workflow is:

- append new query and returned output pairs to the relevant `initial_data/function_*/` arrays after each portal round
- keep the round-specific query, submission, and report files in `src/`
- update the README, datasheet, and model card when the optimisation strategy changes materially

One important maintenance note is that this public snapshot is not yet fully synchronized with the ideal post-round local data state. That limitation is documented here so future readers can distinguish between the committed baseline arrays and the fuller local workflow used during the capstone.

## Risks and Biases

This dataset has no direct personal-data or privacy risk because it is synthetic and course-based. The main risks are analytical:

- uneven sampling coverage can make some regions look more important than they really are
- boundary-heavy querying in some functions can bias later models toward familiar regions
- higher-dimensional functions remain sparse, so model confidence may exceed the evidence available
- interpretability reports can make the process look more certain than it actually is if they are read without the underlying uncertainty context
