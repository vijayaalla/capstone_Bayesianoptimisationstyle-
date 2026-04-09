# Model Card for the BBO Optimisation Approach

## Overview

- **Model name:** Transparency-First Sequential BBO Pipeline
- **Type:** Surrogate-assisted black-box optimisation policy
- **Version:** `v1.0` for the Round 10 repository snapshot
- **Primary implementation:** `src/generate_week10_queries_interpretable.py`

This project does not rely on one immutable model. Instead, it uses an evolving optimisation pipeline that selects one next query point per function from sparse observations. The Round 10 version is the most transparent and reproducible form of that pipeline. It combines:

- tuned surrogate-family selection between SVR and MLP candidates
- bootstrap ensembles for uncertainty estimation
- an explicit scoring rule for candidate ranking
- deterministic selection of the highest-scoring candidate
- local sensitivity analysis to explain which dimensions influenced the chosen point

The final Round 10 score is:

`score = 0.55 * predicted_value + 0.30 * uncertainty + 0.15 * novelty`

## Intended Use

This optimisation approach is suitable for:

- choosing one next query per round for each unknown function in the BBO capstone
- studying exploration versus exploitation under tight evaluation budgets
- comparing small-data surrogate strategies across functions of different dimensionality
- documenting decision logic in a way another researcher can inspect and rerun

This approach should be avoided for:

- safety-critical or regulated optimisation tasks
- problems with hard real-world constraints that are not encoded in the candidate generator
- claims of guaranteed global optimality
- situations where uncertainty estimates must be formally calibrated

## Inputs and Outputs

- **Input:** a function-specific observation set `X, y` where `X` lies in `[0, 1]^d` and `y` contains scalar evaluations.
- **Output:** one proposed next query point in portal format, plus an optional report containing model family, cross-validated error, decision mode, and influential dimensions.

## Approach Details

### How the strategy evolved across ten rounds

| Round | Main approach | Why it was introduced |
|---|---|---|
| 1 | Manual exploratory heuristics | Build initial intuition about each unknown function |
| 2 | Gaussian Process + UCB-style acquisition | Add principled exploration/exploitation from a classic BO baseline |
| 3 | Bootstrap SVR ensemble | Improve robustness on small nonlinear datasets |
| 4 | MLP ensemble | Test whether a neural surrogate could capture more complex structure |
| 5 | Deep ensemble + novelty bonus | Strengthen uncertainty estimates and reduce duplicate sampling |
| 6 | HEBO-inspired hybrid | Blend warping, GP structure, and multiple acquisition signals |
| 7 | Tuned surrogate selection | Replace fixed defaults with evidence-based model-family choice |
| 8 | LLM-inspired structured scoring | Treat selection as a context-limited decision process rather than one raw acquisition value |
| 9 | Scale/emergence comparison | Compare candidate preferences under different context scales |
| 10 | Transparency-first deterministic scoring | Make the final decision rule easier to explain, audit, and reproduce |

### Final Round 10 decision process

For each function, the final pipeline:

1. tunes candidate SVR and MLP surrogates using cross-validated MAE
2. chooses the better-performing family for that function
3. fits a bootstrap ensemble to estimate mean prediction and disagreement
4. samples a candidate pool
5. scores each candidate using normalized predicted value, uncertainty, and novelty
6. removes near-duplicate candidates with a minimum-distance rule
7. selects the top remaining candidate deterministically
8. computes local sensitivity estimates to identify influential dimensions

The Round 10 approach also adapts behavior by function pattern:

- Functions 1 to 3 were treated as more boundary-sensitive, so stronger exploitation near edges remained reasonable.
- Function 4 was treated as a local-refinement problem inside a structured basin.
- Function 5 appeared dominated by a few strong dimensions, so interpretability around those dimensions mattered.
- Functions 6 to 8 remained more uncertainty-sensitive because dimensionality made coverage thinner.

## Performance

### Metrics used

The most reproducible metrics available in this public repository are:

- cross-validated mean absolute error (`cv_mae`) for surrogate-family selection
- normalized CV error in earlier tuned rounds
- ensemble disagreement or uncertainty terms
- Round 10 score decomposition (`pred`, `unc`, `nov`)
- local sensitivity rankings for the chosen query

The metric I would ideally also report is best observed objective value after each portal round. However, the full returned evaluation history was not committed into this public snapshot, so end-to-end optimisation performance is only partially reproducible here.

### Round 10 summary across the eight functions

| Function | Dim | Chosen family | `cv_mae` | Mode | Most influential dimensions |
|---|---:|---|---:|---|---|
| 1 | 2 | SVR | 0.000818 | balanced | `x2`, `x1` |
| 2 | 2 | SVR | 0.204484 | balanced | `x1`, `x2` |
| 3 | 3 | SVR | 0.049168 | exploit | `x3`, `x2`, `x1` |
| 4 | 4 | SVR | 1.230399 | exploit | `x4`, `x3`, `x1` |
| 5 | 4 | MLP | 94.527675 | balanced | `x3`, `x4`, `x2` |
| 6 | 5 | MLP | 0.203024 | explore | `x2`, `x4`, `x3` |
| 7 | 6 | SVR | 0.164712 | balanced | `x6`, `x5`, `x2` |
| 8 | 8 | MLP | 0.163342 | balanced | `x4`, `x3`, `x1` |

### What these results suggest

- SVR remained the preferred family for 5 of the 8 functions, especially in lower-dimensional settings.
- MLP was preferred for Functions 5, 6, and 8, which suggests some higher-complexity regimes benefited from a more flexible surrogate.
- Function 5 remained the hardest case in the reproducible metrics, with a very large `cv_mae`; this likely reflects both scale sensitivity and a more difficult landscape.
- Functions 1 and 3 were the cleanest low-error cases, which matches the intuition that lower-dimensional, boundary-driven structure was easier to model.
- Most final decisions were classified as `balanced`, with only one clearly `explore` case and two `exploit` cases.

Raw objective values are not directly comparable across functions because the output scales differ substantially. For example, the committed baseline data ranges from values near zero in Function 1 to values above one thousand in Function 5.

## Assumptions and Limitations

### Core assumptions

This approach assumes that:

- local surrogate structure is informative enough to guide the next useful query
- cross-validated surrogate fit is a reasonable proxy for next-step decision quality
- ensemble spread is a useful uncertainty signal even without perfect probabilistic calibration
- novelty helps reduce wasteful resampling in already-covered regions
- a single candidate pool is rich enough to include good options for each function

### Main limitations and failure modes

- Sparse data can make both SVR and MLP surrogates confidently wrong.
- High-dimensional functions remain thinly covered even after multiple rounds.
- Boundary-heavy sampling can bias later decisions toward familiar regions.
- The explicit score weights improve transparency, but they are still a modeling choice rather than ground truth.
- Local sensitivity explanations describe the surrogate around the chosen point, not the true global objective.
- Because the public repository does not include the full post-round returned history, some performance claims remain narrative rather than fully auditable from committed data alone.

## Ethical Considerations

Transparency helps this project in three important ways:

- it makes the decision rule easier for another researcher to inspect and reproduce
- it exposes assumptions and failure modes that might otherwise stay hidden inside ad hoc judgment
- it creates a clearer path for adapting the method to real-world settings where domain constraints must be added explicitly

That said, transparency can also create false confidence. A clearly explained choice can still be wrong if it is built on sparse, biased, or outdated data. For real-world adaptation, this style of optimisation would need stronger logging, constraint handling, domain review, and performance monitoring than the capstone setting requires.

## Documentation Scope

This model card is intentionally written at the policy level rather than duplicating every implementation detail from the Python scripts. Adding every hyperparameter range, candidate-pool setting, and report field here would make the document longer without making it much clearer. The current structure is sufficient because it links the big-picture decision logic to the reproducible artifacts already stored in the repository:

- `src/generate_round_queries.py`
- `src/generate_week7_queries_tuned.py`
- `src/generate_week8_queries_llm_strategy.py`
- `src/generate_week9_queries_scaling_emergence.py`
- `src/generate_week10_queries_interpretable.py`
