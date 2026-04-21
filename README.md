# BBO Capstone Project

## Section 1: Project Overview

This capstone project simulates a Bayesian black-box optimisation (BBO) challenge across eight unknown functions.  
For each function, I only see queried inputs and returned outputs. I do not know the underlying equation, gradient, or shape in advance.

The overall goal is to maximise each function with a limited number of sequential queries. This mirrors real-world ML settings where experiments are expensive, delayed, or noisy (for example hyperparameter tuning, scientific testing, or operational optimisation).

Career relevance: this project develops practical skills in uncertainty-aware decision-making, model selection under limited data, iterative experimentation, and communicating technical trade-offs clearly.

Core documentation for the Module 21 activity:
- [BBO Dataset Datasheet](docs/bbo_dataset_datasheet.md)
- [BBO Strategy Model Card](docs/bbo_model_card.md)

## Section 2: Inputs and Outputs

Each function receives an input vector `x` where each component is in `[0, 1]`, and returns one scalar output `y`.

- Query format for portal submission: `x1-x2-...-xn`
- Each `xi` is written with six decimals and starts with `0` (for example `0.123456-0.654321`)
- One query per function per round

Initial data summary:

| Function | Input shape | Output shape |
|---|---|---|
| 1 | `(10, 2)` | `(10,)` |
| 2 | `(10, 2)` | `(10,)` |
| 3 | `(15, 3)` | `(15,)` |
| 4 | `(30, 4)` | `(30,)` |
| 5 | `(20, 4)` | `(20,)` |
| 6 | `(20, 5)` | `(20,)` |
| 7 | `(30, 6)` | `(30,)` |
| 8 | `(40, 8)` | `(40,)` |

Note:
- This table reflects the repository baseline arrays currently checked into `initial_data/`.
- After each portal round, local arrays should be updated before generating the next query set.

## Section 3: Challenge Objectives

The objective is to **maximise** all eight functions while working under strict constraints:

- unknown black-box function structure
- limited query budget (one new point per function per round)
- delayed feedback after submission
- increasing dimensionality (2D to 8D)
- possible noise, non-linearity, and local optima

Success is not only highest observed output; it is also a strong evidence-based process: choosing reasonable points, learning from outcomes, and updating strategy over time.

## Section 4: Technical Approach (Rounds 1-12)

This is a living strategy log and is updated each round.

### Round 1
- Baseline exploratory search using simple heuristics and manual inspection.
- Goal: build intuition about response ranges and promising regions.

### Round 2
- Shifted to model-guided selection with a Gaussian Process surrogate.
- Used an upper-confidence style acquisition to balance predicted value (exploitation) and uncertainty (exploration).

### Round 3
- Introduced an SVM-based approach (SVR ensemble with RBF kernels).
- Built multiple SVR models on bootstrap samples and used:
  `acquisition = mean_prediction + beta * prediction_std`
- Added a distance rule to avoid querying points too close to existing observations.

### Round 4
- Added a neural-network surrogate strategy (MLP regressor ensemble).
- Trained multiple bootstrap models with bounded hyperparameter randomisation (network width, `alpha`, learning rate).
- Selected queries using:
  `acquisition = mean_prediction + beta * prediction_std`
- Used early stopping and regularisation to reduce overfitting risk as data volume grows.

### Round 5
- Added a deep-ensemble style neural strategy with architecture diversity (2-3 hidden layers, varied widths, mixed activations).
- Used a two-stage candidate search:
  1) global random candidate scan
  2) local refinement around top-ranked candidates
- Updated acquisition to:
  `acquisition = mean_prediction + beta * prediction_std + novelty_bonus`
- Kept distance-based filtering to reduce duplicate/local-overlap queries.

### Round 6
- Shifted to a HEBO-inspired hybrid BO pipeline:
  1) input warping
  2) GP surrogate on warped space
  3) blended acquisition from `EI + PI + UCB + uncertainty`
- Added adaptive weighting to increase uncertainty focus when data is sparse.
- Kept novelty and minimum-distance constraints to improve coverage quality.

### Round 7
- Added explicit hyperparameter tuning as a first-class step before query generation.
- Per-function random-search tuning across SVR/MLP candidates using CV MAE.
- Selected best model family per function, then built bootstrap ensembles for uncertainty-aware acquisition.
- Used tuning diagnostics to adjust exploration weight and detect overfitting risk.

### Round 8
- Added an LLM-inspired decision layer on top of tuned surrogates.
- Combined full-context and finite-context model views to mimic attention limits.
- Used structured scoring instead of one raw acquisition value.
- Added constrained decoding controls (`temperature`, `top_k`, `top_p`) and a context budget (`max_tokens`).
- Applied light boundary penalties and strict formatting checks to reduce edge-case output artefacts.

### Round 9
- Added scaling-aware comparison across short-, medium-, and full-context model views.
- Explicitly tracked emergence signals when larger context changed candidate preference.
- Balanced stable scale agreement against disagreement and uncertainty.
- Kept constrained decoding and novelty checks so emergent behavior could inform choices without dominating them.

### Round 10
- Shifted to a transparency-first decision rule for query selection.
- Used an explicit score decomposition:
  `score = 0.55 * predicted_value + 0.30 * uncertainty + 0.15 * novelty`
- Chose candidates deterministically for easier reproducibility.
- Added local sensitivity estimates to explain which input dimensions most influenced each chosen query.

### Round 11
- Reframed the search space through a clustering lens.
- Clustered observed points to identify local groups, centroid trends, and gaps between neighboring regions.
- Ranked clusters using local quality, surrogate predictions, uncertainty, and inter-cluster separation.
- Added three distance cues to guide the final query:
  `centroid_trend`, `boundary_tightening`, and `bridge_probe`
- Selected candidates by combining prediction, uncertainty, novelty, and alignment with the chosen local cluster structure.

### Round 12
- Reframed the observed query history as a high-dimensional dataset and used PCA-style reasoning to guide the next move.
- Identified dominant variation directions before candidate generation.
- Added PCA-aligned local perturbations around strong anchors to reduce randomness.
- Scored candidates with prediction, uncertainty, novelty, and PCA alignment.

### Exploration vs Exploitation Policy
- Low dimensions: slightly more exploitation around promising areas.
- Higher dimensions: stronger exploration due to sparse coverage.
- Hyperparameters are tuned within bounded ranges (not fixed defaults), then filtered with practical heuristics for stability.

### Why this approach
- Combines interpretability, practical robustness, and iterative learning.
- Adapts method complexity as data grows (heuristics -> GP -> SVM ensemble).
- Supports real-world thinking: make the best next decision with incomplete knowledge.

## Section 5: Repository Architecture

The repository follows a modular pipeline architecture:
- `initial_data/` stores current observations for each function.
- `src/generate_*.py` scripts implement per-round surrogate + acquisition logic.
- `src/module*` files store submission-ready query strings, reflections, and discussion drafts.

Detailed architecture decision and rationale are documented in:
- `src/module16_component16_2_software_architecture.md`

## Section 6: Technical Grounding

Module 17 adds explicit technical justification linking this repository to:
- academic BO research (GP surrogates, EI/PI/UCB, input warping)
- state-of-the-art design patterns (hybrid acquisition and adaptive exploration)
- third-party package choices (`numpy`, `scikit-learn`, `scipy`)

See:
- `src/module17_component17_2_technical_justification.md`

## Section 7: Module 17 Deliverables

Current Module 17 artifacts:
- `src/generate_week6_queries_hebo_hybrid.py`
- `src/module17_week6_queries.txt`
- `src/module17_week6_submission.txt`
- `src/module17_discussion_post.md` (Component 17.1 discussion-ready response)
- `src/module17_component17_1_reflection.md`
- `src/module17_component17_2_technical_justification.md`
- `src/module17_2_discussion_post.md` (Component 17.2 discussion-ready response)

## Section 8: Module 18 Deliverables

Current Module 18 artifacts:
- `src/generate_week7_queries_tuned.py`
- `src/module18_week7_queries.txt`
- `src/module18_week7_tuning_report.txt`
- `src/module18_week7_submission.txt`
- `src/module18_component18_1_reflection.md`
- `src/module18_discussion_post.md`

## Section 9: Module 19 Deliverables

Current Module 19 artifacts:
- `src/generate_week8_queries_llm_strategy.py`
- `src/module19_week8_queries.txt`
- `src/module19_week8_prompt_report.txt`
- `src/module19_week8_submission.txt`
- `src/module19_component19_1_reflection.md`
- `src/module19_discussion_post.md`

## Section 10: Module 20 Deliverables

Current Module 20 artifacts:
- `src/generate_week9_queries_scaling_emergence.py`
- `src/module20_week9_queries.txt`
- `src/module20_week9_scaling_report.txt`
- `src/module20_week9_submission.txt`
- `src/module20_component20_1_reflection.md`
- `src/module20_discussion_post.md`

## Section 11: Module 21 Deliverables

Current Module 21 artifacts:
- `src/generate_week10_queries_interpretable.py`
- `src/module21_week10_queries.txt`
- `src/module21_week10_interpretability_report.txt`
- `src/module21_week10_submission.txt`
- `src/module21_component21_1_reflection.md`
- `src/module21_discussion_post.md`
- [docs/bbo_dataset_datasheet.md](docs/bbo_dataset_datasheet.md)
- [docs/bbo_model_card.md](docs/bbo_model_card.md)

## Section 12: Module 22 Deliverables

Current Module 22 artifacts:
- `src/generate_week11_queries_clustering.py`
- `src/module22_week11_queries.txt`
- `src/module22_week11_cluster_report.txt`
- `src/module22_week11_submission.txt`
- `src/module22_discussion_post.md`

## Section 13: Module 23 Deliverables

Current Module 23 artifacts:
- `src/generate_week12_queries_pca.py`
- `src/module23_week12_queries.txt`
- `src/module23_week12_pca_report.txt`
- `src/module23_week12_submission.txt`
- `src/module23_component23_1_reflection.md`
- `src/module23_discussion_post.md`

## Section 14: Reproducible Commands

Query generation commands:
- Round 3 (SVM):  
  `python src/generate_week3_queries_svm.py --data-dir initial_data --output src/module14_week3_queries.txt`
- Round 4 (NN ensemble):  
  `python src/generate_week4_queries_nn.py --data-dir initial_data --output src/module15_week4_queries.txt`
- Round 5 (deep ensemble):  
  `python src/generate_week5_queries_deep_ensemble.py --data-dir initial_data --output src/module16_week5_queries.txt`
- Round 6 (HEBO-inspired hybrid):  
  `python src/generate_week6_queries_hebo_hybrid.py --data-dir initial_data --output src/module17_week6_queries.txt`
- Round 7 (tuned surrogates):  
  `python src/generate_week7_queries_tuned.py --data-dir initial_data --output src/module18_week7_queries.txt --report src/module18_week7_tuning_report.txt`
- Round 8 (LLM-inspired strategy):  
  `python src/generate_week8_queries_llm_strategy.py --data-dir initial_data --output src/module19_week8_queries.txt --report src/module19_week8_prompt_report.txt`
- Round 9 (scaling/emergence strategy):  
  `python src/generate_week9_queries_scaling_emergence.py --data-dir initial_data --output src/module20_week9_queries.txt --report src/module20_week9_scaling_report.txt`
- Round 10 (interpretability strategy):  
  `python src/generate_week10_queries_interpretable.py --data-dir initial_data --output src/module21_week10_queries.txt --report src/module21_week10_interpretability_report.txt`
- Round 11 (clustering-aware strategy):
  `python src/generate_week11_queries_clustering.py --data-dir initial_data --output src/module22_week11_queries.txt --report src/module22_week11_cluster_report.txt`
- Round 12 (PCA-guided strategy):
  `python src/generate_week12_queries_pca.py --data-dir initial_data --output src/module23_week12_queries.txt --report src/module23_week12_pca_report.txt`

Formatting rule reminder:
- Portal query values are emitted with six decimals in `0.xxxxxx` format.
