# BBO Capstone Project

## Section 1: Project Overview

This capstone project simulates a Bayesian black-box optimisation (BBO) challenge across eight unknown functions.  
For each function, I only see queried inputs and returned outputs. I do not know the underlying equation, gradient, or shape in advance.

The overall goal is to maximise each function with a limited number of sequential queries. This mirrors real-world ML settings where experiments are expensive, delayed, or noisy (for example hyperparameter tuning, scientific testing, or operational optimisation).

Career relevance: this project develops practical skills in uncertainty-aware decision-making, model selection under limited data, iterative experimentation, and communicating technical trade-offs clearly.

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

## Section 4: Technical Approach (Rounds 1-6)

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

## Section 8: Reproducible Commands

Query generation commands:
- Round 3 (SVM):  
  `python src/generate_week3_queries_svm.py --data-dir initial_data --output src/module14_week3_queries.txt`
- Round 4 (NN ensemble):  
  `python src/generate_week4_queries_nn.py --data-dir initial_data --output src/module15_week4_queries.txt`
- Round 5 (deep ensemble):  
  `python src/generate_week5_queries_deep_ensemble.py --data-dir initial_data --output src/module16_week5_queries.txt`
- Round 6 (HEBO-inspired hybrid):  
  `python src/generate_week6_queries_hebo_hybrid.py --data-dir initial_data --output src/module17_week6_queries.txt`

Formatting rule reminder:
- Portal query values are emitted with six decimals in `0.xxxxxx` format.
