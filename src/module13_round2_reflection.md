# Module 13 - Round 2 Submission Notes

## Part 1: Query Submission (Draft)

Generated from current local data using a GP-UCB surrogate (`src/generate_round_queries.py`).

- `function_1`: `0.121405-0.735201`
- `function_2`: `0.605451-0.732154`
- `function_3`: `0.781102-0.910980-0.353893`
- `function_4`: `0.402436-0.416461-0.331706-0.458573`
- `function_5`: `0.544589-0.633942-0.879495-0.357664`
- `function_6`: `0.367929-0.128059-0.664246-0.998752-0.001967`
- `function_7`: `0.249447-0.331380-0.293880-0.151089-0.842909-0.737173`
- `function_8`: `0.001012-0.394204-0.094198-0.125813-0.981797-0.508460-0.011126-0.663823`

If you have newer post-portal data, regenerate before submission:

```bash
python src/generate_round_queries.py --data-dir initial_data
```

## Part 2: Reflection Draft

### 1) Main strategy change from last week

The main change was moving from mostly heuristic/manual point selection to a surrogate-model-based strategy. I fit a Gaussian Process per function and selected the next point with a UCB-style acquisition score. This change was prompted by having more observations than week 1, which made model-driven uncertainty estimates more useful and reduced purely random exploration.

### 2) Exploration vs exploitation trade-off

This round used a balanced strategy with a slight exploitation bias in lower dimensions and more exploration in higher dimensions. In practical terms, I prioritized points with high predicted values but kept an uncertainty bonus in the acquisition function to avoid over-committing to a local optimum too early. The trade-off was between short-term gain (high mean prediction) and information gain (high uncertainty regions).

### 3) Influence of peers / class discussion / outputs

Recent outputs had the strongest influence on my approach. I shifted away from aggressive local search when results suggested non-convex response surfaces. Class discussion around local-optimum risk reinforced using explicit exploration terms instead of only chasing the current best region.

### 4) Likely violated assumptions for linear/logistic models

For these black-box functions, linear-model assumptions are often violated:

- linearity between features and response is unlikely in most functions
- residual variance is not constant across the input space
- small sample size relative to dimensionality (especially in 6D and 8D) weakens coefficient stability
- potential interaction effects are strong but not explicitly modeled in basic linear/logistic forms

For logistic regression specifically, thresholded labels may not be separable with a linear boundary in the original feature space.

### 5) Regions that appear linear or form boundaries

Some functions show locally smooth behavior (for example in parts of Function 4 and Function 8), so linear approximations can be partially useful within narrow neighborhoods. A logistic classifier could work reasonably if the task is reframed as classifying points above/below a performance threshold, but it would still miss nonlinear boundary structure unless features are transformed.

### 6) Usefulness of interpretability in query decisions

Interpretability was useful as a diagnostic step. Looking at feature effects and coefficient direction helps sanity-check whether the query is plausible and whether specific dimensions are likely to matter. However, final query decisions relied more on nonlinear surrogate predictions and uncertainty because these functions are black-box and likely non-linear.

## Learning Outcomes Link

- Linear vs logistic suitability: linear regression is weak for global prediction on non-linear black-box surfaces, while logistic regression can still be useful for threshold-based decision support.
- Trade-offs: logistic regression improves interpretability and probabilistic classification, but loses fidelity when boundaries are highly non-linear.
- Output interpretation: model coefficients and validation metrics are useful for diagnostics, while acquisition-driven methods remain better for selecting next-query points under uncertainty.
