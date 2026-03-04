# Module 17.1 Reflection: Strategy Evolution with 15 Data Points and CNN Concepts

By this stage, my strategy has shifted from isolated query decisions to a full optimisation pipeline. With more observations (targeting the 15-point state for early functions), each new query is less about random exploration and more about controlled improvement under uncertainty.

## 1) How CNN concepts influenced this round

CNN ideas helped me structure the process as layered transformations:
- **early layer**: data quality checks and normalization
- **representation layer**: input warping to make the search space smoother
- **model layer**: surrogate selection by regime (GP-focused with NN ensemble fallback)
- **decision layer**: blended acquisition and practical guardrails

This mirrors CNN intuition where useful high-level behavior emerges from sequential processing layers.

## 2) How strategy changed with more data

With larger data support, I reduced purely exploratory moves and increased selective exploitation:
- stronger local refinement near high-performing regions
- retained uncertainty sampling in under-covered/high-dimensional zones
- explicit rejection of near-duplicate points through distance thresholds

The net effect is a more stable explore/exploit balance: fewer extreme jumps, more evidence-based progress.

## 3) Acquisition design and why

Instead of committing to one acquisition rule, I used a hybrid score:
- Expected Improvement (EI) for direct objective gain
- Probability of Improvement (PI) for threshold crossing
- Upper Confidence Bound (UCB) for uncertainty-aware optimism
- uncertainty weight and novelty term for coverage quality

This is similar to ensemble reasoning in deep learning: no single signal is trusted blindly; decisions combine multiple views.

## 4) Trade-offs I managed this round

- **Model capacity vs robustness**: more expressive methods can overfit in sparse regions.
- **Exploration vs exploitation**: too much exploitation risks local traps; too much exploration wastes budget.
- **Consistency vs adaptability**: fixed defaults improve repeatability, but per-function adaptation improves results.

My current stance is to keep repeatable defaults while allowing bounded per-function adaptation.

## 5) What this implies for remaining rounds

- Keep the hybrid BO pipeline as the default loop.
- Adjust uncertainty weight dynamically as coverage improves.
- Prioritize robustness across all eight functions over one-function peak gains.
- Continue documenting rationale for each query so the process is auditable.

Overall, CNN-style layered thinking improved not only model selection, but the structure of the full decision process for BBO.
