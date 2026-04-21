# Module 23.1 Reflection: PCA-Inspired Optimization

This round, I treated the observed query history as a high-dimensional dataset and used PCA-style reasoning to reduce randomness while keeping exploration focused on the most informative directions.

## Strategy evolution

My early rounds were mostly heuristic and uncertainty-driven. By this stage, the process is more structured:
- tune the surrogate family
- estimate dominant variation directions with PCA
- generate candidates along those directions
- score them with prediction, uncertainty, novelty, and PCA alignment

This is much more systematic than the early “search widely and learn” phase.

## Principal directions in the data

The PCA report suggests that different functions are driven by different dominant directions:
- Functions 1 and 2 are strongly shaped by joint movement along both input axes.
- Function 5 is dominated by upper-bound behavior in a few key dimensions.
- Function 4 shows a basin-like structure with mixed contributions from several coordinates.
- In Function 8, a smaller subset of dimensions appears to explain much of the observed variation.

## Reducing vs preserving complexity

PCA reasoning helped me decide which directions to keep emphasizing and which to simplify. If a direction repeatedly explains variation and aligns with better outcomes, I keep exploring it. If a dimension contributes little to principal variation, I reduce attention to it unless uncertainty suggests it still matters.

## Influence on the final round

This round helps prepare the final submission by identifying which directions look robust enough for exploitation. For the last round, I would likely keep only a small amount of exploration in unresolved high-dimensional functions and exploit the strongest PCA-supported directions elsewhere.

## Main lesson

The main PCA insight is that not all variation is equally useful. Focusing on dominant directions helps compress the search problem, but it also carries a risk: low-variance directions can still hide important local structure. So PCA is most useful as a guide for prioritization, not as a rule for ignoring the rest of the space.
