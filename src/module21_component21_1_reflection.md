# Module 21.1 Reflection: Transparency and Interpretability

This round, I prioritized a decision rule that I could explain clearly: each candidate was scored by a transparent combination of predicted value, uncertainty, and novelty, then the final query was chosen deterministically. That made it easier to inspect why a point was selected and which dimensions were driving the decision.

## Reasoning behind the round

Previous rounds suggested different regimes across functions:
- Functions 1-3 still show strong boundary behavior, so exploitation near edges remains sensible.
- Function 4 looks more like local refinement in a structured basin.
- Function 5 remains highly sensitive to a few dominant dimensions.
- Functions 6-8 still require more uncertainty-aware moves because dimensionality is higher and coverage is sparser.

## Transparency and reproducibility

My process is now reasonably transparent because another researcher could reproduce:
- the query generator
- the tuned model family choice
- the explicit scoring formula
- the interpretability report with top influential dimensions

To fully understand the logic, they would need the current data arrays, the generation script, the random seed, and the explanation report.

## Key assumptions

One important assumption is that local surrogate behavior is informative enough to guide the next query. If this assumption is wrong, the model may overemphasize smooth local trends and miss abrupt changes in the true function. That shapes results by favoring interpretable local structure over more speculative global jumps.

## Data gaps and bias

There are still sampling biases in the dataset:
- some functions have repeated attention near boundaries
- some central or off-axis regions remain underexplored
- higher-dimensional functions have inherently thinner coverage

This means observed patterns may reflect where I looked most often, not only the true shape of the objective.

## Main limitation

The biggest limitation is that interpretability does not guarantee correctness. A transparent score can still be built on a biased or incomplete surrogate. So while this round improves trust and reproducibility, it does not remove the risk of model misspecification under limited data.
