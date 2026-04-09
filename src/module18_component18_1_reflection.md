# Module 18.1 Reflection: Hyperparameter Tuning and Strategy Evolution

With larger data coverage (targeting the 16-point state for early functions), hyperparameter tuning became a central part of my BBO loop rather than an optional add-on. Instead of fixing one surrogate setup, I tuned model families and selected configurations based on cross-validated error before generating the next query.

## 1) How tuning changed my decision process

I moved from single-model assumptions to evidence-based selection:
- tested multiple surrogate families (SVR and MLP)
- tuned core hyperparameters through random-search style exploration
- selected the better model per function from CV MAE

This made my strategy more defensible because each query is linked to measured model quality, not only intuition.

## 2) Do tuned models make me rely more on predictions?

Yes, but not blindly. Better-tuned models improved local prediction stability, so I can exploit strong regions more confidently. However, I still preserve exploration through uncertainty and novelty terms in acquisition. Tuning increased trust in the surrogate, but did not remove the need for controlled exploration.

## 3) Exploiting strong regions vs testing unproven configurations

I now split effort by confidence regime:
- high-confidence regions: tighter local refinement
- low-confidence regions: broader candidate coverage

This mirrors practical tuning logic: use tuned settings where they are reliable, but keep enough diversity to avoid local lock-in and model bias.

## 4) What tuning revealed as data increased

Hyperparameter analysis exposed three important effects:
- **Overfitting risk**: some aggressive model settings looked strong in-sample but weak in CV.
- **Irrelevant dimensions**: for several functions, tuning favored smoother models, suggesting some dimensions contribute little signal.
- **Diminishing returns**: after a point, larger model complexity did not improve CV error enough to justify extra variance.

These findings changed my approach from “maximize model complexity” to “maximize decision quality per query.”

## 5) Methods perspective (grid, random, BO, Hyperband)

For this stage, random-search-style tuning gave a good speed/quality trade-off. Conceptually:
- grid search is simple but inefficient in larger spaces
- random search improves coverage of influential hyperparameters
- Bayesian tuning is more sample-efficient when evaluations are costly
- Hyperband is attractive when many poor configurations can be stopped early

As rounds continue, I would likely combine Bayesian tuning with early-stopping style allocation to improve efficiency.

Overall, hyperparameter tuning now acts as the control system for my BBO pipeline: it calibrates how much I trust predictions, how aggressively I exploit, and where I keep exploring.
