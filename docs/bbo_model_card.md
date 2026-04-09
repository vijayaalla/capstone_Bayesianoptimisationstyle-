# BBO Strategy Model Card

## Model details

This project does not use one fixed model. It uses a modular BBO pipeline that has evolved across rounds:
- GP surrogate methods
- SVR ensembles
- MLP ensembles
- hybrid acquisition strategies
- interpretability-focused scoring in later rounds

The current transparency-first round uses tuned surrogate selection plus an explicit score decomposition:
- predicted value
- uncertainty
- novelty

## Intended use

- selecting one next query point per function in the BBO capstone project
- comparing optimization strategies across low- and high-dimensional objectives
- documenting reproducible reasoning for sequential decisions

## Factors

Model behavior depends on:
- number of observed points
- dimensionality of the function
- surrogate family selected by tuning
- coverage bias in previously sampled regions

## Metrics

Relevant internal metrics include:
- cross-validated MAE for surrogate selection
- ensemble disagreement / uncertainty
- score decomposition for chosen candidates
- round-to-round best observed function values

## Evaluation data

Evaluation is performed on the same evolving capstone dataset via cross-validation and sequential query outcomes returned by the portal.

## Limitations

- sparse observations can cause model misspecification
- local interpretability may not reflect true global function structure
- high-dimensional functions remain difficult to cover well with limited budget
- transparency does not eliminate sampling bias or uncertainty

## Risks

The main risk is false confidence: interpretable outputs can still be wrong if the surrogate is fit on incomplete or biased data. Query choices should therefore be read alongside uncertainty and data coverage, not only predicted value.

## Training and tuning

Surrogate hyperparameters are tuned with lightweight search methods such as bounded random search and cross-validation. Later rounds may incorporate additional strategies such as context-aware scoring or scale-aware ensembles.

## How to use responsibly

- update local data before generating the next query set
- inspect both scores and uncertainty before trusting a recommendation
- preserve random seeds and scripts for reproducibility
- treat the model as decision support, not oracle truth
