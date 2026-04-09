# BBO Dataset Datasheet

## Motivation

This dataset supports a capstone project in Bayesian black-box optimisation. The goal is to select sequential query points that maximize eight unknown objective functions under a limited evaluation budget.

## Composition

The dataset consists of eight independent function-specific subsets:
- `function_1`: 2D inputs
- `function_2`: 2D inputs
- `function_3`: 3D inputs
- `function_4`: 4D inputs
- `function_5`: 4D inputs
- `function_6`: 5D inputs
- `function_7`: 6D inputs
- `function_8`: 8D inputs

Each subset contains:
- `initial_inputs.npy`: query points in `[0, 1]^d`
- `initial_outputs.npy`: scalar function values

## Collection process

The data was provided as part of the capstone setup and expanded over time through sequential query submissions. Each round adds one new query per function after portal evaluation.

## Preprocessing

- Inputs are already normalized to `[0, 1]`.
- Outputs are stored as raw scalar values.
- Surrogate models typically standardize inputs internally before fitting.

## Intended use

- sequential black-box optimization experiments
- surrogate-model comparison
- acquisition-function design
- reflection on uncertainty, transparency, and decision-making

## Limitations

- small data regime, especially in lower-round states
- uneven coverage across the search space
- no known ground-truth function equations
- possible bias toward regions explored in earlier rounds

## Risks and ethical considerations

This is a synthetic/educational optimization dataset, so there are no direct privacy concerns. The main risk is analytical overconfidence: sparse data may create misleading patterns if uncertainty is ignored.

## Maintenance

The dataset should be updated after each portal round by appending new query/output pairs to the relevant function arrays. Any downstream query-generation scripts should be rerun after updates.
