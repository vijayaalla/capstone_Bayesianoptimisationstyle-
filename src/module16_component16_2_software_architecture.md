# Module 16.2 - Software Architecture for the BBO Repository

## Architecture choice

I will use a **modular pipeline architecture** with clear boundaries between data handling, surrogate modeling, acquisition logic, and reporting artifacts.

This is the best fit because the capstone workflow is iterative and repeatable:
- load/update function datasets
- train one or more surrogate models
- score candidate points with an acquisition rule
- export query strings and reflections for submission

## Target repository structure

```text
capstone_Bayesianoptimisationstyle-/
  initial_data/                     # function_*/initial_inputs.npy, initial_outputs.npy
  src/
    generate_round_queries.py       # GP baseline
    generate_week3_queries_svm.py   # SVM ensemble strategy
    generate_week4_queries_nn.py    # NN ensemble strategy
    generate_week5_queries_deep_ensemble.py
    module*_submission.*            # portal-ready query files + writeups
    module*_reflection.*            # assessment reflections
  README.md                         # strategy log and project overview
```

## Logical components and responsibilities

1. **Data Layer**
- Source: `initial_data/function_*/initial_inputs.npy` and `initial_outputs.npy`
- Responsibility: provide consistent `(X, y)` arrays per function.
- Rule: after every portal round, update this layer first before generating the next query set.

2. **Surrogate Layer**
- Implements model families (GP, SVR ensemble, MLP/deep ensemble).
- Responsibility: estimate function behavior from sparse observations.
- Design principle: keep each round script reproducible with explicit seeds and bounded hyperparameter ranges.

3. **Acquisition Layer**
- Implements query objective: `mean_prediction + beta * uncertainty (+ optional novelty)`.
- Responsibility: choose one next point per function with exploration/exploitation balance.
- Safety checks: enforce minimum distance from existing points.

4. **Orchestration Layer**
- Round script entry points (`python src/generate_*.py ...`).
- Responsibility: run end-to-end for all functions and write portal-ready outputs.

5. **Documentation Layer**
- `README.md` + module files.
- Responsibility: trace decisions, strategy changes, and reflections by module.

## Data and control flow

1. Load current arrays for each `function_i`.
2. Train surrogate ensemble for that function.
3. Generate candidate pool in `[0,1]^d`.
4. Score candidates with acquisition.
5. Pick best valid candidate and format as portal string.
6. Save `function_i: x1-x2-...` output file.
7. Record rationale/reflection in module document.

## Quality attributes this architecture supports

- **Reproducibility**: deterministic seeds and script-based generation.
- **Extensibility**: new surrogate methods can be added without rewriting data/report layers.
- **Traceability**: each module has explicit submission and reflection artifacts.
- **Pragmatic maintainability**: scripts remain small and purpose-specific for fast iteration.

## Implementation decision for this module

- Keep current file layout for continuity with previous modules.
- Add `generate_week5_queries_deep_ensemble.py` as the Round 5 model pipeline.
- Continue documenting every required component in separate module files for grading clarity.
