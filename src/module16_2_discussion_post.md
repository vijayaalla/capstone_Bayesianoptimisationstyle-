# Module 16.2 Discussion Post - Repository Architecture and Documentation

## Repository structure

So far, I have organized the repository into two main areas:

- `initial_data/` stores function-wise inputs and outputs (`function_*/initial_inputs.npy`, `initial_outputs.npy`).
- `src/` stores query-generation scripts, round submissions, and reflection documents.

This structure has worked for rapid iteration across rounds because it keeps data and experimentation artifacts separate. However, as the project grows, clarity and reproducibility need stronger conventions.

The improvements I am making are:

- Keep one script per strategy milestone (`generate_round_queries.py`, `generate_week3_queries_svm.py`, `generate_week4_queries_nn.py`, `generate_week5_queries_deep_ensemble.py`).
- Keep one submission/reflection document per module so evidence is traceable over time.
- Use reproducible script entry points with explicit seeds and command examples in documentation.
- Update local data before every new round so generated queries match the latest portal state.

These changes improve navigability (clear file roles), reproducibility (deterministic commands), and maintainability (modular scripts instead of one large notebook workflow).

## Coding libraries and packages

The central libraries in my workflow are:

- `NumPy` for data loading, array operations, and candidate generation.
- `scikit-learn` for surrogate modeling (Gaussian Process, SVR, MLP ensembles), preprocessing, and baseline reliability.

I chose these because the capstone is a small-to-medium data optimization workflow where fast experimentation and stable APIs matter. `scikit-learn` is strong for this stage because it lets me compare model families quickly with minimal boilerplate.

Trade-offs considered:

- `scikit-learn` gives speed and simplicity, but less flexibility for custom deep architectures.
- `PyTorch`/`TensorFlow` would be better for larger-scale deep models, custom losses, and production-level training loops, but add implementation overhead for this capstone scope.

So my current approach is pragmatic: use `scikit-learn` for fast, evidence-driven iteration now, while keeping the architecture modular enough to migrate parts to PyTorch/TensorFlow if the problem scale increases.

## Documentation

My README currently explains:

- project purpose (black-box function maximization under query limits),
- input/output format and constraints,
- round-by-round strategy progression,
- exploration vs exploitation policy,
- and repository architecture direction.

Supporting documents in `src/` capture module-specific submissions and reflections, which helps show not just what I did, but why I changed strategy each round.

The updates I am applying for alignment with recent work are:

- Add Round 5 method details (deep-ensemble surrogate, uncertainty-aware acquisition, novelty/distance constraints).
- Keep architecture decisions explicit in a dedicated Module 16.2 document.
- Ensure each reflection references concrete strategy changes rather than generic ML statements.
- Keep commands and file references up to date so another reader can reproduce query generation directly.

Overall, my documentation goal is to make the repository readable as an engineering narrative: data state -> model choice -> query decision -> reflection -> next iteration.
