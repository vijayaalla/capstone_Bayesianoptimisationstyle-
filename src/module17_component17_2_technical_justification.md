# Module 17.2 Technical Justification: Why This BBO Approach Is Credible

## 1) Technical position

My current BBO approach is a practical hybrid:
- Gaussian Process surrogate modeling
- input warping to handle non-stationarity
- mixed acquisition design (EI, PI, UCB, uncertainty term)
- novelty and distance constraints to avoid redundant queries

This is designed for small, sequential, expensive-query settings, which matches the capstone constraints.

## 2) Academic research grounding

The approach is consistent with core Bayesian optimisation literature:

- **Gaussian Process BO**: GP surrogates and acquisition-driven sampling are standard for expensive black-box objectives.
- **Input warping / non-stationarity handling**: warping can improve surrogate fit when response behavior changes across regions.
- **Acquisition trade-offs**: EI/PI/UCB capture different exploration-exploitation behavior; combining them is a practical robustness tactic in heterogeneous problems.
- **Practical BO under uncertainty**: constrained, sequential improvement with explicit uncertainty handling is a common recommendation in applied BO.

## 3) State-of-the-art influences

I drew from patterns used by modern BO systems:
- **HEBO-style thinking**: flexible representation handling plus robust acquisition selection.
- **Ensemble mindset**: avoid dependence on one model/acquisition when objective geometry is uncertain.
- **Adaptive policy by regime**: low-dimensional functions can exploit earlier; high-dimensional functions need stronger uncertainty coverage longer.

I am not claiming full SOTA implementation, but the design choices are aligned with SOTA principles adapted to this project’s scale.

## 4) Third-party packages and why they are appropriate

- **NumPy**: deterministic array operations and candidate generation.
- **scikit-learn**: reliable GP/SVR/MLP implementations with reproducible APIs.
- **SciPy (via scientific stack)**: stable numerical functions used in acquisition calculations.

Why these packages fit:
- fast iteration for weekly rounds
- low implementation overhead
- easy reproducibility and auditability for coursework review

Trade-off:
- compared with specialized BO stacks (BoTorch/GPyTorch/HEBO library), this setup is less expressive, but simpler to control and explain under time constraints.

## 5) Validation logic used to support decisions

To justify technical choices, I use the following checks:
- reproducible scripts and fixed seeds
- distance filters to reduce duplicate sampling
- uncertainty-weighted scoring rather than point estimates only
- round-to-round performance tracking across all functions, not single-function cherry-picking

This supports credibility because decisions are traceable and falsifiable.

## 6) Risks and mitigation

- **Risk: local over-exploitation**  
Mitigation: retain uncertainty and novelty terms.

- **Risk: model misspecification in higher dimensions**  
Mitigation: use blended acquisition + optional NN ensemble fallback.

- **Risk: brittle one-week tuning**  
Mitigation: treat each week as an update to a stable pipeline, not a full reset.

## 7) Conclusion

My BBO strategy is technically grounded, practically implementable, and aligned with established research and modern BO system design. The approach is credible because it combines theory-backed components with transparent engineering controls suitable for constrained, iterative black-box optimisation.

## References

1. Brochu, E., Cora, V. M., & de Freitas, N. (2010). A Tutorial on Bayesian Optimization of Expensive Cost Functions.
2. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms.
3. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). Taking the Human Out of the Loop: A Review of Bayesian Optimization.
4. Eriksson, D., et al. (2021). Scalable Global Optimization via Local Bayesian Optimization (TuRBO).
5. scikit-learn documentation (GaussianProcessRegressor, MLPRegressor, SVR).
