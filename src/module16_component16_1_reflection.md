# Module 16.1 Reflection: Advanced Neural Networks and BBO Strategy

For Round 5, I treated the BBO task as a small-data deep-learning problem: model non-linear response surfaces, estimate uncertainty, and choose the next query with controlled exploration.

## 1) AlexNet/ImageNet lessons applied to BBO

The AlexNet/ImageNet breakthrough showed that model capacity plus computational strategy can uncover structure that simpler models miss.  
In BBO terms, this means I should not rely on one shallow surrogate when function behavior is non-linear and high-dimensional. Instead, I used a **diverse ensemble of neural surrogates** so that:

- different architectures capture different local/global patterns
- model disagreement acts as an uncertainty signal
- query selection is less brittle than single-model predictions

## 2) Five deep-learning building blocks and how they improved decisions

I mapped the main deep-learning building blocks directly into my query pipeline:

1. **Architecture**: use varied hidden-layer depth/width to represent multiple hypotheses about function shape.
2. **Forward prediction**: evaluate many candidate points through all ensemble members.
3. **Loss and optimisation**: train with Adam and bounded learning rates so models converge without extreme instability.
4. **Regularisation**: use L2 penalty and early stopping to reduce overfitting in low-data regions.
5. **Generalisation/uncertainty check**: use ensemble mean and standard deviation in the acquisition function.

This changed my strategy from "find one best surrogate" to "use a calibrated set of surrogates and trust consensus plus disagreement."

## 3) How advanced concepts changed Week 5 query behavior

Compared with earlier rounds, my Week 5 method was more deliberate in how it balances exploration and exploitation:

- exploitation: high ensemble mean prediction
- exploration: uncertainty bonus from ensemble standard deviation
- robustness: distance filter and novelty bonus to avoid repeatedly sampling near old points

I also used a two-stage candidate search (global random scan + local refinement around high-value seeds), which is similar to deep-learning practice of coarse-to-fine optimisation.

## 4) Framework relevance (PyTorch/TensorFlow)

This round used `scikit-learn` MLPs for speed, but the same design scales naturally to PyTorch/TensorFlow:

- easier custom architectures for high-dimensional functions
- explicit training loops for richer uncertainty approaches (MC dropout, deep ensembles)
- better experiment tracking and reproducibility for larger model sweeps

For future rounds, moving to PyTorch/TensorFlow would make it easier to add custom losses and more stable uncertainty estimation when data volume grows.

## 5) What this means for my BBO strategy going forward

- Keep ensemble-based neural surrogates as default for 4D+ functions.
- Adjust exploration weight by dimensionality and sample count.
- Continue using regularisation and early stopping as guardrails.
- Validate model behavior with sensitivity checks before locking final queries.

Overall, advanced deep-learning concepts improved my BBO process by making query selection more robust, uncertainty-aware, and scalable as function complexity increases.
