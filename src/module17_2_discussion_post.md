# Module 17.2 Discussion Post - Technical Justification

1. What is the main technical justification for your current BBO approach? Which aspect of prior research or established methods supports your choice?

My main technical justification is that I am solving a classic expensive black-box optimization problem with limited sequential evaluations, which is exactly the regime where Bayesian Optimization is strongest. I use a GP-centered hybrid pipeline with input warping and mixed acquisition scoring (EI, PI, UCB, plus uncertainty/novelty constraints). This is supported by established BO practice: surrogate modeling to estimate unknown objective structure, then uncertainty-aware acquisition to select the next query efficiently.

2. Which academic papers have you used to guide your design? Which ideas or techniques from the literature are most relevant, and how do they strengthen your project?

The most important references are:
- Brochu, Cora, and de Freitas (2010): foundational BO tutorial and the surrogate + acquisition decision loop.
- Snoek, Larochelle, and Adams (2012): practical BO for ML settings, especially the importance of model calibration and acquisition behavior.
- Shahriari et al. (2016): broader review that reinforces exploration-exploitation design under uncertainty.
- Eriksson et al. (2021, TuRBO): useful guidance for difficult, higher-dimensional search where local structure matters.

These sources strengthen my project because they justify my core choices (GP surrogates, uncertainty-driven query selection, adaptive strategy by regime) and give a defensible rationale for why my method should work under tight query budgets.

3. Which third-party libraries or frameworks (e.g. PyTorch, TensorFlow, scikit-learn) are central to your approach? Why were these the right choices compared with possible alternatives?

The central libraries are NumPy, SciPy, and scikit-learn. NumPy handles deterministic candidate generation and array operations; SciPy supports stable numerical components in acquisition calculations; scikit-learn provides reliable GP/SVR/MLP implementations for fast experimentation.

These were the right choices for this capstone because they reduce implementation overhead and improve reproducibility. Compared with alternatives like BoTorch/GPyTorch or specialized HEBO implementations, my stack is less expressive, but easier to audit and iterate weekly. That trade-off is acceptable for this project stage, where clarity and consistency matter as much as maximum modeling flexibility.

4. How do you plan to document and present these justifications in your GitHub repository so that peers, facilitators and future employers can clearly understand your reasoning?

I will document justification at three levels:
- README strategy log: round-by-round evolution and why each method change was made.
- Reproducible scripts: one script per major strategy stage, with explicit run commands and seeds.
- Module writeups: dedicated files for query submissions, reflections, and technical justification (including references).

This structure shows both what I did and why I did it. It also makes the project auditable: a reviewer can map design claims directly to code, commands, and generated outputs.

5. Looking ahead, what additional sources (research, benchmarks, software) might you consult to continue refining your strategy?

Next, I would consult:
- BoTorch/GPyTorch documentation and examples for stronger BO implementations.
- HEBO and related benchmark studies for robust mixed-strategy acquisition design.
- Benchmark suites such as COCO/BBOB and HPOBench to evaluate method behavior beyond this capstone scenario.
- Comparative optimization tools (for example Optuna/Nevergrad) as baselines.

The goal is to move from "works in this project" to "validated across benchmarked settings," while keeping the repository transparent and reproducible.
