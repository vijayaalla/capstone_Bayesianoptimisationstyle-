# Module 24.1 Reflection: Final Strategy Through an RL Lens

This final round made the exploration-exploitation trade-off feel much more concrete. In the early rounds, exploration dominated because uncertainty was high and local evidence was weak. By the final round, repeated signals made exploitation more defensible, although I still tracked alternative policy arms to avoid overconfidence.

## Exploration vs exploitation

I compared three policy-style options:
- exploit arm
- explore arm
- PCA-guided arm

In the final report, the exploit arm won for all functions, but some explore scores stayed close. That suggests the system had largely converged toward known strong regions, while still recognizing that a few unresolved areas remained worth monitoring.

## Feedback as reward updating

As data accumulated, the output from each round acted like a reward signal. Good outcomes increased trust in some local regions and some policy patterns; weaker outcomes reduced their value. This is similar to RL updates where repeated feedback changes expected value estimates and gradually shifts behavior toward better actions.

## AlphaGo Zero analogy

My process partly resembled autonomous learning because each round’s output fed directly into the next decision without external labels. It was not self-play in the literal AlphaGo Zero sense, but it was an iterative feedback loop where the system improved from its own search history. Overall, the later rounds were more model-based than model-free because surrogate models were used to anticipate future outcomes before choosing the next query.

## Broader RL application

RL ideas could improve real-world optimization by:
- using contextual bandits to choose between query strategies
- applying UCB or Thompson-style exploration to control risk
- learning when to switch from broad exploration to local convergence

The main lesson is that optimization is not only about picking the current best point. It is also about updating beliefs from feedback, managing uncertainty over time, and deciding when enough evidence exists to exploit confidently.
