# Module 20 - Round 9 Submission + Scaling Reflection Draft

## Part 1: Query submission (portal format)

function_1: 0.012238-0.976886  
function_2: 0.976766-0.020969  
function_3: 0.058302-0.996508-0.852649  
function_4: 0.366303-0.572518-0.485293-0.552972  
function_5: 0.218549-0.899971-0.971768-0.931623  
function_6: 0.522684-0.117039-0.907657-0.771524-0.041638  
function_7: 0.099302-0.020511-0.938158-0.004937-0.060092-0.968777  
function_8: 0.314321-0.462935-0.004773-0.165050-0.849560-0.946149-0.075651-0.808843

## Part 2: Reflection (Discussion board post)

1. How do scaling laws influence your current query choices? Do you see diminishing returns or steady improvements?

Scaling now influences my query choices through comparison across short-, medium-, and full-context views rather than assuming more context is automatically better. I do see both effects: some functions benefit from steadier improvements when larger context clarifies the search space, while others show diminishing returns because extra context adds noise or disagreement rather than better ranking.

2. Where might emergent behaviours alter your expectations, and how are you preparing for them?

Emergent behavior matters when larger-context models prefer a region that smaller-context models would not have selected. That can be useful because it may reveal structure that only appears once more evidence is available. At the same time, it can be risky if the signal is unstable. I prepare for this by tracking disagreement between scales and rewarding emergence only when it appears alongside reasonable robustness, rather than chasing every sudden shift.

3. What trade-offs between cost, robustness and performance are shaping your strategy now?

The main trade-off is that scaling context and model views increases computation, but can also improve robustness if it confirms a candidate across multiple perspectives. However, more context can also reduce robustness if it pulls attention toward irrelevant history. So I am no longer optimizing only for the highest predicted value. I am also optimizing for stability across scales and consistency under constrained decoding.

4. How do you balance predictable optimisation with the risk of sudden but uneven emergent capabilities?

I balance this by combining predictable signals and emergent signals instead of choosing one over the other. Predictable optimization still comes from stable high-scoring regions, but emergent gains are allowed to influence the final choice through a separate score. Constrained decoding and format guardrails keep that exploration controlled. This lets me benefit from possible step changes without overcommitting to one surprising behavior that may not generalize.
