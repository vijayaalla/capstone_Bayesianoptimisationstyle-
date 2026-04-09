# Module 21 - Round 10 Submission + Transparency Reflection Draft

## Part 1: Query submission (portal format)

function_1: 0.998699-0.000998  
function_2: 0.883916-0.999907  
function_3: 0.991282-0.986943-0.001498  
function_4: 0.337972-0.378628-0.347483-0.465718  
function_5: 0.858854-0.959788-0.990394-0.994515  
function_6: 0.671224-0.060565-0.994241-0.360432-0.005477  
function_7: 0.071642-0.202510-0.751891-0.008595-0.129151-0.976096  
function_8: 0.129787-0.265632-0.149370-0.965969-0.054148-0.029964-0.003875-0.265343

## Part 2: Reflection (Discussion board post)

1. What reasoning guided your submission for this tenth round? Explain your strategy. How did patterns from the previous rounds influence your decisions for each function?

My main strategy was to use a transparent score built from predicted value, uncertainty, and novelty, then choose the best candidate deterministically. Previous rounds suggested different patterns by function: Functions 1-3 still look boundary-driven, so I leaned more on exploitation near edges; Function 4 looked more like local refinement in a structured basin; Function 5 still appears dominated by a few strong dimensions; Functions 6-8 remain more uncertainty-sensitive because higher dimension makes coverage thinner. I also used the interpretability report to see which dimensions were most influential before finalizing each point.

2. How transparent is your decision-making process? If another researcher reviewed your notes and data, could they follow your logic and reproduce your strategy? What information would they need to fully understand your approach?

My process is fairly transparent now. Another researcher could follow it if they had the current data arrays, the query-generation script, the random seed, and the interpretability report showing the score decomposition and top influential dimensions. The logic is more reproducible than earlier rounds because I reduced ad hoc judgment and used a fixed, explainable scoring rule.

3. What assumptions are you making in your search/optimisation strategy? Identify at least one key assumption related to the functions or the optimisation process. How might this assumption shape or limit your results?

One key assumption is that the local surrogate shape is informative enough to guide the next query. That means I assume local gradients and uncertainty are useful signals rather than noise. If this assumption is wrong, my strategy may over-trust smooth local structure and miss abrupt changes or disconnected high-value regions.

4. Where do you see gaps or potential biases in your data set? Consider the distribution of your queries, unexplored areas or patterns in how you sampled the search space.

The biggest bias is uneven coverage. Some functions have many points near boundaries because earlier rounds suggested those areas were promising, while other regions remain underexplored. Higher-dimensional functions also have sparse coverage almost by definition, so the model may treat limited observations as stronger evidence than they really are. That can bias the search toward familiar regions.

5. What is one significant limitation of your approach? Consider factors such as computational constraints, sampling biases or assumptions about function behaviour that might affect the validity or generalisability of your results.

The biggest limitation is that interpretability improves trust, but it does not guarantee correctness. A transparent decision rule can still be built on a biased dataset or a misspecified surrogate. So while this round is stronger in terms of documentation and reproducibility, the strategy is still constrained by limited samples and by the assumption that the model captures the most important structure of each function.
