# Module 23 - Round 12 Submission + PCA Reflection Draft

## Part 1: Query submission (portal format)

function_1: 0.999999-0.999999  
function_2: 0.925948-0.999999  
function_3: 0.232042-0.265650-0.446766  
function_4: 0.325375-0.355495-0.372045-0.486321  
function_5: 0.000000-0.999999-0.978146-0.999999  
function_6: 0.584333-0.434804-0.616215-0.647230-0.307568  
function_7: 0.050656-0.289916-0.525146-0.312196-0.074406-0.779666  
function_8: 0.011875-0.008162-0.054213-0.016462-0.186538-0.817397-0.010852-0.141712

## Part 2: Reflection (Discussion board post)

1. How has your optimisation strategy evolved since your first few rounds of queries? Which elements now feel more structured or systematic?

My strategy has moved from broad heuristic exploration to a much more structured pipeline. In the early rounds, I mainly relied on simple uncertainty-aware heuristics. Now the process is more systematic: I tune the surrogate family, identify dominant directions in the observed data, generate targeted candidates, and score them with a fixed rule. That makes the workflow more reproducible and less dependent on manual intuition.

2. If you think of your current data set as a ‘high-dimensional’ space, which variables or behaviours seem to drive the largest variation in your results – similar to principal components in PCA?

The PCA report suggests that a small number of directions explain much of the observed variation for each function. For Functions 1 and 2, the two input axes move together strongly. For Function 5, the dominant variation is concentrated near upper-bound movement in a few key dimensions. Function 4 looks more distributed across several coordinates, while Function 8 appears to depend more heavily on a smaller subset of its many inputs than on all dimensions equally.

3. How do you decide which aspects of your strategy to keep exploring versus which to reduce or simplify, as PCA reduces dimensions while retaining essential information?

I keep exploring directions that both explain variance and still carry uncertainty. I simplify or de-emphasize directions that repeatedly contribute little variation or appear redundant. In practice, that means I reduce randomness in weak dimensions while preserving some exploration in directions that still look unresolved. The goal is not to remove complexity completely, but to focus attention on the parts of the space that seem to matter most.

4. How might this round of optimisation influence your next and final round of query submission in Module 24, especially when balancing exploration and exploitation?

This round helps narrow the space for the final submission. If a principal direction repeatedly aligns with stronger outcomes, it becomes a stronger exploitation candidate. If a function still has uncertainty in a dominant direction, I may keep a small amount of exploration there. So for the final round, I expect a more exploitative strategy overall, but still with limited exploration in the least-settled high-dimensional functions.

5. Reflect briefly on how insights from PCA, such as focusing on variance and removing redundancy, might apply to how you interpret your BBO results.

PCA is useful here because it encourages me to ask which patterns are truly carrying information and which are just repeated noise. Focusing on variance helps compress the search problem and reduce unnecessary randomness. At the same time, I have to be careful not to assume low-variance directions are irrelevant, because they could still contain narrow but important improvements. So the lesson is to reduce redundancy without becoming overconfident.
