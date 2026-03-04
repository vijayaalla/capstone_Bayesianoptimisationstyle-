# Module 16 - Round 5 Submission + Reflection Draft

## Part 1: Query submission (portal format)

function_1: 0.000000-1.000000  
function_2: 1.000000-0.000000  
function_3: 0.570957-1.000000-1.000000  
function_4: 0.000000-0.000000-0.197642-0.000000  
function_5: 0.000000-1.000000-1.000000-1.000000  
function_6: 0.000000-0.000000-1.000000-1.000000-0.000000  
function_7: 0.000000-0.391404-0.118670-0.000000-0.127948-0.996521  
function_8: 0.049003-0.000000-0.000000-0.746439-1.000000-1.000000-0.000000-0.924122

## Part 2: Reflection (Discussion board post)

1. How did the ideas of hierarchical feature learning influence the way you thought about structuring or refining your optimisation strategy this round?

Hierarchical feature learning made me think less about one acquisition function and more about building a stack where each layer improves the signal passed to the next. In my workflow, that stack is: (i) data and cleaning, (ii) representation (including input warping, similar to the HEBO-style approach), (iii) surrogate choice (GP vs NN ensemble depending on dimension and data regime), (iv) acquisition choice (EI/PI/UCB instead of committing to one rule), and (v) a decision layer with heuristics that block obviously poor submissions.

2. You saw how breakthroughs such as AlexNet and ImageNet classification reshaped expectations in AI. What parallels do you see between those leaps in performance and the incremental improvements you make in your capstone submissions?

AlexNet/ImageNet felt like a performance jump created by scale plus architecture innovation. In my capstone, this week is more about rebuilding the pipeline in that spirit and testing a new HEBO-aligned regime across multiple rounds. The good part of the capstone format is that I can recalibrate even when earlier points came from a weaker strategy. For example, I still expect previously collected points for Functions 1 and 2 under exploratory UCB to remain informative. Right now, the goal is optimizing the loop itself, not claiming the final answer in one round.

3. When training neural networks, people often weigh trade-offs between depth, complexity and training efficiency. Did you encounter similar trade-offs in deciding whether to explore widely or exploit known promising regions in your queries?

This week introduced more sophisticated modeling choices, which increases capability but also raises overfitting risk with limited data. That maps directly to exploration vs exploitation. Broad exploration early helps avoid over-committing to local maxima, while later rounds should exploit high-confidence regions more aggressively. In other words, I am using exploration to build reliable coverage first, then shifting toward efficiency as evidence accumulates.

4. Reflecting on the building blocks of neural networks (inputs, activations, loss, gradients, weight updates), which of these concepts helped you think differently about how your model learns from the data you’ve accumulated so far?

The neural-network building-block analogy was useful even with GP models. Inputs correspond to normalized parameter vectors. Input warping plays a role similar to activations because it can stabilize variance and make structure easier to learn. Loss minimization in NNs maps to maximizing acquisition value in BO. Weekly queries are like iterative weight updates: each new point changes the model and reshapes uncertainty, which then changes the next decision.

5. Module 16 also introduced PyTorch and TensorFlow as different frameworks for building and scaling models. If you were to frame your current optimisation approach in terms of a ‘framework’, would it be closer to rapid prototyping and flexibility or to structured, production-ready design? Why?

My current setup is a prototype moving toward production readiness. It now mixes multiple acquisition options (EI/UCB/PI) and uncertainty-focused options (including sigma reduction when confidence is low). This is the first week under the updated regime, so I expect continued tuning over upcoming rounds as new outputs arrive. The aim is to keep research-level flexibility while steadily hardening the pipeline into something repeatable and trustworthy.

6. In the guest interview, Giovanni Liotta discussed industry applications of deep learning in sport. How might reflecting on real-world deep learning use cases inform the way you benchmark success in your own capstone challenge?

It is harder to map Giovanni Liotta’s sports examples directly to this capstone because his setting uses strong domain knowledge, while this challenge is intentionally abstract. In a real application, domain experts would help validate whether observed outputs are plausible and where constraints should guide query decisions. So my benchmark here is technical consistency and improvement under limited data, but in industry I would combine those metrics with regular expert feedback before trusting the model’s recommendations.
