# Module 17 - Round 6 Submission + Reflection Draft

## Part 1: Query submission (portal format)

function_1: 0.703450-0.003660  
function_2: 0.694771-0.451208  
function_3: 0.666853-0.995911-0.413924  
function_4: 0.482386-0.477707-0.426482-0.467246  
function_5: 0.262215-0.996503-0.999999-0.999999  
function_6: 0.025785-0.287917-0.528724-0.905888-0.072583  
function_7: 0.012652-0.184847-0.200768-0.140685-0.391243-0.874779  
function_8: 0.016011-0.130441-0.128941-0.012850-0.995973-0.695065-0.150707-0.419810

## Part 2: Reflection (Discussion board post)

1. CNNs build up features from edges and textures to full objects. How did this idea of progressive feature extraction influence the way you thought about refining your BBO strategy?

This idea pushed me to design my BBO process as a progressive stack rather than one isolated model choice. I now separate the workflow into layers: data checks and normalization, representation shaping (input warping), surrogate fitting, acquisition scoring, and final decision constraints. Each layer improves the quality of information passed to the next layer. With 15-point data for early functions, this structure helped me move from broad pattern detection to more targeted local refinement.

2. LeNet and later CNNs redefined what is possible in computer vision. What parallels do you see between those breakthroughs and the incremental improvements you make in your BBO capstone project?

The parallel is the combination of occasional jumps plus steady iteration. In deep learning history, LeNet opened a path and later CNNs expanded performance through better architecture and training. In my capstone, I see similar behavior: some rounds produce visible jumps when a better strategy is introduced, while most rounds deliver smaller incremental gains. My current shift to a HEBO-inspired hybrid pipeline feels like a method-level upgrade, and I expect the main benefits to appear over several rounds rather than in a single submission.

3. Training CNNs often involves balancing depth, computational costs and overfitting risks. Did you face similar trade-offs when choosing whether to explore widely or exploit promising regions in your queries?

Yes. A richer modeling setup improves capability but increases computational cost and overfitting risk in sparse regions. This is directly analogous to exploration versus exploitation in BBO. Wider exploration gives better global coverage but uses budget quickly; aggressive exploitation gives faster local gains but can lock into local optima. My current policy is to keep stronger exploration for uncertain/high-dimensional functions and shift to controlled exploitation where confidence is higher.

4. Convolutions, pooling, activations and loss functions influence how CNNs learn from data. Which of these concepts helped you think differently about how your optimisation model learns from your accumulated data?

The most useful analogies for my BBO setup were activations and loss. Input warping plays a role similar to activation behavior because it reshapes the space into a form that is easier to model. Loss-function thinking maps to acquisition design: EI/PI/UCB each represent different optimization priorities, so combining them is like balancing multiple learning objectives. Pooling also influenced my thinking as a simplification concept; in BBO terms, that translates to reducing noise through robust scoring and avoiding overreaction to single samples.

5. The interview with Andrea Dunbar highlighted the trade-offs of deploying CNNs in edge AI systems. How might reflecting on real-world deployment challenges help you decide how to benchmark success in your own BBO capstone project?

It made me benchmark success beyond peak score. Edge AI emphasizes reliability, efficiency, and repeatability under constraints; I apply the same logic here. For this capstone, success means: consistent improvement across rounds, robust performance across all eight functions, and decisions that are transparent and reproducible. That benchmark is more realistic than judging quality from one best-case query outcome.
