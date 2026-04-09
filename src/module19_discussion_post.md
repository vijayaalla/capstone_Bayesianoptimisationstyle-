# Module 19 - Round 8 Submission + LLM Strategy Reflection Draft

## Part 1: Query submission (portal format)

function_1: 0.003096-0.015342  
function_2: 0.995960-0.302626  
function_3: 0.588952-0.743293-0.011421  
function_4: 0.324718-0.337697-0.334311-0.392052  
function_5: 0.185571-0.824153-0.948712-0.979760  
function_6: 0.069378-0.524118-0.199547-0.856631-0.025358  
function_7: 0.294301-0.118676-0.484343-0.325790-0.159802-0.931507  
function_8: 0.092632-0.264766-0.024995-0.265425-0.667367-0.051237-0.095473-0.442073

## Part 2: Reflection (Discussion board post)

1. Which prompt patterns (zero-shot, few-shot, etc.) did you use, and why? What changed when you simplified vs structured the prompt?

I used a structured few-shot style pattern. The simplified version was equivalent to ranking candidates from one global model score, while the structured version combined a full-context view, a finite-context view, uncertainty, disagreement, novelty, and output guardrails. The structured version was more stable and reduced brittle decisions from any single signal.

2. What temperature, top-k, top-p and max-tokens settings did you choose? How did they trade off coherence vs diversity? How did they affect your chosen query?

I used `temperature=0.65`, `top_k=96`, `top_p=0.88`, and `max_tokens=64`, which translated into a focused context window of 8 observations in the prompt-like layer. The lower temperature improved coherence by keeping selection near the best-scoring candidates, while top-k and top-p preserved enough diversity that I did not collapse to one deterministic choice too early. This produced queries that were still exploratory, but more controlled than pure argmax or random sampling.

3. Did token boundaries or unusual input strings affect the model’s behaviour? When did you notice token count limits or truncation influencing the outputs? If no such cases were observed, explain how you checked for those cases.

I did not observe literal truncation failures because I kept the context deliberately short and the output format fixed. I did check for edge-like strings near `0.000000` or `0.999999`, because those can behave like formatting artefacts in a string-based workflow. To reduce that risk, I added a mild penalty for boundary-heavy candidates and logged the edge count in the prompt report.

4. With 17 data points, what limitations did you encounter, such as prompt overfitting, attention focusing on irrelevant context or diminishing returns from longer inputs?

The main limitation was prompt overfitting to recent or high-performing context. A longer prompt window does not always help, because attention can drift toward less relevant history. I also saw diminishing returns from adding more context once the core useful patterns were already represented. That is why I kept a finite context window instead of assuming more history automatically improves decisions.

5. Which strategies did you try to reduce hallucinations? For example, did you use tighter instructions, retrieval of prior relevant information or constrain the output format?

I used tighter structure, retrieved only a focused subset of prior information, constrained the output format, and combined multiple scoring signals instead of trusting one raw prediction. In practice, this reduced hallucination-like behavior by making the pipeline more explicit about what information mattered and what outputs were valid.

6. In future rounds, how would you scale your prompting and decoding strategies when working with larger data sets or more complex LLMs?

I would scale by using adaptive context windows, retrieval of the most relevant past observations instead of longer raw history, and temperature schedules that respond to uncertainty. For larger models, I would also use stricter schema-constrained outputs and cache summarized context so compute stays manageable.

7. How did these design choices for prompts and decoding help you think like a practitioner balancing exploration, risk and computational constraints in a black-box setting with incomplete information?

They made me think of prompts and decoding as resource-allocation choices, not just text settings. I had to decide how much context to trust, how much randomness to allow, and how many safeguards to impose before accepting a query. That is very similar to real ML practice, where good decisions depend on balancing signal quality, uncertainty, cost, and operational reliability.
