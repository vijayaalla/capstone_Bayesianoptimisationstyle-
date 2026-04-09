# Module 19.1 Reflection: LLM-Centred Strategy and 17-Point Decision Making

This round, I translated LLM design ideas into the query-selection layer of my BBO pipeline rather than treating them as purely theoretical concepts. The main changes were:
- structured prompt-like scoring instead of one raw acquisition value
- a finite context window to mimic attention limits
- decoding controls to balance stability and diversity
- stricter output constraints to reduce formatting and edge-case failures

## Prompt design

I used a structured few-shot style pattern. Instead of relying on one global model view, I combined:
- a full-context surrogate view
- a focused-context view built from a limited recent window
- explicit uncertainty, disagreement, novelty, and format checks

This made the decision process more stable than a simplified single-score prompt equivalent.

## Decoding controls

I set:
- `temperature = 0.65`
- `top_k = 96`
- `top_p = 0.88`
- `max_tokens = 64` which translated to `context_window = 8`

These settings kept the final choice coherent while still allowing some diversity within a high-quality shortlist. Lower temperature reduced erratic jumps; top-k and top-p prevented the search from collapsing to one deterministic candidate too early.

## What became clearer at the 17-point stage

Three issues stood out:
- **Prompt overfitting**: strongly structured prompts can overemphasize recent strong patterns.
- **Attention dilution**: longer context does not always help; too much history can distract from the most relevant information.
- **Diminishing returns**: adding more prompt context beyond a moderate window gave less benefit than improving selection quality inside the window.

## Token and hallucination controls

I did not see truncation failures in this local setup because I explicitly capped context size and enforced short formatted outputs. I also checked for edge-case strings near boundaries and applied mild penalties so token-like formatting artefacts would not dominate ranking.

To reduce hallucination-like behavior, I used:
- tight structured instructions in the scoring layer
- retrieval of only the most relevant subset of context
- constrained decoding
- strict output formatting checks

## Overall effect on BBO strategy

LLM-centred thinking made my BBO process more deliberate about context management. I now ask not only “what is the best predicted point?” but also “what information deserves attention, how much randomness is useful, and where should formatting or context limits act as guardrails?” That is a useful professional mindset for working under incomplete information.
