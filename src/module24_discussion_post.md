# Module 24 - Final Round Submission + RL Reflection Draft

## Part 1: Query submission (portal format)

function_1: 0.889382-0.900099  
function_2: 0.995535-0.976503  
function_3: 0.277470-0.226389-0.445302  
function_4: 0.456075-0.382270-0.412261-0.382384  
function_5: 0.121982-0.888915-0.992806-0.955957  
function_6: 0.609474-0.540093-0.378775-0.934945-0.032678  
function_7: 0.000000-0.480719-0.243063-0.148102-0.294101-0.824436  
function_8: 0.184821-0.130605-0.000000-0.036614-0.442099-0.687638-0.349585-0.829199

## Part 2: Reflection (Discussion board post)

1. How has your understanding of the exploration-exploitation trade-off evolved with increasing data? How did you balance taking risks by exploring new strategies versus exploiting those that had proven effective?

My understanding changed from “explore widely because I know very little” to “exploit when the evidence is repeated, but keep a measured check on alternatives.” Early rounds needed broader exploration because uncertainty dominated. In the final round, the exploit arm won across functions, which shows that repeated evidence had become strong enough to justify more local refinement. I still compared explore and PCA-guided alternatives so that exploitation did not become blind habit.

2. As your data set expanded, how did the nature of feedback influence your optimisation process? Can you relate this to how RL agents adjust their reward expectations or update their Q-values?

Each new round acted like a reward update. Strong outputs increased my confidence in certain regions and in certain policy styles, while weak outputs reduced their value. Over time, this worked like a simple value update process: regions or strategies with repeated positive feedback became more attractive, and those with inconsistent outcomes lost priority. That is similar to how an RL agent updates expected rewards or Q-values after observing the consequences of an action.

3. Think about the AlphaGo Zero case study. How does your iterative improvement reflect the idea of self-play or autonomous learning? Did your process resemble model-free learning (trial and error) or model-based planning (anticipating future outcomes)?

My process was not self-play in the literal game sense, but it did resemble autonomous learning because each round’s outputs directly informed the next round without external labels. Early on, it was closer to model-free trial and error because I was learning mostly from what worked and what failed. Later rounds became more model-based because surrogate models were used to anticipate future outcomes before committing to the next query.

4. How could RL strategies be applied to enhance exploration design, efficiency or convergence speed in real-world optimisation tasks?

RL strategies could improve this kind of optimization by treating strategy choice itself as a policy problem. A contextual bandit could choose between exploit, explore, and structure-aware query arms. UCB- or Thompson-style rules could manage uncertainty more formally. More broadly, RL ideas help define when to keep searching, when to converge locally, and how to use feedback efficiently rather than restarting the search logic each round.
