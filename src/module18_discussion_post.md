# Module 18 - Round 7 Submission + Hyperparameter Reflection Draft

## Part 1: Query submission (portal format)

function_1: 0.000069-0.992322  
function_2: 0.999999-0.999999  
function_3: 0.567864-0.748902-0.048521  
function_4: 0.416243-0.382873-0.372901-0.432460  
function_5: 0.021933-0.990981-0.919130-0.988763  
function_6: 0.721833-0.246550-0.734369-0.611496-0.000000  
function_7: 0.006908-0.129532-0.484908-0.031440-0.077654-0.992662  
function_8: 0.094469-0.178320-0.002054-0.061263-0.860762-0.819836-0.047864-0.719353

## Part 2: Reflection (Discussion board post)

1. Which hyperparameters did you choose to tune, and why did you prioritise them?

I prioritised hyperparameters that most directly affect generalisation and uncertainty quality: for SVR, `C`, `gamma`, and `epsilon`; for MLP, hidden-layer sizes, `alpha` (regularisation), learning rate, and early-stopping choice. I prioritised these because they control model flexibility, smoothness, and stability, which directly affects query quality in a low-data black-box setting.

2. How has hyperparameter tuning changed your query strategy compared to earlier rounds?

Earlier rounds relied more on fixed defaults and manual strategy shifts. This round, tuning made query selection evidence-driven: I compared tuned SVR/MLP candidates with cross-validated MAE, selected the better family per function, then generated queries with uncertainty-aware acquisition. As a result, I rely more on model predictions where fit is strong, but still preserve exploration where uncertainty remains high.

3. Which tuning method(s) did you apply (manual adjustment, grid search, random search, Bayesian optimisation, Hyperband), and what trade-offs did you notice?

I used random-search-style tuning plus bounded manual design of search ranges. Random search gave better coverage than grid search for continuous hyperparameters at lower computational cost. The trade-off is that it can miss narrow optima, while full Bayesian tuning would likely be more sample-efficient but more complex to implement and maintain in this weekly workflow.

4. As your data set grows to 16 points, what limitations of your model become clearer through tuning (e.g. overfitting, irrelevant features, diminishing returns)?

Tuning made three limitations clearer. First, some high-capacity settings improved in-sample fit but degraded cross-validation performance, indicating overfitting. Second, several functions preferred smoother configurations, suggesting some dimensions contribute weak signal at this stage. Third, I saw diminishing returns from extra complexity: beyond a threshold, more tuning effort and larger models produced only small query-quality gains.

5. How might you apply hyperparameter tuning techniques to larger data sets in future rounds of the BBO capstone project submissions or more complex models in future ML/AI projects?

For larger datasets, I would move from random search toward Bayesian hyperparameter optimisation with early-stopping allocation (Hyperband-style) to spend compute more efficiently. I would also separate tuning by regime (low vs high dimensional functions), use stronger validation tracking, and automate retuning triggers so hyperparameters update only when model drift is detected.

6. How does tuning in this black-box set-up prepare you to think like a professional ML/AI practitioner in real-world contexts with incomplete information?

It builds practical decision discipline. I have to make model and tuning choices under uncertainty, justify trade-offs between performance and cost, and avoid over-trusting noisy signals. This mirrors real ML work, where perfect information is rare and credibility depends on reproducible experimentation, transparent reasoning, and consistent improvement rather than one-off results.
