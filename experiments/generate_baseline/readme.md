# Experiment 1: Generate Baseline


We will benchmark different genrate functions in order to calibrate other experiments.


- `transformers` generate function with repetition penalty. **(t_p)**
- `transformers` generate function without repetition penalty. **(t_np)**
- `transformers_lens` generate function with repetition penalty. **(tl_p)**
- `transformers_lens` generate function without repetition penalty. **(tl_np)**


We will genrate using those functions and the score the results using the following metrics:

- Oracle
- Length


