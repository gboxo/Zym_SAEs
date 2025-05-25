## MSA Steering




1) Collect samples from the models 
2) Score the samples with the oracle
3) Perform MSA on the sequences
4) Collect feature activations for all the sequences
5) Re-align the features based on the MSA
6) For each MSA column-indexed features train a sparse linear probe to predict the activity
7) Select best performin columns and the corersponding features
8) Train a HMM profiler on the MSA
9) Use the profiler and the top-features at inference time for steering





