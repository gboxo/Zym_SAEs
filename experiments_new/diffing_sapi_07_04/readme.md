# Diffing Experiments


This will be a series of Diffing experiments on the model `/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard`.

## First Stage


First will run experiment to generate data and train SAE will be separated in several steps:
1. Data gathering
2. Data generation
    - Scoring
    - Folding
    - Sequence alignment
3. Create a dataset with the data (if generated also include it)
4. Use the extracted data to create a dataframe for further analysis
5. Use the datasets to train iterative SAEs on the generated sequences 


## Second Stage

In this second stage we will analyze the resulting SAEs from the previous step:

1. For all the SAEs compute the following weight focus metrics:
    - Cosine Similarity with the base (decoders)
    - The norm of the encoder/decoder
    - The pariwise cosine similarity

2. Get some metrics for the feature firings:
    - Distribution of firings for each feature
    - Percentage of firings (at least 1 time in the sequence)
    - Collect the ranks of the firing positions
    - Find which features only appear in the prompt

3. Evaluate each SAE in the eval part of the dataset: 
    - L2 loss
    - Sparsity
    - Dead features
    - CE Loss 
    - Loss over context length

4. Do some light anotation of the SAE features with respect to **key metrics**:
    - **Key Metrics** 
        - pLDDT
        - Sequence Alignment
        - Oracle activity 
    - Use sparse probes to get important features to predict `high` and `low` instances of the metrics 
    - Use difference in mean probes to get important features to predict `high` and `low` instances of the metrics 
    - Maybe try to annptate as much features as possible by iteratively fitting probes and zeroing rows of the activations

5. Correlate the important features with previous metrics like:
    - Important features vs cosine similarity
    - Important features vs firing rate
    - Important features vs norm of the encoder
    - Important features vs prevalence 
    - For different threshods of activity (connected scatterplot) 
        - This would be for different threholds of activity, the features that best predcit

6. Compare the importance of features along the following axis:
    - Intra stage (e.g. M4D4 vs M0D4)
    - Inter stage (e.g. M4D4 vs M5D5)
    - Ineter metric comparison:
        - Which properties have features that predict high plddt vs high activity
            - When they are shared
    - Change in importance vs change in CS/threshold/etc

7. Gather information about the evolution of the features across iterations
    - A latent that started existing but then became a dead feature
    - When a latent changed wrt previous iteration
    - When a latent changed wrt previous iteration and then changed back
    - Evolution of the thresholds/norm/firing rate
    


## Third Stage

In this stage we will be evaluating the fitness of the features with causal experiments

There will be mainly two evaluations **ablation experiments** and **steering experiments** 


**With the important features gathered in the previous step we will perform ablation and steering experiments to assess the fitness of the feature wrt to the key metrics.**


- Ablation experiments will consist of zero-ablating one or more features during generation (this will use the DPO model of  a given iteration)
- Steering experiments will consist of steering with one or more features during generation (this will use the base model)


This experiments will be repeated for:
- Each type of experiment {ablation, steering}
- Each feature or group of features
- Each iteration
- Each one of the 2 branches M0DX or MXDX
- Each key metric











