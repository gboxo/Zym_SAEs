# Final Steering Experiments


## Setup



**Training**
- We will be using the Finetuned ZymCTRL found in `ZF_FT_alphaamylase_gerard`.
- We will finetune the SAE on the DMS dataset.


**Data Generation**

- We will generate 1000 sequences with FT_ZymCTRL .
- We will compute the expected activity of the generated sequences.
- We will compute the secondary structure of the generated sequences.
- We will fold the sequences with ESMFold.
- We will compute summary statistics of the generated sequences.


**Data Processing**




- We will generate a dataframe with the important things
- We will filter the dataframe by TM Score > 0.5
- We will filter the dataframe by sequence length > 400 <600
- We will gather 100 representative sequences from the generated data
    - We will perform sequence alignment with the DMS dataset.
    - We wil compute the secondary structure
    - Analyze conservation patterns
    - Functional site analysis
    - Hydrophobicity analysis
- For the 3d structure
    - We will compute the TM Score
    - We will compute the RMSD
    - We will compute the RMSF


**Latent Scoring**


```
We will perform latent scoring on 2 sets of sequences:
- The 1000 generated sequences (filtered)
- The 1000 sequences without filtering
- The 10000 DMS sequences
```

*If ther's time we will also perform latent scoring on samples generated with different temperatures*

### Ablation

- We will score the latents with the following methods:
    - Which features don't fire on sequences with activity higer than x
        - Do this for multiple x values and pick up the trend
        - Just select features not only not fire but also do fire on the other sequences

*We will also perform the the other way around trying to create low activity sequences*
*We will also perform the same anotation for TM scores below 0.5 if there's time*


### Clipping

For feature clipping we are interested in featues who's strength dictate the activity of the sequence.


### Steering

We will perform dense steering with the following methods:

- Mean Difference Probing
- With different strenths



