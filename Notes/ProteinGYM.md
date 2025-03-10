#Protein GYM


Predicting the effects of mutations in proteins is crucial to understand genetic disease or design novel proteins.

**ProteinGYM**

- 250 DMS assays
    - spanning 2.7 million of mutated sequences
    - 200 protein families
    - Different functions
    - Different taxa
    - Different depth of homologous sequences
- Clincal dataset (high quality expert annotations)
    - 65k indel substitutions


The ability to manipulate and optimize known sequences and structures represents a great opportunity.

The design of novel, functionally optimized proteins presents several challenges, it begins with learning a mapping between protein sequences and resulting properties.

This mapping is often conceptualized as a **fitness landscape**.


```
Models that learn the protein fitness landscape have been shown to be effective at predicting the effects of mutations.
But the ability to tell apart the sequences that are functional or not is also critical to protein engineering efforts.
```


### Mutation types


**Substitution**

Change in one aa in a protein sequence, this can have varied effects. 
The substitution can be **conservative** (the new aa shares properties with the old one) or **non conservative**


**Indels**

Correspond to insertions or deletions of aa in protein sequences. Indels can affect protein fitness in similar ways to substitution, but they can also have profound impacts on protein structure by altering the protein backbone.


```
For instance, the majority of models trained on Multiple Sequence Alignments are typically unable to score indels due to the fixed coordinate system they operate within (see § 4).
Furthermore, when dealing with probabilistic models, comparing relative likelihoods of sequences with different lengths results in additional complexities and considerations
```


### Dataset types



**Deep Mutational Scanning assays**

It's challenging to isolate ta singular, measurable property that reflects the key aspects of fitness for a given protein.

*We prioritized assays where the experimentally measured property for each mutant protein is likely to represent the role of the protein in organismal fitness.*

- Functional properties:
    - Ligand binding
    - Aggregation
    - Thermosolubility
    - Viral replication
    - Drug Resistance
- Protein families
    - Kinases
    - Ion Channel proteins
    - G-protein coupled receptors
    - Polymerases
    - Transcription factors
    - Tumor suppressors


**Clinical Dataset**

Annotated reports of human genetic variations and associated phenotypes with relevant supporting evidence.

### Zero Shot Benchmarks

**DMS assays**

Due to the often non-linear relationships between protein function and organisms fitness, the Sperman's rank correlarion coefficient is appropiate.


When DMS measurements exhibit a bimodal profile, rank correlarions might not be the optimal choice.

We also measure:
- AUC ROC
- Matthews correlarion Coefficient (MCC) (compare model scores with binarized experimental measurements)


For certain goals like optimizing functional properties of designed proteins, it's more important  that a model is able to correctly identify the most functional protein variants, rather than a properly capture the overall distribution of all assayed variants. Thus, we also calculate the Normalized Discounted Cumulative Gains (NDCG), which up-weights a model if it gives scores to sequences with the highest DMS value.



### Supervised Benchmarks

**Greater care should be dedicated to mitigating  overfitting risks, as observations in biological datasets might not be independant.**


**Three types of cross validation**

1) Random Scheme: Each mutation is randomly assigned to one of 5  different folds.
2) Contiguous Scheme: We split the sequence contiguously along its lengtg, in order to obtain 5 segments of contiguous positions, and assign mutations to each segment based on the position at which occurs.
3) Modulo scheme, we assign positions to each fold using the modulo operator to obtain 5 folds overall.

We report the Sperman's rank  correlation and MSE between predictions and experimental measurements.

```
A more challenging generalization task would
involve learning the relationship between protein representation (sequence, structure, or both) and function using only a handful of proteins, and then extrapolating at inference time to protein families not encountered during training. This setting may be seen as a hybrid between the zero-shot and supervised regimes – closer to zero-shot if we seek to predict different properties across families, and closer to the supervised setting if the properties are similar (eg., predicting the thermostability of proteins with low sequence similarity with the ones in the training set). While this study does not delve into these hybrid scenarios, the DMS assays in ProteinGym can facilitate such analyses.
```










