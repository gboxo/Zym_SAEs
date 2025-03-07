# Protein GYM


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




