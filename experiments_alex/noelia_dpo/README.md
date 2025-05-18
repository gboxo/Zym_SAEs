# Final Experiments for the paper




We will be using the **Finetuned Model trained on the DMS**, after that the model will go trough several rounds of DPO with the following reward components:
- Activity
- TM Score
- Length Penalty


## Next steps 


**Data Generation**

We need to prepare the data that will be used in the interpretability experiments, that will involve the following steps.


1) Generate ~1000 sequences for each iteration
2) Do some light clustering (or whatever to get a sense of how much they change)



**Due Diligence**


*We nee to make sure we are filtering for length*

We would want to do some test to esnure that each iterations creates miningully differnt sequences, for that maybe the easiest thing is to just embed them and do some dimensioanlity reduction we want to see 2 things.


1) The inter iteration variance is higher than the intra iteration variance(Very important)
2) Hopefully there's some step-wise trend on the path from sequential iterations, meaning that two iterations that are closer in terms of iterations are more similar than sequences that are further apart.



- If we do see this trend, we should try to do MSA on a global level and see what the length of the aligned sequences.


- If we don't see this trend, and sequences are all over the place idk what should we do.



**Train SAEs**


*Starting from the SAE FT on the DMS data*


Depending on the previous step results we will either train a SAE for each iteration or group various iterations, and the FT the SAE on the data.


We want to see a handful of features changing (we can try several schemes to get to that result). We will try the obvious things to get few latents to change in freq/norm/direction if they don't work we will proceed regardless.



**Diffing**


We will try to track features that change in direction/frequency/norm over the iterations, we would want to see some stricking results. If they don't seem to appear, we will focus more on the KL's maybe.


If we do see a clear trend of some features changing over the training that's good, to make it more likely to see we can cherrypcik some thresholds to filter out noise.

If the trend is noisy we can make some sensible checks.
- Try to align the features (maybe the same features appear but they are not aligned)
- Train variuos saes with KFold cross validation, to make sure the same feature appear regardless of the actual datapoints.
- If we mix the data do the the union of the expected features emerge.






**POTTS**



The importance of this will likely depend on the success of the SAEs.
If there's enough time to explore this idea, it would be really cool to try to understand the distribution shift of the model trough the lensses of POTTS model trained on sequences of different iterations.

































