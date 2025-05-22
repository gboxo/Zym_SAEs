# Final Experiments for the paper




We will be using the **Finetuned Model trained on the DMS**, after that the model will go trough several rounds of DPO with the following reward components:
- Activity
- TM Score
- Length Penalty


## Next steps 

**Initial sanity check**


We will embed the generated sequences with noelia_ft model, (residual stream) for all the layers, last position, for all the iterations



![Embeddings visualization](../../../boxo/dpo_noelia/tsne_last_token_grid.png)





**Fit a multiclass regression**

![Embeddings visualization](../../../boxo/dpo_noelia/acc_per_layer.png)


<table>
  <thead>
    <tr>
      <th>layer</th>
      <th>mean_accuracy</th>
      <th>std_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>0.12083208135906623</td><td>0.008286921071043708</td></tr>
    <tr><td>1</td><td>0.305999460688008</td><td>0.015027396315749519</td></tr>
    <tr><td>2</td><td>0.31620170268500325</td><td>0.011768777994773157</td></tr>
    <tr><td>3</td><td>0.32679841288185213</td><td>0.018018886168783107</td></tr>
    <tr><td>4</td><td>0.36445548749951845</td><td>0.02482701655303051</td></tr>
    <tr><td>5</td><td>0.37857929812396474</td><td>0.025100481033261256</td></tr>
    <tr><td>6</td><td>0.4040756577680188</td><td>0.012444676057017066</td></tr>
    <tr><td>7</td><td>0.40917523787511073</td><td>0.01059866912353483</td></tr>
    <tr><td>8</td><td>0.41388343156516044</td><td>0.009970998332422371</td></tr>
    <tr><td>9</td><td>0.4189814707808467</td><td>0.014419497936373331</td></tr>
    <tr><td>10</td><td>0.42408182133364153</td><td>0.012283709639596543</td></tr>
    <tr><td>11</td><td>0.4232982780538541</td><td>0.01099965386605695</td></tr>
    <tr><td>12</td><td>0.42368812357949076</td><td>0.012340072129933785</td></tr>
    <tr><td>13</td><td>0.42604337609306986</td><td>0.010675057671791603</td></tr>
    <tr><td>14</td><td>0.4232975076081513</td><td>0.011848029749881448</td></tr>
    <tr><td>15</td><td>0.4252598328132825</td><td>0.010998765149116914</td></tr>
    <tr><td>16</td><td>0.42682923070996565</td><td>0.012334738157193744</td></tr>
    <tr><td>17</td><td>0.4264393851843291</td><td>0.011382094422257202</td></tr>
    <tr><td>18</td><td>0.42800724218960673</td><td>0.011042820754515747</td></tr>
    <tr><td>19</td><td>0.4331075927424015</td><td>0.009233199704540177</td></tr>
    <tr><td>20</td><td>0.42800647174390394</td><td>0.008758985860903602</td></tr>
    <tr><td>21</td><td>0.4272236989098193</td><td>0.007427523744856485</td></tr>
    <tr><td>22</td><td>0.4264378442929234</td><td>0.00844916262406297</td></tr>
    <tr><td>23</td><td>0.42643707384722057</td><td>0.011092958832851003</td></tr>
    <tr><td>24</td><td>0.427220617127008</td><td>0.00970538815603476</td></tr>
    <tr><td>25</td><td>0.4260433760930698</td><td>0.010889006678573532</td></tr>
    <tr><td>26</td><td>0.42800416040679534</td><td>0.010352483410051686</td></tr>
    <tr><td>27</td><td>0.4327115836511422</td><td>0.011809663708175882</td></tr>
    <tr><td>28</td><td>0.4354589930274663</td><td>0.0098201628249753</td></tr>
    <tr><td>29</td><td>0.43467313841057054</td><td>0.010999350691483426</td></tr>
    <tr><td>30</td><td>0.43624176586155095</td><td>0.012234901906333017</td></tr>
    <tr><td>31</td><td>0.43702685003274394</td><td>0.011186377702543677</td></tr>
    <tr><td>32</td><td>0.43624176586155095</td><td>0.01138861334877258</td></tr>
    <tr><td>33</td><td>0.43702685003274394</td><td>0.010549607369667329</td></tr>
    <tr><td>34</td><td>0.43624330675295653</td><td>0.009208514076194413</td></tr>
    <tr><td>35</td><td>0.43624253630725374</td><td>0.011482661704470123</td></tr>
    <tr><td>36</td><td>0.42800724218960673</td><td>0.009304454796859714</td></tr>
  </tbody>
</table>

**Data Generation**

We need to prepare the data that will be used in the interpretability experiments, that will involve the following steps.


1) Generate ~1000 sequences for each iteration
2) Do some light clustering (or whatever to get a sense of how much they change)


There's some amount of collapse in the models, many repeated sequences and sampling difficulty.

**However, the model at iteration3 is eassy to sample and has high enough activity to be interesting.**


We will apply the diffing procedure between the base model an the iteration3



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



![Embeddings visualization](../../../boxo/dpo_noelia/cosine_similarity.png)


So the differences are there, but are not very big.




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

It's difficult to get enough diverse data.































