


## Finetuned ZC on DMS data


Given thtat using oracles for sequences that are longer than 425 is really dificult, we will try to get results by using a model finetuned on DMS data that hopefully will generaet sequences that are of similar length as the DMS data. This should make the oracles work better.



### Sanity Checks

Sanity Checks for the FT model.


1) We need to check that the sequences that the model generates are of similar length.
2) We need to check that for a given oracles the generated sequences have plausible activities (that is the distribution should be similar to the DMS data)


**Sequence Lengths**

count     78.000000
mean     424.256410
std        3.401436
min      399.000000
25%      425.000000
50%      425.000000
75%      425.000000
max      427.000000


**Activities**

**Oracle 2**

       prediction1  prediction2  mean_prediction
count    78.000000    78.000000        78.000000
mean      1.549887     1.523912         1.536900
std       0.971367     0.788917         0.867239
min       0.503019     0.485729         0.510395
25%       0.987460     1.003276         0.986991
50%       1.305222     1.286478         1.296822
75%       1.905340     1.896809         1.936649
90%       2.288894     2.409976         2.277864
95%       2.594236     2.766216         2.743354
99%       4.429833     4.041548         4.235690
max       8.296847     5.728742         7.012795


**Prediction MLP Regressor**
       prediction
count   78.000000
mean     1.215183
std      0.818345
min      0.330808
25%      0.842921
50%      0.968832
75%      1.146965
max      6.475057


**DMS Regressor**
       prediction
count   78.000000
mean     1.165889
std      1.118717
min     -0.004490
25%      0.651701
50%      1.002378
75%      1.355663
max      9.119850










