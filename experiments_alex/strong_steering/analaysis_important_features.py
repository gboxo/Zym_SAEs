import pandas as pd
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

dms_path = "/home/woody/b114cb/b114cb23/boxo/strong_steering/latent_scoring_dms/latent_scoring_FT/saved_features"
base_path = "/home/woody/b114cb/b114cb23/boxo/strong_steering/latent_scoring_base/saved_features"


with open(os.path.join(base_path, "ablation_cv_results_summary.pkl"), "rb") as f:
    dms_data = pkl.load(f)

with open(os.path.join(base_path, "ablation_cv_results_summary.pkl"), "rb") as f:
    base_data = pkl.load(f)






final_dict = {}
for name,path in [("dms", dms_path), ("base", base_path)]:
    for file in os.listdir(path):
        print("============")
        print(name)
        print(file)
        if "ablation" in file:
            type = "ablation"
            id = name+"_ablation"
        elif "diffmeans" in file:
            type = "diffmeans"
            min_activity = file.split("_")[-1].split(".pkl")[0]
            id = name+"_diffmeans_"+min_activity
        elif "sparselr_features" in file:
            type = "sparselr"
            min_activity = file.split("_")[-1].split(".pkl")[0]
            id = name+"_sparselr_"+min_activity
        else:
            continue

        if file.endswith(".pkl"):
            with open(os.path.join(path, file), "rb") as f:
                data = pkl.load(f)
            final_dict[id] = data
            



print("DIFF MEANS")

base_diffmeans = [key for key in final_dict.keys() if "base_diffmeans" in key]
dms_diffmeans = [key for key in final_dict.keys() if "dms_diffmeans" in key]


for k1 in base_diffmeans:
    f1 = final_dict[k1]["features"]
    for k2 in dms_diffmeans:
        f2 = final_dict[k2]["features"]
        if len(set(f1).intersection(set(f2))) > 0:
            print(set(f1).intersection(set(f2)))
            print(len(set(f1).intersection(set(f2))))
            print("==================")




#"Features: 10353"


print("SPARSE LR")
base_sparselr = [key for key in final_dict.keys() if "base_sparselr" in key]
dms_sparselr = [key for key in final_dict.keys() if "dms_sparselr" in key]


for k1 in base_sparselr:
    f1 = final_dict[k1]["features"]
    for k2 in dms_sparselr:
        f2 = final_dict[k2]["features"]
        if len(set(f1).intersection(set(f2))) > 0:
            print(set(f1).intersection(set(f2)))
            print(len(set(f1).intersection(set(f2))))
            print("==================")

#"Features: 11386"





dms_ablation = final_dict["dms_ablation"]
base_ablation = final_dict["base_ablation"]

dms_ablation_dict = {}
for _,row in dms_ablation.iterrows():
    id = str(row["min_activity"]) + "_" + str(row["min_rest_fraction"])
    selected_features = row["selected_features"]
    for idx,feat in enumerate(selected_features):
        dms_ablation_dict[id+"_"+str(idx)] = feat 
base_ablation_dict = {}
for _,row in base_ablation.iterrows():
    id = str(row["min_activity"]) + "_" + str(row["min_rest_fraction"])
    selected_features = row["selected_features"]
    for idx,feat in enumerate(selected_features):
        base_ablation_dict[id+"_"+str(idx)] = feat 

all_intersections = []
for k1 in dms_ablation_dict.keys():
    f1 = dms_ablation_dict[k1].tolist()
    for k2 in base_ablation_dict.keys():
        f2 = base_ablation_dict[k2].tolist()
        
        if len(set(f1).intersection(set(f2))) > 0:
            print(k1,k2)
            print(set(f1).intersection(set(f2)))
            print(len(set(f1).intersection(set(f2))))
            print("==================")
            all_intersections.append(set(f1).intersection(set(f2)))



from collections import Counter

# Convert sets to tuples so they are hashable
all_intersections_flat = [item for sublist in all_intersections for item in sublist]
counter = Counter(all_intersections_flat)

for k,v in counter.items():
    print(k,v)


"""

15224 200
8548 130
13039 80
10436 36
4056 45
1204 29
6186 29
1938 70
14739 2
8885 14
13800 310
10345 31
4263 6
6929 30
8179 60
8721 125
190 105
11572 63
7596 21
8601 30
7943 24
1781 2

"""

