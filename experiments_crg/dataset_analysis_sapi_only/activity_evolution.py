import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd




"""
Get the activity evolution of the sequences in the different iterations
"""

path = "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_07_04/dataframes/"
files = os.listdir(path)



all_dataframes = []
for file in files :
    n = file.split('_')[1].split('.')[0].replace('iteration', '')
    df = pd.read_csv(os.path.join(path, file))
    df['iteration'] = int(n)
    all_dataframes.append(df)



# Add the iteration number to the dataframe

df = pd.concat(all_dataframes, ignore_index=True)
# Sort the rows by iteration
df.sort_values(by='iteration', inplace=True)
path = "/home/woody/b114cb/b114cb23/boxo/dataset_analysis_sapi_only/activity_evolution/"
os.makedirs(path, exist_ok=True)

# Violing plot of the distribution of activity in each iteration

plt.figure(figsize=(15, 6))
sns.violinplot(data = df, x = "iteration", y = "prediction1")
plt.xlabel("Iteration")
plt.ylabel("Activity Oracle 1")
plt.title("Activity Evolution")
plt.savefig(os.path.join(path, "activity_evolution_oracle1.png"))
plt.close()

# Violing plot of the distribution of activity in each iteration

plt.figure(figsize=(15, 6))
sns.violinplot(data = df, x = "iteration", y = "prediction2")
plt.xlabel("Iteration")
plt.ylabel("Activity Oracle 2")
plt.title("Activity Evolution")
plt.savefig(os.path.join(path, "activity_evolution_oracle2.png"))
plt.close()



# Violing plot of the distribution of tm scores in each iteration

plt.figure(figsize=(15, 6))
sns.violinplot(data = df, x = "iteration", y = "alntmscore")
plt.xlabel("Iteration")
plt.ylabel("TM Score")
plt.title("TM Score Evolution")
plt.savefig(os.path.join(path, "tm_score_evolution.png"))
plt.close()



# Violing plot of the distribution of pLDDT scores in each iteration
plt.figure(figsize=(15, 6))
sns.violinplot(data = df, x = "iteration", y = "pLDDT")
plt.xlabel("Iteration")
plt.ylabel("pLDDT")
plt.title("pLDDT Evolution")
plt.savefig(os.path.join(path, "plddt_evolution.png"))
plt.close()