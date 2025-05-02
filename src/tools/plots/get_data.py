import pandas as pd
import pickle as pkl
import os




# ===== CLIPPING =====


df_clipping = pd.read_csv("clipping/activity_predictions_pos.csv")
mask_paths = os.listdir("clipping/mask_activations")
active_tokens_dict = {}
for mask_path in mask_paths:
    

    if mask_path.endswith(".pkl") and not "all" in mask_path:
        _,_,_,feature_id = mask_path.split("_")
        feature_id = int(feature_id.split(".")[0])

        with open(f"clipping/mask_activations/{mask_path}", "rb") as f:
            mask = pkl.load(f)
        active_tokens = {int(key.split("_")[1]):sum(val) for key,val in mask.items()}
        active_tokens_dict["clipping_feature_"+str(feature_id)] = active_tokens
    
    




# Convert dictionary to long format with columns for feature_id, index, and active token count
df_active_tokens = pd.DataFrame([
    {'name': feature_id, 'index': token_idx, 'active_count': count}
    for feature_id, tokens in active_tokens_dict.items()
    for token_idx, count in tokens.items()
])


# Merge the two dataframes on the name and index columns
df_merged_clipping = pd.merge(df_clipping, df_active_tokens, on=['name', 'index'], how='left')
df_merged_clipping = df_merged_clipping.fillna(-1)



# ===== ABLATION =====


df_ablation = pd.read_csv("ablation/activity_predictions_pos.csv")
mask_paths = os.listdir("ablation/ablation_activations")
active_tokens_dict = {}
for mask_path in mask_paths:
    

    if mask_path.endswith(".pkl") and not "all" in mask_path:
        _,_,_,feature_id = mask_path.split("_")
        feature_id = int(feature_id.split(".")[0])

        with open(f"ablation/ablation_activations/{mask_path}", "rb") as f:
            mask = pkl.load(f)
        active_tokens = {int(key.split("_")[1]):sum(val) for key,val in mask.items()}
        active_tokens_dict["ablation_feature_"+str(feature_id)] = active_tokens
    



# Convert dictionary to long format with columns for feature_id, index, and active token count
df_active_tokens = pd.DataFrame([
    {'name': feature_id, 'index': token_idx, 'active_count': count}
    for feature_id, tokens in active_tokens_dict.items()
    for token_idx, count in tokens.items()
])


# Merge the two dataframes on the name and index columns
df_merged_ablation = pd.merge(df_ablation, df_active_tokens, on=['name', 'index'], how='left')
df_merged_ablation = df_merged_ablation.fillna(-1)




# ===== STEER =====


df_steer = pd.read_csv("steering/activity_predictions_pos_new.csv")
df_steer["active_count"] = -1

# Concatenate the three dataframes

df_all = pd.concat([df_merged_clipping, df_merged_ablation, df_steer])


df_all.to_csv("all_data.csv", index=False)






