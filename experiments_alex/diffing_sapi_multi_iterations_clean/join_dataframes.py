import pandas as pd
import os
path = "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations/dataframes"

folders = [[f"dataframe_iteration{j}.csv" for j in range(i,i+5)] for i in range(0,29,5)]


for idx,folder in enumerate(folders):
    df_list = []
    for file in folder:
        df_list.append(pd.read_csv(os.path.join(path, file)))

    final_df = pd.concat(df_list)
    final_df = final_df[final_df["alntmscore"] > 0.5]
    os.makedirs("/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/joined_dataframes",  exist_ok=True)

    final_df.to_csv(os.path.join("/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/joined_dataframes",  f"dataframe_all_iteration{idx}.csv"), index=False)
