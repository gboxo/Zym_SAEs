import pandas as pd
import os
import argparse
from src.tools.data_utils.data_utils import load_config
from argparse import ArgumentParser





def get_plddt_dict(plddt_path, files):
    """
    Compute the average pLDDT for each PDB file in 'files'.

    The pLDDT value is assumed to be stored in columns 61-66 of any ATOM line.
    Returns a dictionary mapping each filename to its average pLDDT (or None if not found).
    """
    plddt_dict = {}
    for file in files:
        pdb_file = os.path.join(plddt_path, file)
        with open(pdb_file, 'r') as f:
            plddt_vals = []
            for line in f:
                if line.startswith("ATOM"):
                    # Extract the pLDDT value; typically in columns 61-66.
                    try:
                        plddt = float(line[60:66].strip())
                        plddt_vals.append(plddt)
                    except ValueError:
                        continue
            # Compute the average pLDDT if there are any ATOM lines, else assign None.
            if plddt_vals:
                avg_plddt = sum(plddt_vals) / len(plddt_vals)
                plddt_dict[file.strip(".pdb")] = avg_plddt
            else:
                plddt_dict[file.strip(".pdb")] = None
    return plddt_dict

def main(iteration_num, label, out_path, tm_score_path, sequences_path, activity_path, plddt_path):

    
    df_tm_score = pd.read_csv(tm_score_path, sep="\t", header=None)
    df_tm_score.columns = ["query", "target", "alntmscore", "qtmscore", "ttmscore", "alnlen"]
    df_tm_score["label"] = df_tm_score["query"]

    df_alntmscore = df_tm_score[["label", "alntmscore"]]
    print(df_alntmscore.head())

    
    # Load the sequences

    with open(sequences_path, "r") as f:
        data = f.read()
        data = data.split(">")
        data = [elem for elem in data if len(elem)>0]
        ids = [elem.split("\n")[0] for elem in data]
        ids = [elem.split("\t")[0] for elem in ids]
        sequences = [elem.split("\n")[1] for elem in data]
        df_sequences = pd.DataFrame({"label": ids, "sequence": sequences})

    
    # Load the activity predictions
    df_activity = pd.read_csv(activity_path, sep="\t|,", header=None)
    df_activity.columns = ["label", "prediction1", "prediction2"]
    # Remove the > in the label column
    df_activity["label"] = df_activity["label"].str.replace(">", "")

    # Load the pLDDT
    files = os.listdir(plddt_path)
    files = [file for file in files if file.endswith(".pdb")]

    # Get average pLDDT for each pdb file
    plddt_dict = get_plddt_dict(plddt_path, files)
    df_plddt = pd.DataFrame(plddt_dict.items(), columns=["label", "pLDDT"])

    df_merged = pd.merge(df_activity, df_plddt, on="label", how="inner")
    df_merged = pd.merge(df_merged, df_sequences, on="label", how="inner")
    df_merged = pd.merge(df_merged, df_alntmscore, on="label", how="inner")
    # Further processing of df and plddt_dict goes here.
    df_merged.to_csv(out_path, index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    args = parser.parse_args()
    cfg_path = args.cfg_path
    config = load_config(cfg_path)




    iteration_num = config["iteration_num"]
    label = config["label"]
    out_path = config["paths"]["out_path"].format(iteration_num)
    tm_score_path = config["paths"]["tm_score_path"].format(label, iteration_num)
    sequences_path = config["paths"]["sequences_path"].format(label, iteration_num)
    activity_path = config["paths"]["activity_path"].format(iteration_num)
    plddt_path = config["paths"]["plddt_path"].format(iteration_num)
    main(iteration_num, label, out_path, tm_score_path, sequences_path, activity_path, plddt_path)
