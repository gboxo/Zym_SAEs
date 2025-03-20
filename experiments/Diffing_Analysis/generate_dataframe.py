import pandas as pd
import os
import argparse

def get_plddt_dict(plddt_path, files):
    """
    Compute the average pLDDT for each PDB file in 'files'.

    The pLDDT value is assumed to be stored in columns 61-66 of any ATOM line.
    Returns a dictionary mapping each filename to its average pLDDT (or None if not found).
    """
    plddt_dict = {}
    for file in files:
        print(file)
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

def main(iteration_num, label):

    
    # Load the sequences

    sequences_path = f"/home/woody/b114cb/b114cb23/boxo/seq_gens/seq_gen_3.2.1.1_iteration{iteration_num-1}.fasta"
    with open(sequences_path, "r") as f:
        data = f.read()
        data = data.split(">")
        data = [elem for elem in data if len(elem)>0]
        ids = [elem.split("\n")[0] for elem in data]
        ids = [elem.split("\t")[0] for elem in ids]
        sequences = [elem.split("\n")[1] for elem in data]
        df_sequences = pd.DataFrame({"label": ids, "sequence": sequences})

    
    # Load the activity predictions
    activity_path = f"/home/woody/b114cb/b114cb23/boxo/activity_predictions/activity_prediction_iteration{iteration_num-1}.txt"
    df_activity = pd.read_csv(activity_path, sep="\t|,", header=None)
    df_activity.columns = ["label", "prediction1", "prediction2"]
    # Remove the > in the label column
    df_activity["label"] = df_activity["label"].str.replace(">", "")

    # Load the pLDDT
    pLDDT_path = f"/home/woody/b114cb/b114cb23/boxo/outputs/output_iterations{iteration_num-1}/PDB/"
    files = os.listdir(pLDDT_path)
    files = [file for file in files if file.endswith(".pdb")]

    # Get average pLDDT for each pdb file
    plddt_dict = get_plddt_dict(pLDDT_path, files)
    df_plddt = pd.DataFrame(plddt_dict.items(), columns=["label", "pLDDT"])

    df_merged = pd.merge(df_activity, df_plddt, on="label", how="inner")
    df_merged = pd.merge(df_merged, df_sequences, on="label", how="inner")
    # Further processing of df and plddt_dict goes here.
    df_merged.to_csv(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/dataframe_iteration{iteration_num-1}.csv", index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int)
    parser.add_argument("--label", type=str)
    args = parser.parse_args()
    iteration_num = args.iteration_num
    label = args.label
    main(iteration_num, label)
