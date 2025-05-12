import os
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr

from torch.utils.data import DataLoader
from datasets import Dataset
import pickle as pkl


oracle_path2 = "/home/woody/b114cb/b114cb23/models/esm1v_t33_650M_UR90S_1"
checkpoint_path2 = "/home/woody/b114cb/b114cb23/Filippo/alpha_amylase_activity_predictor/LoRA_esm1v/Esm1v_GB1_finetuned.pth"

with open("/home/woody/b114cb/b114cb23/boxo/esm_lr/model.pkl", "rb") as f:
    lr  = pkl.load(f)




class SequenceDataset(Dataset):
    def __init__(self, tokenized_sequences):
        self.input_ids = torch.cat([seq["input_ids"] for seq in tokenized_sequences])
        self.attention_mask = torch.cat([seq["attention_mask"] for seq in tokenized_sequences])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }



def main():

    df_path = "/home/woody/b114cb/b114cb23/boxo/experimental_SAPI.csv"
    df = pd.read_csv(df_path)
    # ========== 1) Read all sequences (flatten multiple entries per file) ==========
    sequences = df["sequence"].tolist()
    activity = df["SAPI"].tolist()

    # ========== 2) Tokenize & build DataLoader ==========
    # We only need one tokenizer, so load first oracle
    tokenizer = AutoTokenizer.from_pretrained(oracle_path2)
    model = AutoModel.from_pretrained(
        oracle_path2,
        torch_dtype=torch.float16
    )
    # Tokenize all sequences in a batch
    tokenized = tokenizer(
        sequences,
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )


    print(tokenized["input_ids"])
    tensor_dataset = SequenceDataset([{
        "input_ids": tokenized["input_ids"][i : i+1],
        "attention_mask": tokenized["attention_mask"][i : i+1]
    } for i in range(len(sequences))])
    loader = DataLoader(tensor_dataset, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device).eval()
    preds1 = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Oracle1"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Get the hidden states from layer 33 (last layer)
            for i in range(len(batch["input_ids"])):
                layer_33_output = outputs.hidden_states[-1][i]
                mask = attention_mask[i]
                layer_33_output = layer_33_output[mask == 1]
                preds1.append(layer_33_output.mean(dim=0))
    # free GPU
    torch.cuda.empty_cache()

    preds1 = torch.stack(preds1)
    X = preds1.cpu().numpy()
    y = activity

    y_pred = lr.predict(X)
    print(pearsonr(y_pred, y))



    with open("/home/woody/b114cb/b114cb23/boxo/esm_lr/embs_dict_exp_SAPI.pkl", "rb") as f:
        embs_dict = pkl.load(f)








    # ========== 5) Build DataFrame & Save ==========
    df = pd.DataFrame({
        "name":        [r["name"]       for r in records],
        "index":       [r["index"]      for r in records],
        "prediction1": preds1,
    })

    output_path = config["paths"]["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions → {output_path}")

if __name__ == "__main__":
    main()


