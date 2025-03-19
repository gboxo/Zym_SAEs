import torch
from transformers import AutoTokenizer, EsmForProteinFolding
import argparse
import os


##### Load the module ESM ######
print("Loading tokenizer and model")
tokenizer_esm = AutoTokenizer.from_pretrained("/home/woody/b114cb/b114cb23/models/esm_fold") # Download tokenizer
model_esm = EsmForProteinFolding.from_pretrained("/home/woody/b114cb/b114cb23/models/esm_fold")  # Download model
device_name = "cuda" if torch.cuda.is_available() else "cpu"
print("Moving the device to ", device_name)
device = torch.device(device_name)
model_esm = model_esm.to(device)

parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int)
parser.add_argument("--label", type=str)
args = parser.parse_args()
iteration_num = args.iteration_num
ec_label = args.label
ec_label = ec_label.strip()
    
    

model_esm.eval()


# Put sequences into dictionary
with open(f"/home/woody/b114cb/b114cb23/boxo/seq_gens/seq_gen_{ec_label}_iteration{iteration_num}.fasta", "r") as f:
    data = f.readlines()
print("Loading sequences")


sequences={}
for line in data:
    if '>' in line:
        name = line.strip()
        sequences[name] = str()  #! CHANGE TO corre
        continue
    sequences[name] = line.strip()

print("Number of sequences: ", len(sequences))


count = 0
error = 0

for name, sequence in sequences.items():
  print(f"Processing sequence {name}")
  try:
    count += 1  
    with torch.no_grad():
      output = model_esm.infer_pdb(sequence)
      torch.cuda.empty_cache()
      name = name[1:]
      name = name.split("\t")[0]
      os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/outputs/output_iterations{iteration_num}/PDB", exist_ok=True, parents=True)
      with open(f"/home/woody/b114cb/b114cb23/boxo/outputs/output_iterations{iteration_num}/PDB/{name}.pdb", "w") as f:
            f.write(output)
  except:
    error += 1
    
    print(f'Sequence {name} is processed. {len(sequences)-count} remaining!') 
    print(f"Number of errors: {error}")
    torch.cuda.empty_cache()
del model_esm