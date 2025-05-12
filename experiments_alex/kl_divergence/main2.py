import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
# KL Divergence
from torch.nn import functional as F
from src.utils import load_model, get_ht_model



base_model_path  = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"
dpo_model_path = "/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard/output_iteration29/"


sequences_path = "/home/woody/b114cb/b114cb23/ZF_FT_alphaamylase_gerard/input.fasta"

with open(sequences_path, "r") as f:
    data = f.read()
    entries = data.split(">")
    final_dict = {}
    for entry in entries:
        if entry == "":
            continue
        else:
            split = entry.split("\n")
            id = split[0]
            sequence = "".join(split[1:])
            final_dict[id] = sequence


eg_seq = "MKLFILAALLGLGLAQHNPHTKPGRSAIVHLFEWRWADIADECERFLGPNGFGGVQISPPNEHIVLESPWRPWWQRYQPISYKLCSRSGTEEELRDMIRRCNNVGVNIYVDAVINHMCGAGGGEGTHSSCGSWFNAGNKDFPSVPFSSWDFNDNKCRTGSGEIENYGDIYQVRDCRLVSLLDLALEKDYVRGKVAEFMNSLIDMGVAGFRVDACKHMWPGDLADIYGRLHDLNTKWFSGGSKPFIFQEVIDLGHEAISAREYFHLGRVTEFKYGAKLGTVFRRWHGEKLSYTRNWGEGWGFMPHGDAVVFVDNHDNQRGHGAGGASIVTFWDPRLHKMAVGYMLAHPYGVARVMSSFRWNRHIVNGKDQNDWMGPPSHKDGSTKSVPINPDQTCGDGWVCEHRWRQIKNMVIFRNVVDGQPHSNWWDNNSNQVAFGRGNRGFIVFNNDDWEMDVTLNTGMPGGTYCDVISGQKEGNVCTGKQIQVGDDGRAHFNISNTDEDPFVAIHAESKL"









pdb_folder = "/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard/output_iteration28/PDB"
pdb_files = os.listdir(pdb_folder)


eg_pdb = os.path.join(pdb_folder, pdb_files[0])


def extract_sequence_from_pdb(pdb_file):
    # Dictionary to convert 3-letter amino acid codes to 1-letter codes
    aa_dict = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    with open(pdb_file, "r") as f:
        lines = f.readlines()
    
    # Track the last residue number to avoid duplicates
    last_res_num = None
    all_residues = []
    
    for line in lines:
        if line.startswith("ATOM"):
            res_num = line[22:26].strip()
            # Only add residue if it's a new residue number
            if res_num != last_res_num:
                res_name = line[17:20].strip()
                if res_name in aa_dict:
                    all_residues.append(aa_dict[res_name])
                    last_res_num = res_num
    
    return "".join(all_residues)



all_sequences_dict = {}
for pdb_file in pdb_files:
    name = pdb_file.split(".")[:-1]
    name = "".join(name)
    sequence = extract_sequence_from_pdb(os.path.join(pdb_folder, pdb_file))
    if len(sequence) < 400 or len(sequence) > 600:
        continue
    else:
        all_sequences_dict[name] = sequence











tokenizer,base_model = load_model(base_model_path)

model_config = base_model.config
model_config.attn_implementation = "eager"
model_config.d_model = 5120
base_model = get_ht_model(base_model, model_config)



tokenizer,dpo_model = load_model(dpo_model_path)
dpo_model_config = dpo_model.config
dpo_model_config.attn_implementation = "eager"
dpo_model_config.d_model = 5120
dpo_model = get_ht_model(dpo_model, dpo_model_config)




# --- KL Divergence Calculation ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)
dpo_model.to(device)
base_model.eval()
dpo_model.eval()





# Get the KL divergence for each sequence

def logits_to_log_probs(logits):
    return torch.nn.functional.log_softmax(logits, dim=-1)

def log_probs_to_kl_divergence(log_probs_1, log_probs_2):
    # KL divergence expects log probabilities for input1 and probabilities for input2
    # So we need to exponentiate log_probs_2
    probs_2 = torch.exp(log_probs_2) + 1e-10
    # Calculate KL divergence without reduction first
    kl_div = torch.nn.functional.kl_div(log_probs_1, probs_2, reduction='none')
    # Sum across the vocabulary dimension (last dimension) to get per-token KL divergence
    kl_div = kl_div.sum(dim=-1)
    # Squeeze the batch dimension since we have batch_size=1
    kl_div = kl_div.squeeze(0)
    return kl_div


all_kl_divergences = {} 
with torch.no_grad():
    for name, sequence in tqdm(all_sequences_dict.items()):

        tokenized_eg_seq = tokenizer.encode("3.2.1.1<sep><start>"+sequence+"<end>", return_tensors="pt")

        logits_base = base_model(tokenized_eg_seq)
        logits_dpo = dpo_model(tokenized_eg_seq)

        base_log_probs = logits_to_log_probs(logits_base)
        dpo_log_probs = logits_to_log_probs(logits_dpo)

        kl_divergence = log_probs_to_kl_divergence(base_log_probs, dpo_log_probs)
        kl_divergence = kl_divergence[:-1].cpu().numpy()
        all_kl_divergences[name] = kl_divergence
        del logits_base, logits_dpo, base_log_probs, dpo_log_probs
        torch.cuda.empty_cache()


print("========")
for name, kl_divergence in all_kl_divergences.items():
    print("Sequence: ", name)
    string_rep = "".join("X" if elem>2 else "_" for elem in all_kl_divergences[name])
    print("KL Divergence: ", string_rep)
    print("=======")




# Function to replace B-factors in PDB file
def replace_bfactors_in_pdb(pdb_file, y_values, output_file):
    # Dictionary to map residue numbers to y values
    residue_to_y = {}
    
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    # First pass: count unique residues to verify y vector length
    unique_residues = set()
    for line in lines:
        if line.startswith('ATOM'):
            chain_id = line[21]
            residue_num = int(line[22:26])
            unique_residues.add((chain_id, residue_num))
    
    sorted_residues = sorted(unique_residues)
    print(f"Found {len(unique_residues)} unique residues in PDB file")
    print(f"Length of y vector: {len(y_values)}")
    
    # Verify that values actually vary
    print(f"Min y value: {min(y_values)}, Max y value: {max(y_values)}")
    
    # Second pass: assign y values to residues
    for i, (chain_id, residue_num) in enumerate(sorted_residues):
        if i < len(y_values):
            residue_to_y[(chain_id, residue_num)] = y_values[i]
        else:
            residue_to_y[(chain_id, residue_num)] = 0.0  # Default value if y is shorter
    
    # Third pass: replace B-factors
    new_lines = []
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            chain_id = line[21]
            residue_num = int(line[22:26])
            
            # Get the y value for this residue
            new_bfactor = residue_to_y.get((chain_id, residue_num), 0.0)
            
            # Format the new B-factor (columns 61-66, right-justified)
            new_bfactor_str = f"{new_bfactor:6.2f}"
            
            # Replace the B-factor in the line
            new_line = line[:60] + new_bfactor_str + line[66:]
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    
    # Write the modified PDB file
    with open(output_file, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Modified PDB file saved as {output_file}")



pdb_kl_divergence_folder = "/home/woody/b114cb/b114cb23/boxo/kl_divergence/PDB_kl_divergence"
os.makedirs(pdb_kl_divergence_folder, exist_ok=True)
for pdb_file in pdb_files:
    name = pdb_file.split(".")[:-1]
    name = "".join(name)
    if name in all_kl_divergences.keys():
        replace_bfactors_in_pdb(os.path.join(pdb_folder, pdb_file), all_kl_divergences[name], os.path.join(pdb_kl_divergence_folder,f"{name}_kl_divergence.pdb"))




