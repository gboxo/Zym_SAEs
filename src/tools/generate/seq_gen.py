import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os
from tqdm import tqdm
import math
import argparse
from src.utils import load_config



def remove_characters(sequence, char_list):
    "This function removes special tokens used during training."
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids,model,tokenizer):
    "This function computes perplexities for the generated sequences"
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)
        
def generate_without_intervention(model, tokenizer, prompt: str, max_new_tokens=512, n_samples=10, device="cuda"):
    """
    Generate sequences without any intervention (baseline generation)
    Consolidated from generate_without_penalty.py
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_ids_batch = input_ids.repeat(n_samples, 1)
    
    output = model.generate(
        input_ids_batch, 
        top_k=9,
        eos_token_id=1,
        do_sample=True,
        verbose=False,
        max_new_tokens=max_new_tokens,
    )
    
    all_outputs = tokenizer.batch_decode(output)
    all_outputs = [o.replace("<|endoftext|>", "") for o in all_outputs]
    return all_outputs

def main(label, model,special_tokens,device,tokenizer, n_samples):

    
    # Generating sequences
    input_ids = tokenizer.encode(label,return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids, 
        top_k=9, #tbd
        repetition_penalty=1.2,
        max_length=1024,
        eos_token_id=1,
        pad_token_id=0,
        do_sample=True,
        num_return_sequences=n_samples) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.
    
    # Check sequence sanity, ensure sequences are not-truncated.
    # The model will truncate sequences longer than the specified max_length (1024 above). We want to avoid those sequences.
    new_outputs = [ output for output in outputs if output[-1] == 0]
    if not new_outputs:
        print("not enough sequences with short lengths!!")
    

    # Compute perplexity for every generated sequence in the batch
    ppls = [(tokenizer.decode(output), calculatePerplexity(output, model, tokenizer)) for output in new_outputs ]



    # Sort the batch by perplexity, the lower the better
    ppls.sort(key=lambda i:i[1]) # duplicated sequences?

    # Final dictionary with the results
    sequences={}
    sequences[label] = [(remove_characters(x[0], special_tokens), x[1]) for x in ppls]

    return sequences

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    args = parser.parse_args()
    
    device = torch.device("cuda")
    
    # Standard sequence generation with perplexity scoring
    cfg_path = args.cfg_path
    config = load_config(cfg_path)
    model_name = config["model_name"]
    ec_label = config["label"]
    out_dir = config["paths"]["out_dir"]

    print('Reading pretrained model and tokenizer')
    model_path = config["paths"]["model_path"]    


    n_samples = config["n_samples"]
    n_batches = config["n_batches"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    special_tokens = ['<start>', '<end>', '<|endoftext|>', '<pad>', ' ', '<sep>']

    label = ec_label
    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    print("Starting sequence generation")
    
    all_sequences = []
    unique_sequences = set()
    
    for i in tqdm(range(n_batches)):
        sequences = main(label, model, special_tokens, device, tokenizer, n_samples)
        for key, value in sequences.items():
            for index, val in enumerate(value):
                sequence = val[0]
                if all(char in canonical_amino_acids for char in sequence) and sequence not in unique_sequences:
                    unique_sequences.add(sequence)
                    sequence_info = {
                        'label': label,
                        'batch': i,
                        'index': index,
                        'pepr': float(val[1]),
                        'fasta': f">{label}_{i}_{index}_{model_name}\t{val[1]}\n{sequence}\n"
                    }
                    all_sequences.append(sequence_info)
        print(f"Iteration {i+1}: {len(unique_sequences)} unique sequences found so far")
                
    fasta_content = ''.join(seq['fasta'] for seq in all_sequences)

    os.makedirs(out_dir, exist_ok=True)
    output_filename = f"{out_dir}/seq_gen_{label}_{model_name}.fasta"
    with open(output_filename, "w") as fn:
        fn.write(fasta_content)
    
    print(f"Final result: {len(unique_sequences)} unique sequences saved to {output_filename}")
    fn.close()
    
