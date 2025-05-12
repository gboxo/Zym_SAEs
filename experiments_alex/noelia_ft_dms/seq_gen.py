import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os
from tqdm import tqdm
import math
import argparse
from argparse import ArgumentParser
import numpy as np



def remove_characters(sequence, char_list):
    "This function removes special tokens used during training."
    print(sequence)
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

        
def main(label, model,special_tokens,device,tokenizer):

    
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
        num_return_sequences=20) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.
    
    # Check sequence sanity, ensure sequences are not-truncated.
    # The model will truncate sequences longer than the specified max_length (1024 above). We want to avoid those sequences.
    new_outputs = [ output for output in outputs if output[-1] == 0]
    if not new_outputs:
        print("not enough sequences with short lengths!!")
    new_outputs = [tokenizer.decode(output) for output in new_outputs ]

    


    # Final dictionary with the results
    sequences={}
    sequences[label] = [remove_characters(x, special_tokens) for x in new_outputs]
    return sequences

if __name__=='__main__':

    device = torch.device("cuda") # Replace with 'cpu' if you don't have a GPU - but it will be slow
    print('Reading pretrained model and tokenizer')
    model_name = "/home/woody/b114cb/b114cb23/models/model-3.2.1.1/"
    
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name) # change to ZymCTRL location
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device) # change to ZymCTRL location
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']

    label = "3.2.1.1"
    
    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")  # Set of canonical amino acids
    print("Starting sequence generation")
    
    all_sequences = []
    for i in tqdm(range(10)):
        sequences = main(label, model, special_tokens, device, tokenizer)
        for key, value in sequences.items():
            for index, val in enumerate(value):
                if all(char in canonical_amino_acids for char in val):
                    sequence_info = {
                        'label': label,
                        'batch': i,
                        'index': index,
                        'fasta': f">{label}_{i}_{index}_ZC_FT\n{val}\n"
                    }
                    all_sequences.append(sequence_info)
        fasta_content = ''.join(seq['fasta'] for seq in all_sequences)

        out_dir = "/home/woody/b114cb/b114cb23/boxo/noelia_ft_dms/seq_gen/"
        os.makedirs(out_dir, exist_ok=True)
        output_filename = f"{out_dir}/seq_gen_{label}_ZC_FT.fasta"
        print(fasta_content)
        with open(output_filename, "w") as fn:
                fn.write(fasta_content)
            
        fn.close()
        
