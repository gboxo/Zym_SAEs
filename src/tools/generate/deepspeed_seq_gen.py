
import deepspeed
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os
from tqdm import tqdm
import argparse
from generate_utils import load_config



def remove_characters(sequence, char_list):
    "This function removes special tokens used during training."
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



    # Final dictionary with the results
    sequences={}
    sequences[label] = [(remove_characters(x[0], special_tokens), x[1]) for x in ppls]

    return sequences

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    


    device = torch.device("cuda") # Replace with 'cpu' if you don't have a GPU - but it will be slow
    print('Reading pretrained model and tokenizer')
    
    model_name = "AI4PD/ZymCTRL" 
    #model_name = '/home/woody/b114cb/b114cb23/models/ZymCTRL/'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name) # change to ZymCTRL location
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device) # change to ZymCTRL location
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']

    label = "3.2.1.1"
    
    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")  # Set of canonical amino acids
    print("Starting sequence generation")
    
    all_sequences = []
    for i in tqdm(range(50)):
        sequences = main(label, model, special_tokens, device, tokenizer)
        for key, value in sequences.items():
            for index, val in enumerate(value):
                if all(char in canonical_amino_acids for char in val[0]):
                    sequence_info = {
                        'label': label,
                        'batch': i,
                        'index': index,
                        'pepr': float(val[1]),
                        'fasta': f">{label}_{i}_{index}_iteration{iteration_num}\t{val[1]}\n{val[0]}\n"
                    }
                    all_sequences.append(sequence_info)
    fasta_content = ''.join(seq['fasta'] for seq in all_sequences)

    output_filename = f"seq_gen_{label}.fasta"
    print(fasta_content)
    with open(output_filename, "w") as fn:
            fn.write(fasta_content)
        
    fn.close()
        
