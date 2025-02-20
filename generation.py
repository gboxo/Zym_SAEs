
"""
Code to generat protein sequences and measure key metrics like diversity, and othe stuff.

"""

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
from transformers import AutoConfig

import einops
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens import HookedTransformer



tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
config_ht = AutoConfig.from_pretrained("nferruz/ProtGPT2")
model_ht = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2",
                                                    attn_implementation="eager")


# We generate from nothing 10 sequences of length 100

from transformers import pipeline
protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2")
sequences = protgpt2("<|endoftext|>", max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0)



# We measure the diversity of the sequences using 

sequences_content = [s["generated_text"] for s in sequences]





import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def one_hot_encode_sequence(sequence, amino_acids="ACDEFGHIKLMNPQRSTVWY"):
    """
    One-hot encode an amino acid sequence.

    Parameters:
    sequence (str): Amino acid sequence.
    amino_acids (str): String of all possible amino acids.

    Returns:
    np.ndarray: One-hot encoded matrix of the sequence.
    """
    encoding = np.zeros((len(sequence), len(amino_acids)), dtype=int)
    aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}
    
    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            encoding[i, aa_to_index[aa]] = 1
    
    return encoding

def compute_cosine_similarity(encoded_seq1, encoded_seq2):
    """
    Compute the cosine similarity of two amino acid sequences.

    Returns:
    float: Cosine similarity score between the two sequences.
    """
    # One-hot encode the sequences
    
    # Compute cosine similarity
    similarity = cosine_similarity(encoded_seq1, encoded_seq2)
    
    # Return the average similarity score
    return np.mean(similarity)


# Example usage

one_hot_sequences = [one_hot_encode_sequence(seq.replace("<|endoftext|>","")) for seq in sequences_content]

similarities = np.zeros((len(one_hot_sequences), len(one_hot_sequences)))
for i in range(len(one_hot_sequences)):
    for j in range(i + 1, len(one_hot_sequences)):
        similarity = compute_cosine_similarity(one_hot_sequences[i], one_hot_sequences[j])
        similarities[i, j] = similarity












