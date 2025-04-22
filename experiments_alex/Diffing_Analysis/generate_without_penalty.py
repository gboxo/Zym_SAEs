#from SAELens.sae_lens import HookedSAETransformer, SAE, SAEConfig
from sae_lens import HookedSAETransformer, SAE, SAEConfig
from src.utils import load_model, get_sl_model, load_sae
from functools import partial
import torch
import os




def generate_without_penalty(model: HookedSAETransformer, prompt: str, max_new_tokens=512, n_samples=10):
    input_ids = model.to_tokens(prompt, prepend_bos=False)
    input_ids_batch = input_ids.repeat(n_samples, 1)
    

    all_outputs = []
    output = model.generate(
        input_ids_batch, 
        top_k=9, 
        eos_token_id=1,
        do_sample=True,
        verbose=False,
        max_new_tokens=800,
        ) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.
    
    all_outputs = model.tokenizer.batch_decode(output)
    all_outputs = [o.replace("<|endoftext|>", "") for o in all_outputs]


    return all_outputs



if __name__ == "__main__":


    for model_iteration, data_iteration in [(5,5)]:
        print(f"Model iteration: {model_iteration}, Data iteration: {data_iteration}")
        # Load the dataframe
        model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"


        tokenizer, model = load_model(model_path)
        model = get_sl_model(model, model.config, tokenizer).to("cuda")
    

        prompt = "3.2.1.1<sep><start>"


        os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/without_penalty/M{model_iteration}_D{data_iteration}", exist_ok=True)

        out = generate_without_penalty(model, prompt, max_new_tokens=1024, n_samples=10)
        with open(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/without_penalty/M{model_iteration}_D{data_iteration}/sequences.txt", "w") as f:
            for i, o in enumerate(out):
                f.write(f">3.2.1.1_{i},"+o+"\n")
        
    torch.cuda.empty_cache()