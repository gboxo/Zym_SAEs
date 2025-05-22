# %%
from tooling import *        

from sae_lens import SAE




# %%

prompt = ["In the beginning, God created the heavens and the","In the beginning, God created the heavens and the"]
df = pd.DataFrame.from_records({k:v.__dict__ for k,v in get_pretrained_saes_directory().items()}).T


model = HookedSAETransformer.from_pretrained("gpt2-small", device = device)

saes_dict = {}


for l in range(11,12):

    sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = "gpt2-small-resid-post-v5-32k",
            sae_id = f"blocks.{l}.hook_resid_post",
            device = device
            )
    saes_dict[f"blocks.{l}.hook_resid_post"] = sae
    if l == 0:
        cfg = cfg_dict

dataset = load_dataset(
        path = "/home/gerard/MI/pile-10k",
        split = "train",
        streaming = False
        )



token_dataset = tokenize_and_concatenate(
        dataset = dataset,
        tokenizer = model.tokenizer,
        streaming = True,
        max_length = cfg_dict['context_size'],
        add_bos_token=cfg_dict['prepend_bos']
        )
# %%

import requests
from torch.nn.functional import log_softmax

tokens = token_dataset['tokens'][:4]
prompt = model.tokenizer.batch_decode(tokens)


def metric_fn(logits: torch.Tensor, tokens = tokens) -> torch.Tensor:
    print(logits.shape)
    log_probs = log_softmax(logits, dim = -1)
    x = - torch.gather(log_probs[:,:-1,:],-1, tokens[:,1:].unsqueeze(-1))
    return x



def kl_metric_fn(logtis):
    pass

feature_attribution_df = calculate_feature_attribution(
    model = model,
    input = prompt,
    metric_fn = metric_fn,
    include_saes=saes_dict,
    include_error_term=True,
    return_logits=True,
)
print(feature_attribution_df)





# %%

from sklearn.decomposition import PCA


attrbs = feature_attribution_df.sae_feature_attributions['blocks.7.hook_resid_post']

































