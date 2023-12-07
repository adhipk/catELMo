import pandas as pd
from pathlib import Path
import torch
from allennlp.commands.elmo import ElmoEmbedder

model_dir = Path('./path/of/your/downloaded/catELMo')
weights = model_dir/'weights.hdf5'
options = model_dir/'options.json'
embedder  = ElmoEmbedder(options,weights,cuda_device=-1) # cuda_device=-1 for CPU

def catELMo_embedding(x):
    return torch.tensor(embedder.embed_sentence(list(x))).sum(dim=0).mean(dim=0).tolist()

dat = pd.read_csv('./path/of/binding/affinity/prediction/data.csv')
dat['tcr_embeds'] = None
dat['epi_embeds'] = None

dat['epi_embeds'] = dat[['epi']].applymap(lambda x: catELMo_embedding(x))['epi']
dat['tcr_embeds'] = dat[['tcr']].applymap(lambda x: catELMo_embedding(x))['tcr']

dat.to_pickle("./path/of/binding/affinity/prediction/data.pkl")