import pandas as pd
from torch.utils.data import Dataset
import torch as tr
import os
import json
import pickle
from .embeddings import OneHotEmbedding

class SeqDataset(Dataset):
    def __init__(
        self, dataset_path, min_len=0, max_len=512, verbose=False, cache_path=None, for_prediction=False,  training=False,
 **kargs):
        """
        interaction_prior: none, probmat
        """
        self.max_len = max_len
        self.verbose = verbose
        if cache_path is not None and not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        self.cache = cache_path

        # Loading dataset
        data = pd.read_csv(dataset_path)
        self.training = training

        assert (
            "sequence" in data.columns
            and "id" in data.columns
        ), "Dataset should contain 'id' and 'sequence' columns"

        data["len"] = data.sequence.str.len()

        if max_len is None:
            max_len = max(data.len)
        self.max_len = max_len

        datalen = len(data)

        data = data[(data.len >= min_len) & (data.len <= max_len)]

        if len(data) < datalen:
            print(
                f"From {datalen} sequences, filtering {min_len} < len < {max_len} we have {len(data)} sequences"
            )

        self.sequences = data.sequence.tolist()
        self.ids = data.id.tolist()
        self.embedding = OneHotEmbedding()
        self.embedding_size = self.embedding.emb_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seqid = self.ids[idx]
        cache = f"{self.cache}/{seqid}.pk"
        if (self.cache is not None) and os.path.isfile(cache):
            item = pickle.load(open(cache, "rb"))
        else:
            sequence = self.sequences[idx]
            L = len(sequence)
            seq_emb = self.embedding.seq2emb(sequence)


            item = {"id": seqid,   "length": L, "sequence": sequence, "embedding": seq_emb} 

            if self.cache is not None:
                pickle.dump(item, open(cache, "wb"))
                
        return item

def pad_batch(batch, fixed_length=0):
    """batch is a dictionary with different variables lists"""
    L = [b["length"] for b in batch]
    if fixed_length == 0:
        fixed_length = max(L)
    embedding_pad = tr.zeros((len(batch), batch[0]["embedding"].shape[0], fixed_length))
    

    for k in range(len(batch)):
        embedding_pad[k, :, : L[k]] = batch[k]["embedding"]

    out_batch = {
                 "id": [b["id"] for b in batch],
                 "length": L, 
                 "sequence": [b["sequence"] for b in batch],
                "embedding": embedding_pad, 
                 }
    
    return out_batch
