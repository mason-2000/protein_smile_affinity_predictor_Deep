#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
import sys

# Global model/tokenizer objects.
# Each worker process needs its own copy, initialized in init_model().
global model, tokenizer
model = None
tokenizer = None


def save_output(embs, affinity):
    """
    Merge the drug embeddings into the existing protein embeddings file
    produced by encode_proteins.py, aligning everything by Drug_Index.
    """

    # Convert list of (embedding, index) into dict: index → embedding
    emb_dict = { idx: emb for (emb, idx) in embs }

    # Load the intermediate file created during protein encoding
    encoded_data = pd.read_pickle(sys.argv[2])

    # Fill the 'drugs_matrix' column using Drug_Index ordering
    encoded_data['drugs_matrix'] = [
        emb_dict[idx] for idx in affinity['Drug_Index']
    ]

    # Save combined protein+drug embeddings to final output file
    encoded_data.to_pickle(sys.argv[4])


def init_model():
    """
    Initialize the ChemBERTa tokenizer and model.
    This is executed once per worker process when using multiprocessing.
    """

    global model, tokenizer

    # Load tokenizer and model from HuggingFace model hub
    tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
    model = AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM')

    # Set to evaluation (no gradient computation)
    model.eval()


def smiles_to_chemberta_embedding(smiles: str) -> np.ndarray:
    """
    Encode a drug molecule (SMILES string) using ChemBERTa.
    Mean pooling is applied to produce a fixed-length vector.

    Args:
        smiles (str): Molecular SMILES string.

    Returns:
        np.ndarray: Mean-pooled ChemBERTa embedding (float32).
    """

    # Tokenize the SMILES string and convert to model inputs
    inputs = tokenizer(
        smiles,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )

    # Forward pass with gradients disabled
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract last hidden state: shape (1, seq_len, hidden_dim)
    embeddings = outputs.last_hidden_state

    # === MEAN POOLING over sequence dimension ===
    embeddings = embeddings.mean(dim=0)   # → shape (1, hidden_dim)
    embeddings = embeddings.squeeze(0)    # → shape (hidden_dim,)

    print(f'SMILE: {smiles[:10]}... encoded')

    # Return numpy float32 vector
    return embeddings.cpu().numpy().astype(np.float32)


def load_data():
    """
    Load the drugs table and the affinity table used to map Drug_Index.
    """
    
    # Input CSV provided by Snakemake (drugs.csv)
    drugs = pd.read_csv(sys.argv[1], index_col=0)

    # Remove unused metadata columns
    drugs.drop(['CID', 'Canonical_SMILES'], axis=1, inplace=True)

    # Load affinity table to know each drug's index mapping
    affinity = pd.read_csv('raw_data/drug_protein_affinity.csv')

    return drugs, affinity


if __name__ == '__main__':

    # Load input data (drugs + affinity table)
    drugs, affinity = load_data()

    # Number of parallel worker processes
    threads_n = int(sys.argv[3])
    print(f'[INFO] using {threads_n} threads')

    # Extract SMILES strings and drug indices for batch processing
    smiles = [str(smi).strip() for smi in drugs['Isomeric_SMILES']]
    drugs_index = drugs.index.to_list()

    # Parallel embedding using multiprocessing
    with mp.Pool(
        processes=threads_n,
        initializer=init_model
    ) as p:
        embs = p.map(smiles_to_chemberta_embedding, smiles)
        
    # Combine embeddings with indices
    embs = zip(embs, drugs_index)

    # Merge with protein embeddings and save final file
    save_output(embs, affinity)


    

        