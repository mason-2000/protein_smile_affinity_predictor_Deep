#!/usr/bin/env python3
import pandas as pd 
import torch
import numpy as np
import multiprocessing as mp
import sys

# Global variables used by worker processes in multiprocessing
global model, alphabet, batch_converter
model = None
alphabet = None
batch_converter = None


def save_out(embs, affinity):
    """
    Merge the generated protein embeddings into a DataFrame aligned
    with the affinity table, then save it to disk.
    """
    # Convert list of (embedding, index) pairs into a dict: index → embedding
    emb_dict = { idx: emb for (emb, idx) in embs }
     
    # Create an empty DataFrame with rows matching affinity rows
    encoded_data = pd.DataFrame(
        index=affinity.index,
        columns=['protein_matrix', 'drugs_matrix', 'affinity']
    )

    # Fill 'protein_matrix' column using the Protein_Index mapping
    encoded_data['protein_matrix'] = [
        emb_dict[idx] for idx in affinity['Protein_Index']
    ]

    # Copy the target affinity values
    encoded_data['affinity'] = affinity['Affinity']

    # Save to the output file path provided as sys.argv[4]
    encoded_data.to_pickle(sys.argv[4])
 

def embed_single_sequence(seq: str, protein_name, p_index) -> np.ndarray:
    """
    Encode a single protein sequence using ESM2 and return the
    mean-pooled embedding along with the protein index.

    Args:
        seq (str): Amino acid sequence.
        protein_name (str): Identifier of the protein.
        p_index (int): Protein index (row ID in dataset).

    Returns:
        Tuple[np.ndarray, int]: (mean-pooled embedding, protein index)
    """
    
    # Clean and normalize the sequence
    seq = seq.strip().upper()

    # ESM expects a list of (label, sequence) pairs
    data = [(protein_name, seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Forward pass with no gradient tracking (inference mode)
    with torch.no_grad():
        results = model(
            batch_tokens,
            repr_layers=[model.num_layers],
            return_contacts=False
        )
        # Extract token-level embeddings from the last layer
        token_reps = results['representations'][model.num_layers]

    # Extract embeddings for actual residues (skip special tokens),
    # then apply mean pooling over the sequence length.
    emb = token_reps[0, 1:len(seq)+1].mean(dim=0)

    print(f'Protein: {protein_name} encoded ')
    
    # Return the embedding as float32 + index to preserve ordering
    return emb.cpu().numpy().astype(np.float32), p_index


def init_model_ESM2():
    """
    Load the ESM2 model and its alphabet once per worker process.
    This avoids reloading the model for every sequence.
    """
    global model, alphabet, batch_converter

    # Load ESM2 model via torch.hub
    model, alphabet = torch.hub.load(
        'facebookresearch/esm:main',
        'esm2_t6_8M_UR50D'
    )

    # Batch converter turns raw sequences into token tensors
    batch_converter = alphabet.get_batch_converter()

    # Put model in evaluation mode
    model.eval()


def load_data():
    """
    Load proteins and affinity tables from the file paths provided
    on the command line.

    Expected sys.argv:
        1: proteins.csv
        2: affinity.csv
        3: number of threads
        4: output file (passed to save_out)
    """
    proteins = pd.read_csv(sys.argv[1], index_col=0)
    affinity = pd.read_csv(sys.argv[2])

    # Remove non-sequence metadata not needed for embedding
    proteins.drop('Accession_Number', axis=1, inplace=True)
        
    return proteins, affinity
        

if __name__ == '__main__':
    
    # Load protein sequences and affinity table
    proteins, affinity = load_data()

    # Number of worker processes for multiprocessing
    threads_n = int(sys.argv[3])
    print(f'[INFO] using {threads_n} threads')

    # Extract sequences, names, and indices for batch processing
    seqs = [
        str(proteins.loc[index, 'Sequence'].strip().upper())
        for index in proteins.index
    ]
    p_names = [
        str(proteins.loc[index, 'Gene_Name'].strip())
        for index in proteins.index
    ]
    p_index = proteins.index.tolist()

    # Zip everything into argument tuples for starmap
    args = zip(seqs, p_names, p_index)
    
    # Multiprocessing pool: each process loads its own ESM2 model
    with mp.Pool(
        processes=threads_n,
        initializer=init_model_ESM2
    ) as p:
        # Parallel embedding of all sequences
        embs = p.starmap(embed_single_sequence, args)
        
    # Store embeddings aligned with the affinity file
    save_out(embs, affinity)
