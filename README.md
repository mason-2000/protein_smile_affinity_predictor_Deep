# protein_smile_affinity_predictor_Deep
A transformer-based pipeline that encodes proteins (ESM2) and drugs (ChemBERTa) to train a neural model for drug–protein affinity prediction.

## Overview

This project implements a fully automated workflow to:

1. Encode protein sequences using ESM2.
2. Encode drug molecules (SMILES) using ChemBERTa.
3. Merge embeddings into a unified dataset.
4. Train a feed-forward neural network to predict drug–protein affinity.
5. Evaluate the model using RMSE.
6. Export the trained model and final evaluation metrics.

All steps are orchestrated with Snakemake for reproducibility, parallel execution, and dependency tracking.

## Project Structure

```
project/
│
├── raw_data/
│   ├── proteins.csv
│   ├── drugs.csv
│   ├── drug_protein_affinity.csv
│
├── encoded_data/
│   ├── proteins_embeddings.pkl
│   ├── merged_embeddings.pkl
│
├── model/
│   ├── Affinity_Predictor.pth
│   ├── RMSE.txt
│
├── encode_proteins.py
├── encode_drugs.py
├── tune_model.py
├── Snakefile
└── README.md
```

## Workflow Summary

### 1. Protein Embedding (ESM2)
Script: encode_proteins.py 
Model: esm2_t6_8M_UR50D 
Output: encoded_data/proteins_embeddings.pkl

Each protein sequence is processed with ESM2, and mean pooling is applied to obtain a fixed-size embedding vector.

### 2. Drug Embedding (ChemBERTa) and Dataset Merge
Script: encode_drugs.py 
Model: ChemBERTa-77M-MLM 
Output: encoded_data/merged_embeddings.pkl

Each SMILES string is encoded into a 768-dimensional embedding, mean pooled, and aligned with protein embeddings according to indices in the affinity table.

### 3. Model Training and Evaluation
Script: tune_model.py 
Model architecture:

Input → 512 → 256 → 128 → 1 
(Feed-forward MLP with ReLU and dropout)

The dataset is split into train, validation, and test sets.
The model is trained using RMSE as the main metric.
The best validation checkpoint is restored before testing.

Outputs:
- model/Affinity_Predictor.pth
- model/RMSE.txt

## Snakemake Pipeline

Run the entire workflow with:

```
snakemake --cores N
```

This will automatically:
1. Encode proteins
2. Encode drugs and merge datasets
3. Train and evaluate the affinity predictor

## Installation

### Create a Python environment

```
conda create -n affinity python=3.10
conda activate affinity
```

### Install dependencies

```
pip install torch transformers pandas scikit-learn snakemake
```

Optional: Install GPU-enabled PyTorch from https://pytorch.org/

## Running the Project

### Step 1: Prepare raw data

Ensure the following files exist in raw_data/:
- proteins.csv
- drugs.csv
- drug_protein_affinity.csv

### Step 2: Run the full pipeline

```
snakemake -j 6
```

### Step 3: Check output files

- model/Affinity_Predictor.pth
- model/RMSE.txt

## Customization

### Change batch size  
Edit in tune_model.py:

```
init_datasets(batch_size=64)
```

### Change number of training epochs  
Edit:

```
tune_model(..., epochs=100)
```

### Change transformer models  
Modify them in encode_proteins.py or encode_drugs.py.

## Notes

- Embeddings use mean pooling to ensure fixed-size vectors.
- Multiprocessing is used for faster encoding.
- Snakemake ensures reproducibility and incremental execution.
- The best checkpoint is selected based on validation RMSE.

