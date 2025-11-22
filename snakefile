rule all:
    input: 
        "model/Affinity_Predictor.pth",
        "model/RMSE.txt"

rule encode_proteins:
    input: 
        proteins="raw_data/proteins.csv",
        affinity="raw_data/drug_protein_affinity.csv"
    params:
        threads=6
    output: 
        "encoded_data/proteins_embeddings.pkl"
    shell:
        "python3 encode_proteins.py {input.proteins} {input.affinity} {params.threads} {output}"

rule encode_drugs:
    input: 
        drugs="raw_data/drugs.csv",
        proteins_emb="encoded_data/proteins_embeddings.pkl"
    params:
        threads=6
    output: 
        "encoded_data/merged_embeddings.pkl"
    shell:
        "python3 encode_drugs.py {input.drugs} {input.proteins_emb} {params.threads} {output}"

rule tune_model:
    input:
        "encoded_data/merged_embeddings.pkl"
    output:
        "model/Affinity_Predictor.pth",
        "model/RMSE.txt"
    shell:
        "python3 tune_model.py {input} {output}"