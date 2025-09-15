import pandas as pd

def fuse_data(pheno_csv, smri_csv, output_csv):
    """
    Merges the phenotypic data with the sMRI features based on the subject ID.

    Args:
        pheno_csv (str): Path to the preprocessed phenotypic data CSV.
        smri_csv (str): Path to the sMRI features CSV.
        output_csv (str): Path to save the fused dataset.
    """
    data_pheno = pd.read_csv(pheno_csv)
    data_smri = pd.read_csv(smri_csv)
    
    # Ensure the 'subject' column exists in both dataframes
    if 'subject' not in data_pheno.columns or 'subject' not in data_smri.columns:
        raise ValueError("Both CSVs must contain a 'subject' column for merging.")
        
    merged_data = pd.merge(data_pheno, data_smri, on='subject', how='inner')
    merged_data.to_csv(output_csv, index=False)
    print(f"Fused data saved to {output_csv}")