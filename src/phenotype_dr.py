import pandas as pd
from sklearn.decomposition import PCA

def reduce_phenotype_dimensionality(input_csv, output_csv='pca20.csv', n_components=20):
    """
    Reduces the dimensionality of the phenotype data using PCA.

    Args:
        input_csv (str): Path to the preprocessed phenotype data.
        output_csv (str): Path to save the dimensionality-reduced data.
        n_components (int): The number of principal components to keep.
    """
    data = pd.read_csv(input_csv)
    
    cols_to_drop = ['DX_GROUP', 'subject']
    x = data.drop(columns=cols_to_drop, axis=1)
    
    if 'Unnamed: 0' in x.columns:
        x.drop('Unnamed: 0', axis=1, inplace=True)
        
    subjects = data['subject']

    pca = PCA(n_components=n_components)
    fit = pca.fit_transform(x)
    
    pca_data = pd.DataFrame(fit, columns=[f'PC_{i+1}' for i in range(n_components)])
    pca_data['subject'] = subjects
    
    pca_data.to_csv(output_csv, index=False)
    print(f"PCA data with {n_components} components saved to {output_csv}")