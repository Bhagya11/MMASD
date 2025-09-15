import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fill_nan_with_random(column):
    """
    Fills NaN values in a pandas Series with random values from the non-NaN
    values of the same series.

    Args:
        column (pd.Series): The pandas Series with potential NaN values.

    Returns:
        pd.Series: The Series with NaN values filled.
    """
    mask = column.isnull()
    num_nulls = mask.sum()
    if num_nulls > 0:
        random_values = np.random.choice(column[~mask], num_nulls)
        column[mask] = random_values
    return column

def preprocess_phenotype_data(input_csv, output_csv):
    """
    Performs preprocessing on the phenotype data including dropping columns
    with many missing values, filling NaNs, and label encoding categorical columns.

    Args:
        input_csv (str): The path to the raw phenotype CSV file.
        output_csv (str): The path to save the preprocessed phenotype CSV file.
    """
    df = pd.read_csv(input_csv)

    # Drop columns with 100 or more missing values
    cols_to_drop = [col for col in df.columns if df[col].isna().sum() >= 100]
    df.drop(cols_to_drop, axis=1, inplace=True)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # Fill NaN values in 'FIQ' column
    if 'FIQ' in df.columns:
        df['FIQ'] = fill_nan_with_random(df['FIQ'])

    # Label encode categorical columns
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Fill any remaining NaNs with 0
    df.fillna(0, inplace=True)

    df.to_csv(output_csv, index=False)
    print(f"Preprocessed phenotype data saved to {output_csv}")