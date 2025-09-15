# src/sitewise_cv.py

"""
Performs Leave-One-Site-Out Cross-Validation (LOSO-CV) to evaluate model
generalizability across different data acquisition sites.

This module trains a suite of classifiers on data from all sites except for one,
which is held out as the test set. A majority vote from all classifiers determines
the final prediction. This process is repeated for each site, and the performance
metrics are calculated and aggregated to produce a site-wise performance table.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

def perform_sitewise_cv(phenotype_csv, features_csv, output_latex_path=None):
    """
    Executes the Leave-One-Site-Out cross-validation workflow.

    Args:
        phenotype_csv (str): Path to the CSV file containing phenotype data,
                             including 'subject', 'DX_GROUP', and 'SITE_ID'.
        features_csv (str): Path to the CSV file containing the features
                            (e.g., from VGG, ResNet) for classification.
        output_latex_path (str, optional): If provided, the final results table
                                           will be saved as a .tex file to this
                                           path. Defaults to None.

    Returns:
        None: The function prints the final results DataFrame to the console
              and optionally saves a LaTeX file.
    """
    print("--- Starting Leave-One-Site-Out Cross-Validation ---")

    # ======================
    # 1. Load and Prepare Data
    # ======================
    try:
        pheno = pd.read_csv(phenotype_csv)
        features = pd.read_csv(features_csv)
    except FileNotFoundError as e:
        print(f"Error: Could not find input file. {e}")
        return

    # Add subject IDs to the feature dataframe to ensure correct merging.
    if 'subject' not in features.columns:
        features.insert(0, 'subject', pheno['subject'])
        
    # Combine the necessary columns into a single dataframe.
    df = pd.merge(pheno[['subject', 'DX_GROUP', 'SITE_ID']], features, on='subject', how='inner')

    # Standardize labels: ASD=1, Control=0
    df['DX_GROUP'] = df['DX_GROUP'].replace(2, 0)
    print(f"Data loaded and merged. Total subjects: {df.shape[0]}")

    # ======================
    # 2. Handle Duplicates (if any)
    # ======================
    # This step ensures that each subject is unique within a given site.
    df = df.drop_duplicates(subset=['subject', 'SITE_ID'], keep='first').reset_index(drop=True)
    print(f"Data after dropping duplicates: {df.shape[0]} subjects")

    sites = df['SITE_ID'].unique()
    X = df.drop(columns=['subject', 'DX_GROUP', 'SITE_ID'])
    y = df['DX_GROUP']

    # ======================
    # 3. Define Models
    # ======================
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGB": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
        "GradientBoost": GradientBoostingClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42)
    }

    # ======================
    # 4. Main Cross-Validation Loop
    # ======================
    site_results = []
    print(f"\nPerforming CV across {len(sites)} sites...")

    for site in sites:
        # Split data into training (all other sites) and testing (current site)
        train_idx = df['SITE_ID'] != site
        test_idx = df['SITE_ID'] == site

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Store predictions from each model
        all_predictions = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            all_predictions.append(preds)

        # Implement majority voting for the final prediction
        predictions_array = np.array(all_predictions)
        majority_vote_preds = []
        for i in range(predictions_array.shape[1]):
            # Count votes for each prediction and take the most common one
            counts = Counter(predictions_array[:, i])
            majority_vote_preds.append(counts.most_common(1)[0][0])
        
        majority_vote_preds = np.array(majority_vote_preds)

        # Calculate performance metrics for the current site
        acc = accuracy_score(y_test, majority_vote_preds)
        prec = precision_score(y_test, majority_vote_preds, zero_division=0)
        rec = recall_score(y_test, majority_vote_preds, zero_division=0)
        f1 = f1_score(y_test, majority_vote_preds, zero_division=0)

        site_results.append([site, acc, prec, rec, f1])
    
    print("Cross-validation loop completed.")

    # ======================
    # 5. Aggregate and Format Results
    # ======================
    results_df = pd.DataFrame(site_results, columns=["Site", "Accuracy", "Precision", "Recall", "F1-score"])

    # Map site codes to full, readable names for the final table
    site_mapping = {
        "CALTECH": "Caltech", "CMU_a": "Carnegie Mellon", "CMU_b": "Carnegie Mellon",
        "KKI": "Kennedy Krieger", "LEUVEN_1": "Leuven", "LEUVEN_2": "Leuven",
        "MAX_MUN_a": "MaxMun", "MAX_MUN_b": "MaxMun", "MAX_MUN_c": "MaxMun", "MAX_MUN_d": "MaxMun",
        "NYU": "New York Univ (NYU)", "OHSU": "OHSU", "OLIN": "Olin College",
        "PITT": "Pittsburgh (Pitt)", "SBL": "SBL", "STANFORD": "Stanford",
        "TRINITY": "Trinity College", "UCLA_1": "UCLA", "UCLA_2": "UCLA",
        "UM_1": "UM", "UM_2": "UM", "USM": "USM", "YALE": "Yale",
    }
    results_df['Site'] = results_df['Site'].replace(site_mapping)

    # For sites with multiple subsets (e.g., UCLA_1, UCLA_2), average the metrics
    results_df = results_df.groupby("Site", as_index=False).mean()

    # Calculate the average performance across all sites
    avg_metrics = results_df[['Accuracy', 'Precision', 'Recall', 'F1-score']].mean().to_frame().T
    avg_metrics.insert(0, 'Site', 'Average')

    # Combine site-specific results with the average row
    final_df = pd.concat([results_df, avg_metrics], ignore_index=True)

    # Sort alphabetically but keep 'Average' at the bottom
    final_df = pd.concat([
        final_df[final_df['Site'] != 'Average'].sort_values("Site"), 
        final_df[final_df['Site'] == 'Average']
    ], ignore_index=True)
    
    # Format floating point numbers for readability
    float_cols = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    final_df[float_cols] = final_df[float_cols].round(2)

    print("\n--- Final Site-wise Performance ---")
    print(final_df.to_string(index=False))

    # ======================
    # 6. Save LaTeX Table (Optional)
    # ======================
    if output_latex_path:
        latex_table = final_df.to_latex(
            index=False,
            column_format="lcccc",
            float_format="%.2f",
            caption="Site-wise Majority Voting Performance (Leave-One-Site-Out CV)",
            label="tab:site_majority"
        )
        try:
            with open(output_latex_path, "w") as f:
                f.write(latex_table)
            print(f"\nLaTeX table successfully saved to {output_latex_path}")
        except IOError as e:
            print(f"Error: Could not write LaTeX file. {e}")

# This allows the script to be run directly for testing purposes
if __name__ == '__main__':
    perform_sitewise_cv(
        phenotype_csv="data_phenotypic.csv",
        features_csv="vgg170.csv",
        output_latex_path="site_majority_table_vgg.tex"
    )