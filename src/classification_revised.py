import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, confusion_matrix)
import matplotlib.pyplot as plt
import numpy as np

def calculate_specificity(y_true, y_pred):
    """Calculates specificity from the confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def run_classification(X, y, title_suffix):
    """
    Runs a standardized classification workflow and prints results.

    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target labels.
        title_suffix (str): A suffix for plot titles and filenames.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "AdaBoost Classifier": AdaBoostClassifier(),
        "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "Gradient Boost Classifier": GradientBoostingClassifier(),
        "RandomForest": RandomForestClassifier()
    }

    majority_voting_classifier = VotingClassifier(estimators=list(models.items()), voting='hard')
    
    # Train
    for model in models.values():
        model.fit(X_train, y_train)
    majority_voting_classifier.fit(X_train, y_train)

    # Evaluate
    results = []
    individual_predictions = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        individual_predictions[name] = preds
        results.append({
            'Model': name,
            'Precision': precision_score(y_test, preds, zero_division=0),
            'Recall': recall_score(y_test, preds, zero_division=0),
            'F1 Score': f1_score(y_test, preds, zero_division=0),
            'Accuracy': accuracy_score(y_test, preds),
            'Specificity': calculate_specificity(y_test, preds)
        })

    majority_preds = majority_voting_classifier.predict(X_test)
    results.append({
        'Model': 'Majority Voting',
        'Precision': precision_score(y_test, majority_preds, zero_division=0),
        'Recall': recall_score(y_test, majority_preds, zero_division=0),
        'F1 Score': f1_score(y_test, majority_preds, zero_division=0),
        'Accuracy': accuracy_score(y_test, majority_preds),
        'Specificity': calculate_specificity(y_test, majority_preds)
    })

    results_df = pd.DataFrame(results)
    print(results_df)

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUC-ROC Curves (Revised) - {title_suffix}')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_revised_{title_suffix}.png')
    plt.show()

def revised_classification_smri(features_csv, phenotype_csv):
    """Revised classification for sMRI data."""
    print(f"\n--- Revised Classification for sMRI ({features_csv}) ---")
    features_df = pd.read_csv(features_csv)
    phenotype_df = pd.read_csv(phenotype_csv)
    data = pd.merge(features_df, phenotype_df[['subject', 'DX_GROUP']], on='subject', how='inner')
    data['DX_GROUP'] = data['DX_GROUP'].replace(2, 0)
    X = data.drop(columns=['subject', 'DX_GROUP'])
    y = data['DX_GROUP']
    run_classification(X, y, f"sMRI_{features_csv.split('.')[0]}")

def revised_classification_phenotype(phenotype_csv):
    """Revised classification for phenotype data."""
    print(f"\n--- Revised Classification for Phenotype ---")
    data = pd.read_csv(phenotype_csv)
    data['DX_GROUP'] = data['DX_GROUP'].replace(2, 0)
    X = data.drop(columns=['subject', 'DX_GROUP'])
    y = data['DX_GROUP']
    run_classification(X, y, "Phenotype")

def revised_classification_fusion(fused_csv, phenotype_csv):
    """Revised classification for fused data."""
    print(f"\n--- Revised Classification for Fused Data ---")
    data = pd.read_csv(fused_csv)
    phenotype_df = pd.read_csv(phenotype_csv)
    data = pd.merge(data, phenotype_df[['subject', 'DX_GROUP']], on='subject', how='inner')
    data['DX_GROUP'] = data['DX_GROUP'].replace(2, 0)
    X = data.drop(columns=['subject', 'DX_GROUP'])
    y = data['DX_GROUP']
    run_classification(X, y, "Fusion")