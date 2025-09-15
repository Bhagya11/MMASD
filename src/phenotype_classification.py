import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import os

def classify_phenotype_data(features_csv, phenotype_csv):
    """
    Performs classification on phenotype features using multiple models and a majority voting ensemble.

    Args:
        features_csv (str): Path to the phenotype features CSV (e.g., from PCA).
        phenotype_csv (str): Path to the original phenotype CSV to get the labels.
    """
    print(f"\n--- Classification for {features_csv} ---")
    
    features_df = pd.read_csv(features_csv)
    phenotype_df = pd.read_csv(phenotype_csv)
    
    data = pd.merge(features_df, phenotype_df[['subject', 'DX_GROUP']], on='subject', how='inner')
    
    data['DX_GROUP'] = data['DX_GROUP'].replace(2, 0)
    
    X = data.drop(columns=['subject', 'DX_GROUP'])
    y = data['DX_GROUP']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "AdaBoost Classifier": AdaBoostClassifier(),
        "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "Gradient Boost Classifier": GradientBoostingClassifier(),
        "RandomForest": RandomForestClassifier()
    }

    majority_voting_classifier = VotingClassifier(estimators=list(models.items()), voting='hard')
    
    # Train models
    for model in models.values():
        model.fit(X_train, y_train)
    majority_voting_classifier.fit(X_train, y_train)

    # Evaluate models
    results = []
    for name, model in models.items():
        preds = model.predict(X_test)
        results.append({
            'Model': name,
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds),
            'F1 Score': f1_score(y_test, preds),
            'Accuracy': accuracy_score(y_test, preds)
        })

    # Evaluate majority voting
    majority_preds = majority_voting_classifier.predict(X_test)
    results.append({
        'Model': 'Majority Voting',
        'Precision': precision_score(y_test, majority_preds),
        'Recall': recall_score(y_test, majority_preds),
        'F1 Score': f1_score(y_test, majority_preds),
        'Accuracy': accuracy_score(y_test, majority_preds)
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
    plt.title(f'AUC-ROC Curves for Phenotype ({os.path.basename(features_csv)})')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_phenotype_{os.path.basename(features_csv).split(".")[0]}.png')
    plt.show()