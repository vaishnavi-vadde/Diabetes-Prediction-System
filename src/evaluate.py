import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from preprocessing import prepare_data

def evaluate_all():
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    trained_models = joblib.load("models/all_models.pkl")

    results = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1-Score": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
        })

        print(f"=== {name} Confusion Matrix ===")
        print(confusion_matrix(y_test, y_pred))
        print()

    comparison_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    print("=== Model Comparison Summary (CO-5) ===")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv("models/model_comparison.csv", index=False)

    # Feature Importance from Random Forest / XGBoost
    best_tree_model = trained_models.get("XGBoost", trained_models["Random Forest"])
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": np.round(best_tree_model.feature_importances_, 4)
    }).sort_values(by="Importance", ascending=False)
    feat_imp.to_csv("models/feature_importance.csv", index=False)
    print("\n=== Feature Importance ===")
    print(feat_imp.to_string(index=False))

    # CO-3: PCA Dimensionality Reduction (retaining 95% variance)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    pca_lr = LogisticRegression(random_state=42)
    pca_lr.fit(X_train_pca, y_train)
    pca_auc = roc_auc_score(y_test, pca_lr.predict_proba(X_test_pca)[:, 1])
    print(f"\n=== PCA Analysis (CO-3) ===")
    print(f"Original Features: {X_train.shape[1]} -> PCA Components (95% variance): {pca.n_components_}")
    print(f"Explained Variance Ratio: {np.round(pca.explained_variance_ratio_, 4)}")
    print(f"Logistic Regression ROC-AUC after PCA: {pca_auc:.4f}")

if __name__ == "__main__":
    evaluate_all()
    evaluate_all()
