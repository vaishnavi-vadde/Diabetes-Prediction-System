import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from preprocessing import prepare_data

def train_models():
    X_train, X_test, y_train, y_test, feature_names = prepare_data()

    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42),
        "Support Vector Machine": SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)
    }

    best_model = None
    best_model_name = ""
    best_auc = 0.0
    trained_models = {}

    print("Training and comparing models (CO-2 & CO-4)...\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"{name:24s} | Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        trained_models[name] = model

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = name

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(trained_models, "models/all_models.pkl")
    print(f"\nSaved best model ({best_model_name} with ROC-AUC: {best_auc:.4f}) to models/best_model.pkl")

if __name__ == "__main__":
    train_models()
