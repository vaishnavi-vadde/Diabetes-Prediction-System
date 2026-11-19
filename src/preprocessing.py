import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_clean_data(filepath="dataset/diabetes.csv"):
    """
    Loads the Pima Indians Diabetes dataset, replaces biologically invalid
    zero values with median values, and returns the cleaned DataFrame.
    """
    if not os.path.exists(filepath):
        # Fallback URL if local CSV is not placed yet
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
        ]
        df = pd.read_csv(url, names=columns)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Downloaded dataset to {filepath}")
    else:
        df = pd.read_csv(filepath)

    # Columns where 0 is biologically implausible
    zero_not_accepted = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for column in zero_not_accepted:
        df[column] = df[column].replace(0, np.nan)
        median_val = df[column].median(skipna=True)
        df[column] = df[column].fillna(median_val)

    return df

def get_statistical_summary(df):
    """CO-1: Computes mean, median, standard deviation, and correlation matrix."""
    stats = pd.DataFrame({
        "Mean": df.mean(),
        "Median": df.median(),
        "Std Dev": df.std(),
        "Variance": df.var()
    })
    corr_matrix = df.corr()
    return stats, corr_matrix

def prepare_data(filepath="dataset/diabetes.csv", test_size=0.2, random_state=42):
    """Splits and scales features for model training."""
    df = load_and_clean_data(filepath)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    df = load_and_clean_data()
    stats, corr = get_statistical_summary(df)
    print("--- Statistical Summary (CO-1) ---")
    print(stats)
    print("\n--- Correlation with Outcome ---")
    print(corr["Outcome"].sort_values(ascending=False))
