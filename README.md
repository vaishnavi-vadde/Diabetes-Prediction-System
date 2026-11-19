# Diabetes Prediction System Using Machine Learning

## Project Overview
An educational screening prototype developed for the **M.Tech AI & Data Science (AI & Machine Learning with Applied Mathematics)** course. This system analyzes health-related features to predict diabetes risk using multiple supervised machine learning algorithms.

**Disclaimer:** This is an educational screening prototype, not a medical diagnostic tool.

## Project Structure
```
Diabetes-Prediction-System/
├── dataset/            # Place diabetes.csv here
├── notebooks/          # Exploratory data analysis & mathematical statistics
├── models/             # Saved trained models (best_model.pkl, scaler.pkl)
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── app.py              # Streamlit interactive web application
└── requirements.txt
```

## Setup Instructions
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Install dependencies: `pip install -r requirements.txt`
4. Place `diabetes.csv` inside the `dataset/` folder.
5. Run preprocessing, training, and evaluation scripts in `src/`.
6. Launch the web app: `streamlit run app.py`
