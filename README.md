# Classic Capital Co-operative Society Credit App

This project is a Streamlit-based credit scoring and borrower assessment application built for Classic Capital Co-operative Society.

It provides a branded interface for:
- borrower data entry
- AI-based default risk prediction
- risk classification and lending recommendation
- record storage and review

## Project overview

The app uses a trained machine learning model saved as `credit_model.pkl` to estimate the probability that a borrower may default.

The current version is a prototype. It is useful for demonstration, testing, and development. It should not be used for real lending decisions until it is retrained on complete historical company data.

## Main files

- `app.py` — main Streamlit application
- `credit_model.pkl` — trained machine learning model
- `logo.png` — company logo used in the interface
- `borrower_assessments.csv` — saved assessment records
- `train_model.py` — script used to train the model
- `classic_capital_credit_dataset_filled_sample.csv` — sample dataset used for prototype model training
- `requirements.txt` — Python dependencies for deployment

## Features

- company-branded interface using logo and colors
- borrower profile input form
- machine learning-based default probability prediction
- risk category classification
- recommendation output
- saved records table
- downloadable assessment data

## Technology stack

- Python
- Streamlit
- pandas
- scikit-learn
- joblib

## How to run locally

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Open terminal in the project folder.

3. Run the app:

```bash
streamlit run app.py
```

4. Open the browser link shown in the terminal, usually:

```bash
http://localhost:8501
```

## How to retrain the model

If you update the training dataset, retrain the model with:

```bash
python train_model.py
```

This will overwrite `credit_model.pkl` with a new trained model.

## Deployment

This app can be deployed on Streamlit Community Cloud.

For deployment, upload the project files to a GitHub repository and connect the repository to Streamlit Community Cloud.

## Important note

The current trained model was built on a very small sample dataset with several assumed values for prototype testing. Because of that, the current prediction results are only for demonstration.

Before operational use, the model should be retrained on:
- real borrower records
- real loan repayment outcomes
- complete default history

## Recommended next step

The strongest next phase is to collect real historical loan data from the co-operative and retrain the model to improve accuracy, fairness, and operational usefulness.
