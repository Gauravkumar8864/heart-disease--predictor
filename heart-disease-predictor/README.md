# Heart Disease Predictor (ML)

A clean, reproducible machine-learning project to predict heart disease using an ensemble (stacking) of tree-based models with a scikit-learn preprocessing pipeline, plus a Gradio app for interactive inference.

## Features
- End-to-end pipeline (`src/train.py`) with robust preprocessing (imputation, scaling, one-hot encoding).
- Stacking ensemble (RandomForest, XGBoost, LightGBM, CatBoost) with Logistic Regression meta-learner.
- Saved model artifact (`models/stacking_pipeline.joblib`) for easy reuse.
- Simple Gradio web app (`app/gradio_app.py`) to try predictions.
- Reproducible environment via `requirements.txt`.
- CI example on GitHub Actions (pytest).
- MIT License.

## Dataset
Put your CSV at `data/heart.csv`. It should include at least the following columns:

- **Numeric**: `Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`  
- **Categorical**: `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`  
- **Target**: `HeartDisease` (0 or 1)

> Tip: If your CSV has different column names, update `src/train.py` accordingly.

## Quickstart

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Add the dataset
# Place your file at: data/heart.csv

# 4) Train the model and generate reports
python -m src.train --data data/heart.csv --out models/stacking_pipeline.joblib --reports reports

# 5) Launch the Gradio app
python app/gradio_app.py
```

Open the printed URL to use the app.

## Project Structure
```
heart-disease-predictor/
├─ app/
│  └─ gradio_app.py
├─ data/
│  └─ README.md
├─ models/
│  └─ .gitkeep
├─ reports/
│  └─ .gitkeep
├─ scripts/
│  └─ heart_disease_colab.py     # your original Colab script
├─ src/
│  ├─ preprocess.py
│  └─ train.py
├─ tests/
│  └─ test_end_to_end.py
├─ .github/workflows/ci.yml
├─ .gitignore
├─ LICENSE
├─ README.md
└─ requirements.txt
```

## Results & Metrics
After training, find metrics in `reports/metrics.json` and plots in `reports/` (confusion matrix and ROC).

## Contributing
PRs welcome. Please open an issue to discuss major changes first.

## License
MIT © Gaurav