# HealthAI

HealthAI is a full-stack **demonstration** of explainable CKD and diabetes risk assessment. It combines a FastAPI API, scikit-learn preprocessing, XGBoost / Random Forest / LightGBM probability fusion, SHAP and LIME explanations, SQLite assessment summaries, and a React + Vite dashboard.

> **Medical safety:** This project is not a medical device, is not clinically validated, and must not be used to diagnose, screen, or guide treatment. The out-of-the-box training data is synthetic and has no clinical validity. Model explanations describe model behavior, not causes. Consult a qualified healthcare professional for medical decisions.

## Project layout

```text
HealthAI/
├── backend/
│   ├── data/                 # Optional input CSVs
│   ├── ml/
│   │   ├── train_models.py   # Train both weighted ensembles
│   │   └── explainer.py      # SHAP, LIME, and summary text
│   ├── models/               # Generated model artifacts
│   ├── database.py           # SQLite / SQLAlchemy history
│   ├── main.py               # FastAPI endpoints
│   └── requirements.txt
└── frontend/
    ├── src/
    ├── package.json
    └── vite.config.js
```

## Run locally in VS Code

Open the `HealthAI` folder in VS Code and use two integrated terminals. If your
PowerShell prompt currently ends in `Downloads>`, first enter the project root:

```powershell
cd .\HealthAI
```

Run the backend commands below from the project root (the folder containing
`backend` and `frontend`); do not `cd backend` before running them.

### 1. Train the models

From the project root, create and activate a virtual environment, then install backend dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python -m backend.ml.train_models
```

Training saves the two fitted preprocessing/model bundles under `backend/models/`. If `backend/data/ckd.csv` or `backend/data/diabetes.csv` is absent, the trainer generates reproducible synthetic demo data.

### 2. Start the API

In the first terminal, from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload
```

The API is at `http://127.0.0.1:8000`; interactive API documentation is at `http://127.0.0.1:8000/docs`. Models must be trained before prediction requests. `GET /api/health` reports whether model artifacts loaded.

### 3. Start the React app

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://127.0.0.1:5173`).

## Optional CSV training data

Place CSV files at `backend/data/ckd.csv` and `backend/data/diabetes.csv`. The CKD CSV uses the 19 feature fields accepted by the API (or the commonly abbreviated UCI CKD field names); the diabetes CSV uses the Pima feature schema. Header names are case-insensitive. Include a target column named `target`, `class`, `classification`, or `Outcome`. Accepted target labels include 0/1, positive/negative, and ckd/notckd. The Pima convention of zero-as-missing is applied to glucose, blood pressure, skin thickness, insulin, and BMI. Data must contain both target classes.

## API summary

- `GET /api/health` — API and loaded model status.
- `POST /api/predict/ckd` — CKD probability fusion, risk category, SHAP and LIME contributions, and explanation.
- `POST /api/predict/diabetes` — diabetes probability fusion and explanation.
- `GET /api/history` — latest 200 saved score/category/timestamp summaries. Clinical input values are not stored.

Ensemble risk is `100 × (0.40 × P_XGBoost + 0.35 × P_RandomForest + 0.25 × P_LightGBM)`, bounded to 0–100. Categories are LOW (<30), MODERATE (30–<60), HIGH (60–<80), and CRITICAL (80–100).
