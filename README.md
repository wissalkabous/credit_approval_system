# Credit Approval System

A machine learning–powered web application that automates bank credit decisions — replacing slow, manual review with an instant, data-driven prediction of **Approved / Rejected**, a **confidence score**, and a **risk level**.

Built with **FastAPI** and **scikit-learn**, the project applies SOLID software engineering principles (dependency inversion, interfaces, dependency injection) on top of a trained **Random Forest** classifier, and exposes everything through a REST API with a simple web form.

> Academic project — Machine Learning & Python Programming modules, Master's in AI & Data Analytics, Université Ibn Zohr.

---

## Features

- **Instant credit decisions** — submit 6 applicant attributes, get a decision in under 50ms
- **Random Forest classifier** trained on applicant financial data, selected over Logistic Regression for its ability to capture non-linear feature interactions
- **REST API** built with FastAPI — async request handling, automatic OpenAPI docs, zero-boilerplate validation
- **Strict input validation** via Pydantic — out-of-range or malformed input is rejected before it ever reaches the model
- **Clean, testable architecture** — every component depends on an abstract interface (Dependency Inversion Principle), not a concrete implementation
- **Background logging** — prediction events are logged asynchronously after the response is sent, so the client never waits on I/O
- **Self-contained web interface** — single HTML/JS form served directly by FastAPI, no separate frontend build

---

## How It Works

1. User submits applicant details (age, income, debt, monthly payment, repayment history, credit score) through the web form
2. **Pydantic** validates every field's type and range — invalid input returns an HTTP 422 immediately, before touching the model
3. Input is preprocessed (encoded, scaled) using the same pipeline fitted during training
4. The trained **Random Forest** model returns a probability via `predict_proba()`
5. A decision threshold (`P_approve >= 0.5`) converts that probability into **APPROVED** / **REJECTED**, plus a risk band (Low / Medium / High)
6. The result is returned to the client instantly; the prediction event is logged in the background via FastAPI's `BackgroundTasks`

---

## Tech Stack

| Library | Role |
|---|---|
| **scikit-learn** | Random Forest classifier, StandardScaler, LabelEncoder, evaluation metrics |
| **FastAPI** | Async REST API, automatic OpenAPI docs, dependency injection |
| **Pydantic V2** | Request/response schema validation with type and range enforcement |
| **Pandas** | Data loading, cleaning, and preprocessing |
| **NumPy** | Underlying numerical operations |
| **Uvicorn** | ASGI server |
| **Pickle** | Model and scaler serialization |

---

## Architecture

The codebase follows the **Dependency Inversion Principle** — every component depends on an abstract interface defined in `src/interfaces.py`, not a concrete class. This keeps each layer independently testable and swappable (e.g. replacing Random Forest with XGBoost requires zero changes to the API or data layers).

```
credit_approval_system/
├── api/
│   ├── main.py           # FastAPI routes, startup, HTML interface
│   └── schemas.py        # Pydantic request/response schemas
├── src/
│   ├── interfaces.py     # Abstract base classes (DIP)
│   ├── data_loader.py    # DataProcessor — preprocessing pipeline
│   ├── model.py          # CreditModel — training + evaluation
│   ├── predictor.py      # CreditPredictor — inference orchestrator
│   ├── repository.py     # ModelRepository — model persistence (MLOps)
│   ├── decorators.py     # @log_execution, @timing, @validate_input
│   ├── functional.py     # map_transform, filter_by, pipe, memoize...
│   └── container.py      # Dependency Injection Container (Singleton)
├── data/raw/              # credit_data.csv
├── models/                # credit_model.pkl + metadata.json
└── train.py               # Training entry point
```

**Design patterns applied:**
- **Abstract Base Classes (ABC)** — mandatory contracts for every layer (`IModel`, `IDataProcessor`, `IPredictor`, `IRepository`, `ILogger`)
- **Dependency Inversion** — components receive dependencies (logger, repository) as interfaces, not concrete classes
- **Singleton** — a single shared `DIContainer` instance across the app
- **Decorators** — `@timing` and `@log_execution` add logging/timing to functions without touching their logic
- **Async / BackgroundTasks** — prediction logging runs after the response is already sent to the client

---

## Model Details

**Dataset:** 30 labelled applicant records with 6 input features and 1 binary target (Approved / Rejected).

| Feature | Type | Range |
|---|---|---|
| Age | Numerical | 18 – 100 |
| Annual Income (€) | Numerical | > 0 |
| Existing Debt (€) | Numerical | ≥ 0 |
| Monthly Payment (€) | Numerical | ≥ 0 |
| Repayment History | Ordinal | 0 (worst) – 5 (best) |
| Credit Score | Numerical | 300 – 850 |

**Preprocessing:** duplicate removal, null imputation, label encoding of ordinal features, stratified 80/20 train/test split, and `StandardScaler` normalization (fitted on training data only, to avoid data leakage).

**Algorithm:** Random Forest Classifier, chosen over Logistic Regression for its ability to model non-linear decision boundaries and feature interactions without manual feature engineering.

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 100 | Reduces variance without excessive compute cost |
| `max_depth` | 10 | Overfitting guard on a small dataset |
| `random_state` | 42 | Reproducible results across runs |
| `n_jobs` | -1 | Parallelizes tree construction |

**Evaluation** (held-out test set):

| Metric | Score |
|---|---|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |
| ROC-AUC | 1.000 |

> Note: perfect scores reflect the small, clean, academic-scale dataset (30 records). On real-world, noisy, high-dimensional data these scores would be expected to drop, and techniques like k-fold cross-validation and `GridSearchCV` would be needed for reliable model selection. These results primarily confirm the pipeline is implemented correctly end-to-end.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the HTML web interface |
| `GET` | `/api/health` | Returns API status and whether the model is loaded |
| `GET` | `/api/model-info` | Returns model accuracy, features, and training metadata |
| `POST` | `/api/predict` | Accepts applicant data, returns decision + confidence + risk level |

**Example request** (`POST /api/predict`):

```json
{
  "age": 35,
  "income": 50000,
  "debt": 10000,
  "monthly_payment": 1500,
  "repayment_history": 4,
  "credit_score": 720
}
```

**Example response:**

```json
{
  "decision": "APPROVED",
  "confidence": "87.3%",
  "risk_level": "Low"
}
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/wissalkabous/credit_approval_system.git
cd credit_approval_system

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (generates models/credit_model.pkl)
python train.py

# Launch the API
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser for the web form, or `http://localhost:8000/docs` for the interactive API documentation (Swagger UI).

**Example request via cURL:**

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 50000,
    "debt": 10000,
    "monthly_payment": 1500,
    "repayment_history": 4,
    "credit_score": 720
  }'
```

> **Note on the dataset:** the project initially started from a Kaggle credit dataset (UCI German Credit), but was later scoped down to a custom 30-record dataset on the instructor's guidance, to keep the focus on the ML pipeline and architecture rather than large-scale data cleaning. The dataset lives at `data/raw/credit_data.csv`.

---

## Team

This project was built as a team project for the Machine Learning and Python Programming modules.

---

## License

This project was developed for academic purposes as part of a Master's program and does not carry a formal open-source license.
