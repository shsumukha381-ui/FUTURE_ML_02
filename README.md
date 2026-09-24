# 🎫 Support Ticket Classification & Prioritization System

An end-to-end **Machine Learning system** that automatically classifies IT support tickets into categories and predicts priority levels. Built with **scikit-learn** for ML, **FastAPI** for the backend API, and a modern **glassmorphism web UI**.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-Deployed-000000?style=for-the-badge&logo=vercel&logoColor=white)

---

## 🚀 Live Demo

> **Frontend**: Deployed on Vercel — [View Live App](https://future-ml-02.vercel.app)
>
> ⚠️ The live frontend requires a running backend API. To test locally, follow the [Quick Start](#-quick-start) below.

---

## 📸 Features

### 🔍 Single Ticket Classification
Paste any support ticket text and get instant AI-powered predictions with confidence scores and full probability distributions.

### 📊 Batch Upload
Upload a CSV file of tickets, view results in a sortable/filterable table, and export predictions as CSV.

### 📈 Insights Dashboard
Model evaluation metrics, category/priority distribution charts, and per-class performance breakdown.

---

## 🟢 Feature Status

The UI is a 9-page admin product branded **"SupportMind — AI Intelligence"**. Below is a transparency guide distinguishing fully functional features from demo/illustrative panels.

| Feature | Status | Details |
|---------|--------|---------|
| **Ticket Analyzer** (`/predict`) | ✅ Fully functional | Real ML inference via FastAPI backend |
| **Dashboard — AI Accuracy & Total Tickets** | ✅ Real data | From model metadata & dataset support counts |
| **Dashboard — Category Distribution** | ✅ Real data | From per-class support counts in training data |
| **Model Comparison Table** | ✅ Real data | From 5-fold cross-validation across 4 models |
| **API Docs** | ✅ Real | Documents actual `/predict` endpoint with working code samples |
| **Batch CSV Upload** | ✅ Fully functional | Parses CSV, sends to `/predict/batch`, shows results |
| Dashboard — Volume Trend | 🟡 Demo | Simulated daily volumes (no real time-series data) |
| Dashboard — Critical/Resolved/Response/CSAT/Open/SLA metrics | 🟡 Demo | Placeholder values — no ticket lifecycle tracking |
| Training Configuration | 🟡 Demo controls | UI-only — no in-browser retraining endpoint |
| Accuracy/Loss Curves | 🟡 Demo | Plotted from cross-validation fold scores |
| Dataset Upload — Cleaning Toggles | 🟡 Partial | Upload works; cleaning toggles are UI-only |
| Analytics — Sentiment | 🟡 Demo | No sentiment model — clearly marked "Future Feature" |
| Team — Agent Workload | 🟡 Demo | Simulated agent data |
| Team — AI Routing Rules | 🟡 Demo | Illustrative toggles — clearly marked |
| Admin Panel | 🟡 Demo | All simulated data |
| Settings | 🟡 Demo | UI-only, no persistence |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Web Frontend (Vercel)                             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │
│  │ Single Classify │  │  Batch Upload  │  │  Dashboard / Insights  │ │
│  └───────┬────────┘  └───────┬────────┘  └───────────┬────────────┘ │
└──────────┼───────────────────┼───────────────────────┼──────────────┘
           │                   │                       │
           ▼                   ▼                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (:8000)                         │
│  POST /predict    POST /predict/batch   GET /health  GET /model-info│
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │   Text Cleaning → TF-IDF Vectorizer → Category Model           │ │
│  │                                     → Priority Model           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

### Category Classifier (8 classes)
| Metric | Score |
|--------|-------|
| **Accuracy** | 85.78% |
| **F1 Score (macro)** | 86.05% |
| **Precision (macro)** | 88.01% |
| **Best Model** | Linear SVC |

### Priority Classifier (3 classes)
| Metric | Score |
|--------|-------|
| **Accuracy** | 82.93% |
| **F1 Score (macro)** | 82.42% |
| **Precision (macro)** | 82.48% |
| **Best Model** | Logistic Regression |

### Cross-Validation Comparison

| Model | Category F1 | Priority F1 |
|-------|------------|------------|
| **Linear SVC** ✅ | 0.851 | 0.814 |
| Logistic Regression | 0.847 | **0.816** ✅ |
| Random Forest | 0.842 | 0.815 |
| Multinomial NB | 0.765 | 0.760 |

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/shsumukha381-ui/FUTURE_ML_02.git
cd FUTURE_ML_02
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Dataset

Download the **Classification of IT Support Tickets** dataset from [Zenodo](https://zenodo.org/records/7648117) and place the CSV in `data/raw/`:

```
data/raw/all_tickets_processed_improved_v3.csv
```

### 4. Train the Models

```bash
python -m src.run_training
```

This takes ~5–10 minutes and produces:
- Trained models in `models/`
- Evaluation reports with confusion matrices in `reports/`

### 5. Start the Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 6. Open the Frontend

```bash
start frontend/index.html
```

Or use a local server:
```bash
python -m http.server 5500 --directory frontend
```

The frontend connects to `http://localhost:8000` by default.

---

## 🌐 Vercel Deployment

The frontend is deployed as a static site on Vercel.

### Deploy Your Own

1. Fork this repository
2. Go to [vercel.com](https://vercel.com) → New Project → Import your fork
3. Vercel auto-detects the `vercel.json` configuration
4. Click **Deploy**

> **Note**: The Vercel-hosted frontend needs a running backend API to make predictions. For the backend, deploy separately on [Render](https://render.com), [Railway](https://railway.app), or run locally.

### Configure API URL

Set the backend URL in your browser console or via environment:
```javascript
window.API_BASE_URL = 'https://your-backend-url.com';
```

---

## 🔌 API Reference

### `GET /health`
Health check.
```json
{ "status": "healthy", "model_loaded": true }
```

### `GET /model-info`
Model metadata and evaluation metrics.

### `POST /predict`
Classify a single ticket.
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My laptop screen is broken and I need a replacement urgently"}'
```
```json
{
  "category": "Hardware",
  "category_confidence": 0.98,
  "priority": "High",
  "priority_confidence": 0.97,
  "category_probabilities": { "Hardware": 0.98, "Access": 0.01, ... },
  "priority_probabilities": { "High": 0.97, "Medium": 0.02, "Low": 0.01 }
}
```

### `POST /predict/batch`
Classify multiple tickets via JSON.
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"tickets": [{"text": "Need password reset"}, {"text": "Laptop not turning on"}]}'
```

### `POST /predict/upload`
Classify tickets from CSV file upload.
```bash
curl -X POST http://localhost:8000/predict/upload -F "file=@tickets.csv"
```

---

## 🧠 How It Works

### Ticket Categorization
1. **Text Cleaning** — lowercase, remove punctuation, stopwords (NLTK), lemmatization
2. **Feature Extraction** — TF-IDF with 10,000 features, unigrams + bigrams
3. **Classification** — Linear SVC trained on 38,269 tickets, selected via 5-fold cross-validation

### Priority Prediction
The dataset lacks priority labels, so we derive them transparently:

| Signal | Logic |
|--------|-------|
| **Category urgency** | Access/Admin rights → High; Hardware/Storage → Medium; Others → Low |
| **Urgency keywords** | "urgent", "blocked", "critical", etc. bump priority up |
| **Text complexity** | Very long tickets (top 10%) indicate escalated issues |

The ML model then learns these patterns from text alone, predicting priority without running rules at inference time.

> 📖 **Full explanation**: See [notebooks/NOTEBOOK.md](notebooks/NOTEBOOK.md) for a plain-language guide written for non-technical stakeholders.

---

## 📁 Project Structure

```
├── data/raw/                    # Dataset CSV (gitignored)
├── notebooks/
│   └── NOTEBOOK.md              # Plain-language model explanation
├── src/
│   ├── config.py                # Dataset mapping, priority rules, hyperparameters
│   ├── preprocessing.py         # Text cleaning, priority derivation, data loading
│   ├── features.py              # TF-IDF & BoW feature extraction
│   ├── train.py                 # Multi-model training & artifact persistence
│   ├── evaluate.py              # Metrics, confusion matrices, report generation
│   └── run_training.py          # End-to-end training script
├── models/                      # Serialized models (gitignored, reproducible)
├── reports/                     # Evaluation plots & metrics
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── schemas.py               # Pydantic request/response models
│   └── requirements.txt
├── public/
│   ├── index.html               # 9-page SPA (SupportMind admin UI)
│   ├── styles.css               # Dark glassmorphism theme
│   └── app.js                   # API client, charts & UI logic
├── vercel.json                  # Vercel deployment config
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔧 Using a Different Dataset

The pipeline is dataset-agnostic. To swap datasets:

1. Place your CSV in `data/raw/`
2. Edit `src/config.py`:
```python
DATASET_CONFIG = {
    "filename": "your_file.csv",
    "text_column": "your_text_column",
    "category_column": "your_category_column",
    "priority_column": "your_priority_column",  # or None
    "derive_priority": False,  # True if no priority column
}
```
3. Update `CATEGORY_LABELS` to match your dataset
4. Re-run: `python -m src.run_training`

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Pipeline** | scikit-learn, NLTK, pandas, numpy |
| **Feature Extraction** | TF-IDF (primary), Bag-of-Words (alternative) |
| **Backend** | FastAPI, Pydantic, uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS, Chart.js, Papa Parse |
| **Deployment** | Vercel (frontend), local/cloud (backend) |
| **Data** | Zenodo IT Support Tickets (47,837 records) |

---

## 📄 License

This project is for educational and portfolio purposes. Training data sourced from [Zenodo](https://zenodo.org/records/7648117).

---

<p align="center">Built with ❤️ as part of Future Interns ML Internship</p>
