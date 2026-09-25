"""
FastAPI backend for the Support Ticket Classification & Prioritization system.

Endpoints:
    POST /predict        — Classify a single ticket
    POST /predict/batch  — Classify multiple tickets (JSON list or CSV upload)
    GET  /health         — Health check
    GET  /model-info     — Model version, training date, and metrics summary

Usage:
    cd support-ticket-classifier
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import sys
import csv
import logging
import uuid
import time
import zipfile
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Ensure the project root is on sys.path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import (
    TicketInput,
    BatchTicketInput,
    PredictionResult,
    BatchPredictionResult,
    HealthResponse,
    ModelInfoResponse,
    TrainRequest,
    TrainStatusResponse,
    TrainModelResult,
    TrainFoldResult,
)
from src.preprocessing import clean_text
from src.train import load_artifacts

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------
_model_state = {
    "vectorizer": None,
    "category_model": None,
    "priority_model": None,
    "metadata": None,
    "loaded": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup; cleanup on shutdown."""
    try:
        logger.info("Loading model artifacts...")
        vectorizer, cat_model, pri_model, metadata = load_artifacts()
        _model_state["vectorizer"] = vectorizer
        _model_state["category_model"] = cat_model
        _model_state["priority_model"] = pri_model
        _model_state["metadata"] = metadata
        _model_state["loaded"] = True
        logger.info(
            "Models loaded successfully (version %s, trained %s)",
            metadata["version"],
            metadata["trained_at"],
        )
    except FileNotFoundError as e:
        logger.error("Model artifacts not found: %s", e)
        logger.error("Run 'python -m src.run_training' to train models first.")
        _model_state["loaded"] = False
    yield
    logger.info("Shutting down API...")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Support Ticket Classifier API",
    description=(
        "Classifies support tickets into categories and priority levels "
        "using machine learning. Trained on IT service ticket data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Vercel frontend + localhost for dev
_cors_origins = [
    "https://future-ml-02.vercel.app",
    "https://future-ml-02-gac5i6v2-neural-networks.vercel.app",
    "http://localhost:5500",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:3000",
    "null",  # file:// origin sends 'null'
]
# Allow additional origins via environment variable (comma-separated)
import os
_extra_origins = os.environ.get("CORS_ORIGINS", "")
if _extra_origins:
    _cors_origins.extend([o.strip() for o in _extra_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo/portfolio project
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ensure_models_loaded():
    """Raise 503 if models aren't loaded."""
    if not _model_state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models not loaded. Train models first with: "
                "python -m src.run_training"
            ),
        )


def _predict_single(text: str) -> PredictionResult:
    """Run prediction on a single ticket text."""
    # Clean text using the same pipeline as training
    cleaned = clean_text(text)
    if not cleaned.strip():
        # If cleaning removes everything, use original text lowered
        cleaned = text.lower().strip()

    # Vectorize
    X = _model_state["vectorizer"].transform([cleaned])

    # Category prediction
    cat_model = _model_state["category_model"]
    cat_pred = cat_model.predict(X)[0]
    cat_proba = cat_model.predict_proba(X)[0]
    cat_classes = cat_model.classes_
    cat_confidence = float(np.max(cat_proba))
    cat_probs_dict = {
        str(cls): round(float(prob), 4) for cls, prob in zip(cat_classes, cat_proba)
    }

    # Priority prediction
    pri_model = _model_state["priority_model"]
    pri_pred = pri_model.predict(X)[0]
    pri_proba = pri_model.predict_proba(X)[0]
    pri_classes = pri_model.classes_
    pri_confidence = float(np.max(pri_proba))
    pri_probs_dict = {
        str(cls): round(float(prob), 4) for cls, prob in zip(pri_classes, pri_proba)
    }

    # Truncate text for response (keep first 500 chars)
    display_text = text[:500] + ("..." if len(text) > 500 else "")

    return PredictionResult(
        text=display_text,
        category=str(cat_pred),
        category_confidence=round(cat_confidence, 4),
        priority=str(pri_pred),
        priority_confidence=round(pri_confidence, 4),
        category_probabilities=cat_probs_dict,
        priority_probabilities=pri_probs_dict,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy" if _model_state["loaded"] else "degraded",
        model_loaded=_model_state["loaded"],
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return model version, training date, and evaluation metrics."""
    _ensure_models_loaded()

    metadata = _model_state["metadata"]
    eval_data = metadata.get("evaluation", {})

    return ModelInfoResponse(
        version=metadata.get("version", "unknown"),
        trained_at=metadata.get("trained_at", "unknown"),
        category_model=type(_model_state["category_model"]).__name__,
        priority_model=type(_model_state["priority_model"]).__name__,
        category_metrics=eval_data.get("category"),
        priority_metrics=eval_data.get("priority"),
        cross_validation=metadata.get("cross_validation"),
    )


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_single(ticket: TicketInput):
    """
    Classify a single support ticket.

    Returns the predicted category and priority with confidence scores
    and full probability distributions across all classes.
    """
    _ensure_models_loaded()

    logger.info(
        "Prediction request — text length: %d chars",
        len(ticket.text),
    )

    result = _predict_single(ticket.text)

    logger.info(
        "Prediction: category=%s (%.2f), priority=%s (%.2f)",
        result.category, result.category_confidence,
        result.priority, result.priority_confidence,
    )

    return result


@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["Prediction"])
async def predict_batch(batch: BatchTicketInput):
    """
    Classify multiple support tickets from a JSON list.

    Accepts a JSON body: `{"tickets": [{"text": "..."}, ...]}`
    Returns predictions for all tickets (max 1000).
    """
    _ensure_models_loaded()

    texts = [t.text for t in batch.tickets]

    if len(texts) > 1000:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(texts)} exceeds maximum of 1000."
        )

    logger.info("Batch prediction request — %d tickets", len(texts))

    predictions = [_predict_single(text) for text in texts]

    logger.info("Batch prediction complete — %d tickets processed", len(predictions))

    return BatchPredictionResult(
        predictions=predictions,
        total=len(predictions),
    )


@app.post("/predict/upload", response_model=BatchPredictionResult, tags=["Prediction"])
async def predict_upload(file: UploadFile = File(...)):
    """
    Classify tickets from a CSV file upload.

    The CSV must contain a text column named one of:
    'text', 'Document', 'ticket_text', 'description', 'Ticket Description'.
    Returns predictions for all rows (max 1000).
    """
    _ensure_models_loaded()

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        content = await file.read()
        decoded = content.decode("utf-8")
        df = pd.read_csv(io.StringIO(decoded))

        # Auto-detect text column
        text_col = None
        for candidate in ["text", "Document", "ticket_text", "description", "Ticket Description"]:
            if candidate in df.columns:
                text_col = candidate
                break
        if text_col is None:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must have a text column. Found columns: {list(df.columns)}. "
                       f"Expected one of: text, Document, ticket_text, description."
            )

        texts = df[text_col].dropna().astype(str).tolist()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    if not texts:
        raise HTTPException(status_code=400, detail="No ticket texts found in CSV.")

    if len(texts) > 1000:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(texts)} exceeds maximum of 1000."
        )

    logger.info("CSV upload prediction — %d tickets from %s", len(texts), file.filename)

    predictions = [_predict_single(text) for text in texts]

    logger.info("CSV prediction complete — %d tickets processed", len(predictions))

    return BatchPredictionResult(
        predictions=predictions,
        total=len(predictions),
    )


# ---------------------------------------------------------------------------
# Training Jobs — background retraining with progress tracking
# ---------------------------------------------------------------------------
_training_jobs = {}  # job_id -> dict with status, progress, results
_training_lock = threading.Lock()


def _run_training_job(job_id: str, config: TrainRequest):
    """
    Execute the full training pipeline in a background thread.

    Updates _training_jobs[job_id] with progress as each step completes,
    so the frontend can poll GET /train/status/{job_id} for live updates.
    """
    job = _training_jobs[job_id]
    start_time = time.time()

    try:
        # ---------------------------------------------------------------
        # Step 1: Load and prepare data
        # ---------------------------------------------------------------
        job["status"] = "running"
        job["current_step"] = "Loading and preparing dataset..."
        job["progress"] = 0.05

        from src.preprocessing import prepare_dataset
        from src.config import TFIDF_PARAMS, BOW_PARAMS

        # Override config from request
        import src.config as cfg
        cfg.CROSS_VALIDATION_FOLDS = config.cv_folds
        cfg.TEST_SIZE = config.test_split
        cfg.TFIDF_PARAMS["max_features"] = config.max_features
        cfg.BOW_PARAMS["max_features"] = config.max_features

        X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test, df = prepare_dataset()
        job["progress"] = 0.15
        job["current_step"] = "Extracting features..."

        # ---------------------------------------------------------------
        # Step 2: Feature extraction
        # ---------------------------------------------------------------
        from src.features import build_tfidf_vectorizer, build_bow_vectorizer, transform_features

        if config.feature_extraction == "tfidf":
            vectorizer, X_train_feat = build_tfidf_vectorizer(X_train)
        else:
            vectorizer, X_train_feat = build_bow_vectorizer(X_train)

        X_test_feat = transform_features(X_test, vectorizer)
        job["progress"] = 0.25
        job["current_step"] = "Training classifiers..."

        # ---------------------------------------------------------------
        # Step 3: Train classifiers with per-fold reporting
        # ---------------------------------------------------------------
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from src.train import _get_classifiers, save_artifacts

        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=42)

        # Determine which classifiers to run
        all_classifiers = _get_classifiers()
        if config.algorithm != "all":
            all_classifiers = [(n, c) for n, c in all_classifiers if n == config.algorithm]

        total_models = len(all_classifiers) * 2  # category + priority
        models_done = 0

        # --- Category classifiers ---
        cat_cv_results = {}
        cat_best_score = -1.0
        cat_best_model = None
        cat_best_name = ""

        for name, clf in all_classifiers:
            job["current_step"] = f"Training {name} (Category)..."
            try:
                f1_scores = cross_val_score(clf, X_train_feat, y_cat_train, cv=cv, scoring="f1_macro", n_jobs=-1)
                acc_scores = cross_val_score(clf, X_train_feat, y_cat_train, cv=cv, scoring="accuracy", n_jobs=-1)

                fold_results = []
                for i in range(len(f1_scores)):
                    fold_results.append(TrainFoldResult(
                        fold=i + 1,
                        accuracy=round(float(acc_scores[i]), 4),
                        f1=round(float(f1_scores[i]), 4),
                        loss=round(1.0 - float(acc_scores[i]), 4),
                    ))

                result = {
                    "mean_f1": float(np.mean(f1_scores)),
                    "std_f1": float(np.std(f1_scores)),
                    "mean_accuracy": float(np.mean(acc_scores)),
                    "std_accuracy": float(np.std(acc_scores)),
                }
                cat_cv_results[name] = result

                model_result = TrainModelResult(
                    name=name,
                    mean_accuracy=round(result["mean_accuracy"], 4),
                    std_accuracy=round(result["std_accuracy"], 4),
                    mean_f1=round(result["mean_f1"], 4),
                    std_f1=round(result["std_f1"], 4),
                    fold_results=fold_results,
                )
                job["models_completed"].append(model_result)

                if result["mean_f1"] > cat_best_score:
                    cat_best_score = result["mean_f1"]
                    cat_best_model = clf
                    cat_best_name = name

            except Exception as e:
                logger.error("Training %s (Category) failed: %s", name, e)
                cat_cv_results[name] = {"error": str(e)}

            models_done += 1
            job["progress"] = 0.25 + (models_done / total_models) * 0.5

        # Refit best category model on full training set
        if cat_best_model:
            job["current_step"] = f"Refitting {cat_best_name} on full training set..."
            cat_best_model.fit(X_train_feat, y_cat_train)

        # --- Priority classifiers ---
        pri_cv_results = {}
        pri_best_score = -1.0
        pri_best_model = None
        pri_best_name = ""

        for name, clf in _get_classifiers():
            if config.algorithm != "all" and name != config.algorithm:
                continue
            job["current_step"] = f"Training {name} (Priority)..."
            try:
                # Need fresh classifier instances for priority
                from src.train import _get_classifiers as get_clfs
                fresh_classifiers = {n: c for n, c in get_clfs()}
                clf_pri = fresh_classifiers[name]

                f1_scores = cross_val_score(clf_pri, X_train_feat, y_pri_train, cv=cv, scoring="f1_macro", n_jobs=-1)
                acc_scores = cross_val_score(clf_pri, X_train_feat, y_pri_train, cv=cv, scoring="accuracy", n_jobs=-1)

                result = {
                    "mean_f1": float(np.mean(f1_scores)),
                    "std_f1": float(np.std(f1_scores)),
                    "mean_accuracy": float(np.mean(acc_scores)),
                    "std_accuracy": float(np.std(acc_scores)),
                }
                pri_cv_results[name] = result

                if result["mean_f1"] > pri_best_score:
                    pri_best_score = result["mean_f1"]
                    pri_best_model = clf_pri
                    pri_best_name = name

            except Exception as e:
                logger.error("Training %s (Priority) failed: %s", name, e)
                pri_cv_results[name] = {"error": str(e)}

            models_done += 1
            job["progress"] = 0.25 + (models_done / total_models) * 0.5

        # Refit best priority model
        if pri_best_model:
            job["current_step"] = f"Refitting {pri_best_name} on full training set..."
            pri_best_model.fit(X_train_feat, y_pri_train)

        # ---------------------------------------------------------------
        # Step 4: Evaluate on test set
        # ---------------------------------------------------------------
        job["progress"] = 0.80
        job["current_step"] = "Evaluating on test set..."

        from src.evaluate import evaluate_model
        from src.config import CATEGORY_LABELS, PRIORITY_LABELS

        cat_metrics = evaluate_model(
            cat_best_model, X_test_feat, y_cat_test, CATEGORY_LABELS, "Category"
        ) if cat_best_model else {}

        pri_metrics = evaluate_model(
            pri_best_model, X_test_feat, y_pri_test, PRIORITY_LABELS, "Priority"
        ) if pri_best_model else {}

        job["category_results"] = {
            "cv": cat_cv_results,
            "evaluation": cat_metrics,
            "best_model": cat_best_name,
        }
        job["priority_results"] = {
            "cv": pri_cv_results,
            "evaluation": pri_metrics,
            "best_model": pri_best_name,
        }

        # ---------------------------------------------------------------
        # Step 5: Save artifacts
        # ---------------------------------------------------------------
        job["progress"] = 0.90
        job["current_step"] = "Saving model artifacts..."

        if cat_best_model and pri_best_model:
            save_artifacts(
                vectorizer=vectorizer,
                category_model=cat_best_model,
                priority_model=pri_best_model,
                category_cv_results=cat_cv_results,
                priority_cv_results=pri_cv_results,
                category_eval=cat_metrics,
                priority_eval=pri_metrics,
            )

            # Reload models into the running API
            _model_state["vectorizer"] = vectorizer
            _model_state["category_model"] = cat_best_model
            _model_state["priority_model"] = pri_best_model

            # Reload metadata
            from src.config import MODELS_DIR
            import json
            with open(MODELS_DIR / "metadata.json") as f:
                _model_state["metadata"] = json.load(f)
            _model_state["loaded"] = True

        # Mark best model in completed list
        for m in job["models_completed"]:
            if m.name == cat_best_name:
                m.is_best = True

        job["progress"] = 1.0
        job["status"] = "completed"
        job["current_step"] = "Training complete!"
        job["elapsed_seconds"] = round(time.time() - start_time, 1)

        logger.info(
            "Training job %s completed in %.1fs — Best category: %s, Best priority: %s",
            job_id, job["elapsed_seconds"], cat_best_name, pri_best_name,
        )

    except Exception as e:
        logger.error("Training job %s failed: %s", job_id, e, exc_info=True)
        job["status"] = "failed"
        job["error"] = str(e)
        job["elapsed_seconds"] = round(time.time() - start_time, 1)


@app.post("/train", tags=["Training"])
async def start_training(config: TrainRequest):
    """
    Start a model training job in the background.

    Returns a job_id that can be polled via GET /train/status/{job_id}.
    Only one training job can run at a time.
    """
    # Check if a training job is already running
    with _training_lock:
        for jid, job in _training_jobs.items():
            if job["status"] == "running":
                raise HTTPException(
                    status_code=409,
                    detail=f"Training job {jid} is already running. Wait for it to complete.",
                )

    job_id = str(uuid.uuid4())[:8]
    _training_jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "current_step": "Queued...",
        "models_completed": [],
        "category_results": None,
        "priority_results": None,
        "error": None,
        "elapsed_seconds": None,
        "config": config.model_dump(),
    }

    logger.info("Starting training job %s with config: %s", job_id, config.model_dump())

    thread = threading.Thread(
        target=_run_training_job,
        args=(job_id, config),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/train/status/{job_id}", response_model=TrainStatusResponse, tags=["Training"])
async def train_status(job_id: str):
    """Poll the status of a training job."""
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found.")

    job = _training_jobs[job_id]
    return TrainStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        current_step=job["current_step"],
        models_completed=job["models_completed"],
        category_results=job["category_results"],
        priority_results=job["priority_results"],
        error=job["error"],
        elapsed_seconds=job["elapsed_seconds"],
    )


@app.get("/train/export", tags=["Training"])
async def export_model():
    """
    Download the current trained model artifacts as a ZIP file.

    Includes: vectorizer.joblib, category_model.joblib, priority_model.joblib, metadata.json
    """
    from src.config import MODELS_DIR

    required_files = [
        MODELS_DIR / "vectorizer.joblib",
        MODELS_DIR / "category_model.joblib",
        MODELS_DIR / "priority_model.joblib",
        MODELS_DIR / "metadata.json",
    ]

    for f in required_files:
        if not f.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model artifact {f.name} not found. Train a model first.",
            )

    # Create ZIP in memory
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in required_files:
            zf.write(f, f.name)

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=supportmind_model.zip"},
    )

