"""
ML Pipeline Deployment — FastAPI Application
Project 4: Enterprise-Grade Model Serving
Portfolio: Lockheed Martin ML Engineer
Author: Tyler
"""

import asyncio
import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from database import (
    init_db, log_prediction, log_drift_check,
    get_recent_predictions, get_prediction_stats,
    get_anomaly_timeline, get_top_drivers,
    start_simulator_session, end_simulator_session,
)
from drift import DriftBaseline, compute_drift
from simulator import (
    CMAPSSLoader, create_simulator,
    stop_simulator, list_simulators,
    get_dataset_info, get_available_fault_modes,
    MODEL_SENSORS,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR     = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH    = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "isolation_forest.pkl"))
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

API_VERSION  = "1.0.0"
SERVICE_NAME = "ml-pipeline-deployment"

# Thread pool for running blocking I/O without freezing the event loop
_executor = ThreadPoolExecutor(max_workers=4)


# ── Model Registry ────────────────────────────────────────────────────────────
class ModelRegistry:
    model           = None
    scaler          = None
    explainer       = None
    metadata: dict  = {}
    feature_names: list = []
    startup_time: float = 0.0

    @classmethod
    def _load_sync(cls):
        """All blocking I/O in one synchronous method — run in executor."""
        t0 = time.perf_counter()

        logger.info(f"Loading model from: {MODEL_PATH}")
        cls.model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded: {type(cls.model).__name__}")

        if os.path.exists(SCALER_PATH):
            cls.scaler = joblib.load(SCALER_PATH)
            logger.info("Scaler loaded.")
        else:
            logger.warning("No scaler.pkl — inputs will not be scaled.")

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as f:
                cls.metadata = json.load(f)
            cls.feature_names = cls.metadata.get("features", [])
            logger.info(f"Metadata loaded: {cls.metadata}")

        logger.info("Initializing SHAP TreeExplainer...")
        cls.explainer = shap.TreeExplainer(cls.model)
        logger.info("SHAP explainer ready.")

        cls.startup_time = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(f"All artifacts loaded in {cls.startup_time}ms.")

    @classmethod
    async def load(cls):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, cls._load_sync)

    @classmethod
    def is_ready(cls) -> bool:
        return cls.model is not None

    @classmethod
    def safe_expected_value(cls) -> float:
        ev = cls.explainer.expected_value
        if hasattr(ev, "__len__"):
            return float(ev[0])
        return float(ev)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    # Load ML artifacts in thread pool (blocking I/O — won't freeze event loop)
    await ModelRegistry.load()

    # Initialize database (async — fine as-is)
    await init_db()

    # Build drift baseline in thread pool
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _executor,
        DriftBaseline.load,
        ModelRegistry.scaler,
        ModelRegistry.feature_names,
    )

    # Pre-load FD001 in thread pool
    try:
        await loop.run_in_executor(_executor, CMAPSSLoader.load, "FD001")
        logger.info("C-MAPSS FD001 pre-loaded into cache.")
    except Exception as e:
        logger.warning(f"Could not pre-load C-MAPSS data: {e}")

    logger.info(f"{SERVICE_NAME} v{API_VERSION} ready.")
    yield
    logger.info(f"{SERVICE_NAME} shutting down.")
    _executor.shutdown(wait=False)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ML Pipeline Deployment",
    description=(
        "Enterprise-grade REST API for turbofan engine anomaly detection. "
        "Isolation Forest model with SHAP explainability, feature drift detection, "
        "real-time WebSocket streaming, NASA C-MAPSS fault injection simulator, "
        "and full prediction audit logging."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="Sensor vector: s9, s14, s4, s3, s17, s7, s12, s11, s2",
        example=[0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.02, -0.01],
    )

    @validator("features")
    def features_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("features list must not be empty.")
        return v


class BatchPredictRequest(BaseModel):
    samples: list[list[float]] = Field(..., min_items=1, max_items=512)


class DriftRequest(BaseModel):
    samples: list[list[float]] = Field(..., min_items=1, max_items=1000)


class FeatureContribution(BaseModel):
    rank: int
    feature: str
    shap_value: float
    direction: str
    abs_impact: float


class PredictResponse(BaseModel):
    prediction: int
    label: str
    anomaly_score: float
    latency_ms: float
    model_version: str


class ExplainResponse(BaseModel):
    prediction: int
    label: str
    anomaly_score: float
    base_value: float
    top_driver: str
    contributions: list[FeatureContribution]
    latency_ms: float
    model_version: str


class BatchPredictResponse(BaseModel):
    predictions: list[int]
    labels: list[str]
    anomaly_scores: list[float]
    anomaly_count: int
    anomaly_rate: float
    count: int
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    shap_ready: bool
    drift_ready: bool
    db_ready: bool
    service: str
    version: str
    startup_ms: float


# ── Utilities ─────────────────────────────────────────────────────────────────
def preprocess(features: np.ndarray) -> np.ndarray:
    if ModelRegistry.scaler is not None:
        return ModelRegistry.scaler.transform(features)
    return features


def run_inference(features: np.ndarray):
    X_scaled    = preprocess(features)
    predictions = ModelRegistry.model.predict(X_scaled)
    scores      = ModelRegistry.model.decision_function(X_scaled)
    return predictions, scores, X_scaled


def build_contributions(shap_values: np.ndarray) -> list[FeatureContribution]:
    sv    = shap_values[0] if shap_values.ndim == 2 else shap_values
    items = sorted(enumerate(sv), key=lambda x: abs(x[1]), reverse=True)
    return [
        FeatureContribution(
            rank=rank + 1,
            feature=ModelRegistry.feature_names[i] if i < len(ModelRegistry.feature_names) else f"feature_{i}",
            shap_value=round(float(val), 6),
            direction="increases_anomaly" if val > 0 else "decreases_anomaly",
            abs_impact=round(abs(float(val)), 6),
        )
        for rank, (i, val) in enumerate(items)
    ]


def _predict_features(features: list) -> dict:
    X    = np.array(features).reshape(1, -1)
    predictions, scores, _ = run_inference(X)
    pred = int(predictions[0])
    return {
        "prediction":    pred,
        "label":         "anomaly" if pred == -1 else "normal",
        "anomaly_score": round(float(scores[0]), 6),
    }


def _run_shap(features: list):
    """Run SHAP in executor-safe way — returns contributions + top_driver."""
    try:
        X = np.array(features).reshape(1, -1)
        _, _, X_scaled = run_inference(X)
        sv      = ModelRegistry.explainer.shap_values(X_scaled)
        contribs = build_contributions(sv)
        return contribs, contribs[0].feature if contribs else None
    except Exception:
        return [], None


# ── Middleware ────────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    ms       = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({ms:.1f}ms)")
    return response


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Full system health — model, SHAP, drift baseline, and database."""
    if not ModelRegistry.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready.")
    return HealthResponse(
        status="ok",
        model_ready=True,
        shap_ready=ModelRegistry.explainer is not None,
        drift_ready=DriftBaseline.is_ready(),
        db_ready=True,
        service=SERVICE_NAME,
        version=API_VERSION,
        startup_ms=ModelRegistry.startup_time,
    )


@app.get("/model/info", tags=["System"])
def model_info():
    """Model type, feature order, training metadata, scaler and SHAP status."""
    if not ModelRegistry.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_type":        type(ModelRegistry.model).__name__,
        "scaler_loaded":     ModelRegistry.scaler is not None,
        "shap_ready":        ModelRegistry.explainer is not None,
        "drift_ready":       DriftBaseline.is_ready(),
        "expected_features": ModelRegistry.feature_names,
        "n_features":        len(ModelRegistry.feature_names),
        "metadata":          ModelRegistry.metadata,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Single-sample anomaly detection.
    1 = normal, -1 = anomaly. Every prediction is logged to the audit database.
    """
    if not ModelRegistry.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready.")

    start = time.perf_counter()
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, _predict_features, request.features)
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=422, detail=f"Inference failed: {str(e)}")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    await log_prediction(
        endpoint="/predict",
        features=request.features,
        prediction=result["prediction"],
        label=result["label"],
        anomaly_score=result["anomaly_score"],
        latency_ms=latency_ms,
    )

    return PredictResponse(**result, latency_ms=latency_ms, model_version=API_VERSION)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest):
    """
    Batch anomaly detection — up to 512 samples.
    Returns anomaly_count and anomaly_rate for fleet-level monitoring.
    """
    if not ModelRegistry.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready.")

    def _batch():
        X = np.array(request.samples)
        return run_inference(X)

    start = time.perf_counter()
    try:
        loop = asyncio.get_event_loop()
        predictions, scores, _ = await loop.run_in_executor(_executor, _batch)
    except Exception as e:
        logger.exception("Batch inference error")
        raise HTTPException(status_code=422, detail=f"Batch inference failed: {str(e)}")

    latency_ms  = round((time.perf_counter() - start) * 1000, 2)
    pred_list   = predictions.tolist()
    n_anomalies = sum(1 for p in pred_list if p == -1)

    return BatchPredictResponse(
        predictions=pred_list,
        labels=["anomaly" if p == -1 else "normal" for p in pred_list],
        anomaly_scores=[round(s, 6) for s in scores.tolist()],
        anomaly_count=n_anomalies,
        anomaly_rate=round(n_anomalies / len(pred_list) * 100, 2),
        count=len(pred_list),
        latency_ms=latency_ms,
        model_version=API_VERSION,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain(request: PredictRequest):
    """
    SHAP-based prediction explanation.
    Returns ranked sensor contributions sorted by abs_impact.
    Positive SHAP = increases anomaly likelihood.
    """
    if not ModelRegistry.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready.")
    if ModelRegistry.explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not initialized.")

    start = time.perf_counter()
    try:
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(_executor, _predict_features, request.features)

        def _shap():
            X_raw = np.array(request.features).reshape(1, -1)
            _, scores, X_scaled = run_inference(X_raw)
            sv           = ModelRegistry.explainer.shap_values(X_scaled)
            contributions = build_contributions(sv)
            base_value   = ModelRegistry.safe_expected_value()
            return contributions, base_value, float(scores[0])

        contributions, base_value, score = await loop.run_in_executor(_executor, _shap)

    except Exception as e:
        logger.exception("SHAP explanation error")
        raise HTTPException(status_code=422, detail=f"Explanation failed: {str(e)}")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    pred       = result["prediction"]
    top_driver = contributions[0].feature if contributions else "unknown"

    await log_prediction(
        endpoint="/explain",
        features=request.features,
        prediction=pred,
        label=result["label"],
        anomaly_score=round(score, 6),
        latency_ms=latency_ms,
        top_driver=top_driver,
        shap_values={c.feature: c.shap_value for c in contributions},
    )

    return ExplainResponse(
        prediction=pred,
        label=result["label"],
        anomaly_score=round(score, 6),
        base_value=round(base_value, 6),
        top_driver=top_driver,
        contributions=contributions,
        latency_ms=latency_ms,
        model_version=API_VERSION,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DRIFT DETECTION ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/drift", tags=["Drift Detection"])
async def drift_check(request: DriftRequest):
    """
    KS-test drift detection comparing live sensor readings to training distribution.
    Returns per-sensor drift status, severity, and actionable recommendation.
    High-severity drift = model operating outside training domain.
    """
    if not DriftBaseline.is_ready():
        raise HTTPException(status_code=503, detail="Drift baseline not initialized.")

    start = time.perf_counter()
    try:
        X    = np.array(request.samples)
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(
            _executor, compute_drift, X, ModelRegistry.feature_names
        )
    except Exception as e:
        logger.exception("Drift detection error")
        raise HTTPException(status_code=422, detail=f"Drift check failed: {str(e)}")

    latency_ms         = round((time.perf_counter() - start) * 1000, 2)
    report["latency_ms"] = latency_ms

    ks_scores = {r["feature"]: r["ks_statistic"] for r in report.get("feature_reports", [])}
    await log_drift_check(
        features=request.samples[0] if request.samples else [],
        ks_scores=ks_scores,
        drift_detected=report["drift_detected"],
        drifted_sensors=report.get("drifted_features", []),
        latency_ms=latency_ms,
    )

    return report


@app.get("/drift/baseline", tags=["Drift Detection"])
def drift_baseline():
    """Returns the training distribution baseline used for drift comparison."""
    if not DriftBaseline.is_ready():
        raise HTTPException(status_code=503, detail="Drift baseline not initialized.")
    return {
        "feature_names": DriftBaseline.feature_names,
        "means":         DriftBaseline.means.tolist(),
        "stds":          DriftBaseline.stds.tolist(),
        "mins":          DriftBaseline.mins.tolist(),
        "maxs":          DriftBaseline.maxs.tolist(),
        "percentiles":   DriftBaseline.percentiles,
        "ks_threshold":  0.05,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATOR ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/simulator/datasets", tags=["Simulator"])
async def simulator_datasets():
    """Returns metadata for all 4 NASA C-MAPSS datasets."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, get_dataset_info)


@app.get("/simulator/fault-modes", tags=["Simulator"])
def simulator_fault_modes():
    """Returns available fault injection modes and affected sensors."""
    return get_available_fault_modes()


@app.get("/simulator/engines/{dataset}", tags=["Simulator"])
async def simulator_engines(dataset: str):
    """Lists available engine IDs for a given dataset."""
    try:
        loop = asyncio.get_event_loop()
        ids  = await loop.run_in_executor(
            _executor, CMAPSSLoader.get_engine_ids, dataset.upper()
        )
        return {"dataset": dataset.upper(), "engine_ids": ids, "count": len(ids)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset} not found.")


@app.get("/simulator/active", tags=["Simulator"])
def simulator_active():
    """Lists all currently running simulator sessions."""
    return {"active_simulators": list_simulators()}


@app.post("/simulator/stop/{session_key}", tags=["Simulator"])
def simulator_stop(session_key: str):
    """Stop a running simulator session."""
    stop_simulator(session_key)
    return {"stopped": session_key}


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET — LIVE STREAMING
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/simulate")
async def websocket_simulate(websocket: WebSocket):
    """
    WebSocket live engine simulation stream.

    Send config JSON after connecting:
    {
        "dataset": "FD001",
        "engine_id": null,
        "fault_mode": "bearing_wear",
        "fault_start_pct": 0.6,
        "speed_multiplier": 5.0
    }
    """
    await websocket.accept()
    session_key = None

    try:
        config_raw = await websocket.receive_text()
        config     = json.loads(config_raw)

        dataset          = config.get("dataset", "FD001").upper()
        engine_id        = config.get("engine_id", None)
        fault_mode       = config.get("fault_mode", "none")
        fault_start_pct  = float(config.get("fault_start_pct", 0.6))
        speed_multiplier = float(config.get("speed_multiplier", 5.0))

        loop = asyncio.get_event_loop()
        session_key, sim = await loop.run_in_executor(
            _executor, lambda: create_simulator(
                dataset=dataset,
                engine_id=engine_id,
                fault_mode=fault_mode,
                fault_start_pct=fault_start_pct,
                speed_multiplier=speed_multiplier,
            )
        )

        db_session_id = await start_simulator_session(
            dataset=dataset,
            engine_id=sim.engine_id,
            fault_mode=fault_mode,
        )

        await websocket.send_json({
            "type":         "session_start",
            "session_key":  session_key,
            "engine_id":    sim.engine_id,
            "dataset":      dataset,
            "fault_mode":   fault_mode,
            "total_cycles": sim.total_cycles,
        })

        anomaly_count = 0
        cycle_count   = 0

        async for reading in sim.stream():
            try:
                result = await loop.run_in_executor(
                    _executor, _predict_features, reading["features"]
                )
            except Exception:
                continue

            contributions = []
            top_driver    = None
            if result["prediction"] == -1 and ModelRegistry.explainer is not None:
                contribs, top_driver = await loop.run_in_executor(
                    _executor, _run_shap, reading["features"]
                )
                contributions = [c.dict() for c in contribs]

            cycle_count += 1
            if result["prediction"] == -1:
                anomaly_count += 1

            await log_prediction(
                endpoint="/ws/simulate",
                features=reading["features"],
                prediction=result["prediction"],
                label=result["label"],
                anomaly_score=result["anomaly_score"],
                latency_ms=0.0,
                top_driver=top_driver,
                source="simulator",
            )

            await websocket.send_json({
                "type":          "reading",
                "cycle":         reading["cycle"],
                "cycle_pct":     reading["cycle_pct"],
                "engine_id":     reading["engine_id"],
                "dataset":       reading["dataset"],
                "fault_mode":    reading["fault_mode"],
                "fault_active":  reading["fault_active"],
                "features":      reading["features"],
                "sensor_values": reading["sensor_values"],
                "prediction":    result["prediction"],
                "label":         result["label"],
                "anomaly_score": result["anomaly_score"],
                "top_driver":    top_driver,
                "contributions": contributions,
                "anomaly_count": anomaly_count,
                "timestamp":     reading["timestamp"],
            })

        await end_simulator_session(db_session_id, cycle_count, anomaly_count)
        await websocket.send_json({
            "type":          "session_end",
            "total_cycles":  cycle_count,
            "anomaly_count": anomaly_count,
            "anomaly_rate":  round(anomaly_count / max(cycle_count, 1) * 100, 2),
        })

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {session_key}")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        if session_key:
            stop_simulator(session_key)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD DATA ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard/stats", tags=["Dashboard"])
async def dashboard_stats():
    """Aggregate prediction statistics for the dashboard health panel."""
    return await get_prediction_stats()


@app.get("/dashboard/recent", tags=["Dashboard"])
async def dashboard_recent(limit: int = 50, label: Optional[str] = None):
    """Recent predictions — optionally filtered by 'normal' or 'anomaly'."""
    return await get_recent_predictions(limit=limit, label_filter=label)


@app.get("/dashboard/timeline", tags=["Dashboard"])
async def dashboard_timeline(hours: int = 24):
    """Hourly anomaly counts for the timeline chart."""
    return await get_anomaly_timeline(hours=hours)


@app.get("/dashboard/top-drivers", tags=["Dashboard"])
async def dashboard_top_drivers(limit: int = 10):
    """Most frequently flagged sensors across all anomaly predictions."""
    return await get_top_drivers(limit=limit)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Main:app", host="0.0.0.0", port=8000, reload=False)