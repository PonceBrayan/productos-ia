import os, json, joblib, numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# === Artefactos sin subcarpeta ===
MODEL_DIR = os.getenv("MODEL_DIR", "../artefactos")
PIPE_PATH   = os.path.join(MODEL_DIR, "pipeline_mlp_ohe.joblib")
SCHEMA_PATH = os.path.join(MODEL_DIR, "input_schema.json")
LABEL_PATH  = os.path.join(MODEL_DIR, "label_map.json")
POLICY_PATH = os.path.join(MODEL_DIR, "decision_policy.json")

# === Carga de artefactos ===
pipe = joblib.load(PIPE_PATH)
input_schema = json.load(open(SCHEMA_PATH, "r", encoding="utf-8"))
label_map = {int(k): v for k, v in json.load(open(LABEL_PATH, "r", encoding="utf-8")).items()}
policy = json.load(open(POLICY_PATH, "r", encoding="utf-8"))

FEATURES = list(input_schema["properties"].keys())
CLASSES = [label_map[i] for i in sorted(label_map)]
DOMS = {k: v["enum"] for k, v in input_schema["properties"].items()}

def _norm(d: dict): 
    return {f: str(d[f]).strip().replace("\n"," ").title() for f in FEATURES}

def _check(d: dict):
    bad = {f: {"value": d[f], "allowed": DOMS[f]} for f in FEATURES if d[f] not in DOMS[f]}
    if bad: 
        raise HTTPException(status_code=422, detail={"invalid_enum": bad})

def _policy(proba: np.ndarray) -> str:
    mx = float(proba.max()); ta = policy["thresholds"]["accept"]; tr = policy["thresholds"]["review"]
    return "Aceptar" if mx >= ta else ("Revisar" if mx >= tr else "Incierto")

app = FastAPI(title="Desercion API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Payload(BaseModel):
    PromedioPonderado: str; DeudaFinanciera: str; SintomasDepresion: str; Edad: str
    SituacionLaboral: str; Trasladado: str; Asistencia: str
    @field_validator("*")
    @classmethod
    def norm(cls, v): return str(v).strip().replace("\n"," ").title()

@app.get("/health")
def health():
    return {"status":"ok","model_dir": os.path.abspath(MODEL_DIR),
            "features":FEATURES,"classes":CLASSES}

@app.get("/schema")
def schema(): 
    return input_schema


def _to_df(row: dict) -> pd.DataFrame:
    # garantiza orden de columnas
    return pd.DataFrame([row], columns=FEATURES)

@app.post("/predict")
def predict(inp: Payload):
    x = _norm(inp.model_dump())
    _check(x)
    X_df = _to_df(x)                 # <-- usar DataFrame (1x7)
    proba = pipe.predict_proba(X_df)[0]
    order = np.argsort(pipe.classes_) if hasattr(pipe, "classes_") else np.arange(len(CLASSES))
    proba = proba[order]
    idx = int(np.argmax(proba))
    return {
        "prediction": {"index": idx, "label": CLASSES[idx]},
        "probabilities": {CLASSES[i]: float(proba[i]) for i in range(len(CLASSES))},
        "decision": _policy(proba)
    }

# (si tienes /predict_batch, ajústalo también)
from typing import List

class BatchPayload(BaseModel):
    items: List[Payload]

@app.post("/predict_batch")
def predict_batch(batch: BatchPayload):
    rows = []
    for item in batch.items:
        row = _norm(item.model_dump())
        _check(row)
        rows.append(row)
    X_df = pd.DataFrame(rows, columns=FEATURES)  # <-- DataFrame batch
    prob = pipe.predict_proba(X_df)
    order = np.argsort(pipe.classes_) if hasattr(pipe, "classes_") else np.arange(len(CLASSES))
    prob = prob[:, order]
    idxs = prob.argmax(axis=1)
    return {
        "results": [
            {
                "prediction": {"index": int(i), "label": CLASSES[int(i)]},
                "probabilities": {CLASSES[j]: float(p[j]) for j in range(len(CLASSES))}
            }
            for i, p in zip(idxs, prob)
        ]
    }   
    