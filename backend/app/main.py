from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI News API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict")
def predict(item: dict):
    text = item.get("text", "")
    # Placeholder prediction
    return {"label": "REAL", "prob": 0.51, "length": len(text)}
