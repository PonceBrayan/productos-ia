import os, uvicorn
if __name__ == "__main__":
    os.environ.setdefault("MODEL_DIR", "../artefactos")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)