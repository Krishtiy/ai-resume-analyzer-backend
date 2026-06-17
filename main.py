from fastapi import FastAPI

app = FastAPI(title="AI Resume Analyzer API")

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "AI Resume Analyzer Backend is running successfully!"}