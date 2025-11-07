from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from rag import process_assessments, generate_answer

app = FastAPI(title="SHL Assessment Recommendation API")


json_folder = Path(__file__).parent / "assessment_files"
process_assessments(json_folder)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Welcome to SHL Assessment Recommendation API"}

@app.post("/recommend")
def recommend(request: QueryRequest):
    result = generate_answer(request.query)
    return result
