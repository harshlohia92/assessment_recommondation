from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from rag import process_assessments, generate_answer  
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SHL Assessment Recommendation Engine")


json_file = Path(__file__).parent / "assisments" / "ass.json"
process_assessments(json_file)


class QueryRequest(BaseModel):
    query: str


@app.post("/recommend")
async def recommend_post(request: QueryRequest):
    """
    Recommend assessment based on JSON POST body:
    {
        "query": "ai engineer"
    }
    """
    try:
        result = generate_answer(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend")
async def recommend_get(query: str = Query(..., description="Job role or skills to get assessment recommendation")):
    """
    Recommend assessment via query string:
    Example: /recommend?query=ai+engineer
    """
    try:
        result = generate_answer(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT dynamically
    uvicorn.run("server:app", host="0.0.0.0", port=port)
