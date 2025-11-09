import os
import json
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from rag import generate_answer, process_assessments

app = FastAPI(title="Assessment Recommendation API")


@app.on_event("startup")
def startup_event():
    json_file = Path(__file__).parent / "assessments" / "ass.json"
    process_assessments(json_file)
    print("Assessments processed and vector DB initialized.")


@app.get("/")
def home():
    return {"message": "Assessment Recommendation API is running!"}


@app.api_route("/recommend", methods=["GET", "HEAD"])
def recommend(query: str = Query(..., description="Job role or skill to recommend assessments for")):
    try:
        result = generate_answer(query)

        history = {
            "query": query,
            "recommendation": result.get("recommendation"),
            "reasoning": result.get("reasoning"),
        }
        with open("recommendation_history.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(history, ensure_ascii=False) + "\n")

        return JSONResponse(
            content={
                "message": "Saved recommendation to recommendation_history.json",
                **result,
            },
            ensure_ascii=False
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
