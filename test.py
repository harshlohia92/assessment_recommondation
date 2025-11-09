import json
import pandas as pd
from pathlib import Path
from rag import process_assessments, generate_answer


json_file = Path(__file__).parent / "assisments" / "ass.json"
process_assessments(json_file)


test_file = Path(__file__).parent / "assisments"/ "test_queries.json"  
with open(test_file, "r") as f:
    test_data = json.load(f)


results = []
for item in test_data:
    query = item["query"]
    prediction = generate_answer(query)["recommendation"]
    results.append({"id": item["id"], "prediction": prediction})


submission_file = Path(__file__).parent / "harsh_lohia.csv"  
df = pd.DataFrame(results)
df.to_csv(submission_file, index=False)

print(f"Submission file created: {submission_file}")
