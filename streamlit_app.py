import streamlit as st
from pathlib import Path
from rag import process_assessments, generate_answer

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 SHL Assessment Recommendation Engine")
st.markdown("Enter a job role or skills, and get the best SHL assessment with reasoning.")

# Use the JSON file, not a folder
json_file = Path(__file__).parent / "ass.json"
process_assessments(json_file)

query = st.text_input("Enter job role or skills:")

if query:
    with st.spinner("Generating recommendation..."):
        result = generate_answer(query)

    st.success(result["message"])
    st.subheader("Recommendation")
    st.text(result["recommendation"])
    st.subheader("Reasoning")
    st.text(result["reasoning"])
