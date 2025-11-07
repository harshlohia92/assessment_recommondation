from pathlib import Path
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rich.console import Console
from rich.table import Table
import google.generativeai as genai

load_dotenv()
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
VECTORSTORE_DIR = Path("/tmp/vectorstore")  # Cloud-friendly temporary directory
COLLECTION_NAME = "ASSESSMENTS"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

LOG_FILE = Path("/tmp/recommendation_history.json")  # Cloud-friendly

console = Console()
llm = None
vector_store = None


def initialize_component():
    global llm, vector_store

    if llm is None:
        console.print("[cyan]Initializing Gemini LLM...[/cyan]")
        if not GEMINI_API_KEY or not GEMINI_MODEL:
            raise ValueError("GEMINI_API_KEY or GEMINI_MODEL not set in environment variables!")
        genai.configure(api_key=GEMINI_API_KEY)
        llm = genai.GenerativeModel(GEMINI_MODEL)

    if vector_store is None:
        # Use HF embeddings
        embeddings = HuggingFaceEmbeddings(
            model=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True},
        )
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )


def load_assessments(json_file):  # Accept single file
    docs = []
    json_path = Path(json_file)
    if not json_path.exists():
        console.print(f"[red]Error: File {json_file} does not exist![/red]")
        return docs

    with open(json_path, "r") as f:
        data = json.load(f)
        if isinstance(data, list):
            for assessment in data:
                if not all(k in assessment for k in ["name", "category", "skills_measured", "job_roles", "difficulty", "duration_minutes", "id"]):
                    continue
                content = f"""
Assessment Name: {assessment['name']}
Category: {assessment['category']}
Skills measured: {', '.join(assessment['skills_measured'])}
Job roles: {', '.join(assessment['job_roles'])}
Difficulty: {assessment['difficulty']}
Duration: {assessment['duration_minutes']} minutes
"""
                docs.append(Document(page_content=content, metadata={'id': assessment["id"]}))
    return docs


def process_assessments(json_file):
    console.print("[green]Initialize component[/green]")
    initialize_component()

    console.print("[yellow]Loading assessments...[/yellow]")
    docs = load_assessments(json_file)

    if not docs:
        console.print("[red]No assessments loaded. Exiting...[/red]")
        return

    console.print("[yellow]Splitting text into chunks...[/yellow]")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[" ", "\n", "\n\n", "\t"],
        chunk_size=CHUNK_SIZE,
    )
    docs = text_splitter.split_documents(docs)

    console.print("[yellow]Adding chunks to vector DB...[/yellow]")
    vector_store.add_documents(docs)

    console.print("[green]✅ Vector DB ready![/green]")


def generate_answer(query):
    if vector_store is None:
        initialize_component()

    retriever = vector_store.as_retriever()
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a helpful AI assistant. Use the context below to recommend SHL assessments based on skills or job roles.

Context:
{context}

Question: {query}

Provide your answer in the following format:

Recommendation: <assessment name>
Skills measured: <skills>
Difficulty: <level>
Duration: <minutes>

Reasoning: <Explain why this assessment is suitable for the given role or skills.>
"""

    response = llm.generate_content(prompt)
    text = response.text.strip()

    # Cleanly split recommendation and reasoning
    recommendation_text = ""
    reasoning_text = ""

    if "Reasoning:" in text:
        recommendation_part, reasoning_part = text.split("Reasoning:", 1)
        recommendation_text = recommendation_part.strip()
        # Remove any repeated 'Recommendation:' label if model added it
        if recommendation_text.lower().startswith("recommendation:"):
            recommendation_text = recommendation_text[len("Recommendation:"):].strip()
        reasoning_text = reasoning_part.strip()
    else:
        recommendation_text = text
        reasoning_text = ""

    # Re-add 'Recommendation:' at the start for consistency
    recommendation_text = f"Recommendation: {recommendation_text}"

    save_to_json_log(query, recommendation_text, reasoning_text)

    return {
        "message": f"💾 Saved recommendation to {LOG_FILE.name}",
        "recommendation": recommendation_text,
        "reasoning": reasoning_text
    }


def save_to_json_log(query, recommendation, reasoning):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "recommendation": recommendation,
        "reasoning": reasoning
    }

    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    json_file = Path(__file__).parent / "assessment.json"

    process_assessments(json_file)

    console.print("\n[bold cyan]=== Assessment Recommendation Engine ===[/bold cyan]")
    console.print("Type 'exit' to quit.\n")

    while True:
        query = console.input("[bold green]Enter job role or skills:[/bold green] ").strip()
        if query.lower() in ["exit", "quit"]:
            console.print("[yellow]Exiting...[/yellow]")
            break

        result = generate_answer(query)

        console.print(f"\n[blue]{result['message']}[/blue]\n")

        table = Table(title="Recommendation Results", show_header=True, header_style="bold magenta")
        table.add_column("Recommendation", style="cyan", width=70)
        table.add_column("Reasoning", style="green", width=70)

        table.add_row(result["recommendation"], result["reasoning"])
        console.print(table)
