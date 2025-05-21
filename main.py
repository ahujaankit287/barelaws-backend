import fire
import uvicorn
from fastapi import FastAPI
from step_1_chunking.chunk_default import run_chunking
from step_2_embedding.embed_default import run_embedding
from step_3_storing_with_meta_data.vector_db_unchunked import run_vector_db_insertion
from step_4_semantic_search.basic_semantic_search import query_documents
from step_5_summarise_result.openai_gpt_4o import generate_summary

app = FastAPI()

# --- PIPELINE COMMANDS ---

def index():
    """Run full indexing pipeline"""
    print("🔹 Step 1: Chunking")
    run_chunking()
    print("✅ Chunking complete")

    print("🔹 Step 2: Embedding")
    run_embedding()
    print("✅ Embedding complete")

    print("🔹 Step 3: Vector DB Insertion")
    run_vector_db_insertion()
    print("✅ Documents inserted into vector DB")

def query(query_text: str):
    """Run semantic query + summary"""
    print("🔹 Querying...")
    results = query_documents(query_text)
    print("✅ Top Results:")
    for r in results:
        print(f"  - {r['text'][:80]}...")

    print("\n🔹 Generating Summary...")
    summary = generate_summary(results)
    print("\n✅ Summary:\n", summary)


# --- API ENDPOINTS ---

@app.get("/")
def health():
    return {"status": "ok", "message": "Semantic Search API is running"}

@app.get("/query")
def query_endpoint(q: str):
    results = query_documents(q)
    summary = generate_summary(results)
    return {
        "query": q,
        "top_matches": results,
        "summary": summary
    }


# --- MAIN ENTRY POINT ---

def serve(port: int = 8000):
    """Serve API for prod use"""
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

def cli():
    """Expose commands to CLI"""
    fire.Fire({
        "index": index,
        "query": query,
        "serve": serve,
    })

if __name__ == "__main__":
    cli()