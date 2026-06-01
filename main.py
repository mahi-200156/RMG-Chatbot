from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import traceback

from schemas import EmployeeSearchRequest, JobSearchRequest, GeneralSearchRequest
from utils import (
    init_llm, init_embeddings, load_vectorstores,
    get_all_jobs, find_jobs_for_employee,
    find_employees_for_job, general_search
)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting RMG Chatbot")
    app.state.llm = init_llm()
    print("  ✅ Groq LLM ready")
    app.state.embeddings = init_embeddings()
    print("  ✅ HuggingFace embeddings ready")
    app.state.job_vs, app.state.emp_vs = load_vectorstores(app.state.embeddings)
    print("  ✅ Vector stores loaded")
    print("✅ Server ready!\n")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="RMG Job Matching Chatbot — Free Version",
    description="""
    AI-powered job matching using Groq (Llama 3.1 70B) + HuggingFace embeddings.
    100% free — no OpenAI API key needed.

    **Endpoints:**
    - GET  /jobs              → List all jobs with filters
    - POST /search/employee   → Find jobs for an employee
    - POST /search/job        → Find employees for a job
    - POST /search/general    → General natural language search
    """,
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/jobs", tags=["Part 1"])
def list_jobs(location: str = None, title: str = None):
    """List all available jobs with optional filters."""
    try:
        return get_all_jobs(location=location, title=title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/employee", tags=["Case 1"])
def search_for_employee(request: EmployeeSearchRequest):
    """Find best matching jobs for a bench employee."""
    try:
        result = find_jobs_for_employee(
            request.employee_name, request.top_k,
            app.state.llm, app.state.job_vs
        )
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/job", tags=["Case 2"])
def search_for_job(request: JobSearchRequest):
    """Find best matching bench employees for a job opening."""
    try:
        result = find_employees_for_job(
            request.job_id, request.top_k,
            app.state.llm, app.state.emp_vs
        )
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/general", tags=["Case 3"])
def search_general(request: GeneralSearchRequest):
    """General natural language search — hybrid SQL + RAG."""
    try:
        return general_search(
            request.query, request.top_k,
            app.state.llm, app.state.emp_vs
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy", "llm": "groq/llama-3.1-70b", "embeddings": "HuggingFace/all-MiniLM-L6-v2"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)