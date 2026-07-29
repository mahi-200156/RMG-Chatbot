# RMG Job Matching Chatbot

> AI-powered internal talent matching system using
> semantic search and hybrid retrieval to match
> bench employees with suitable job openings.

# Live Demo
 [**Try the app here**] (https://rmg-chatbot.streamlit.app/)

# Problem Statement
When an employee's process closes and they go on bench, Resource Management Group (RMG) must manually screen CVs to find suitable internal roles — slow, inconsistent, and dependent on recruiter memory.

This system automates talent matching using semantic AI — finding the right match even when skill descriptions don't match word-for-word.

# Architecture

User Query
    │
    ▼
Streamlit UI (4 Modes)
    │
    ▼
FastAPI Backend
    │
    ├── /chat → ReAct Agent (3 Tools)
    │   ├── find_jobs_tool      → Employee → Jobs
    │   ├── find_employees_tool → Job → Employees
    │   └── search_bench_tool   → General Search
    │
    ├── /search/employee → Case 1 (direct)
    ├── /search/job      → Case 2 (direct)
    ├── /search/general  → Case 3 Hybrid Search
    └── /jobs            → Browse all openings
    
SQL keyword search first
→ RAG semantic fallback if insufficient
→ LLM natural language answer

# Features

**4 Search Modes in UI:**
- 💬 **Smart Chat** — Agent decides which search to run
- 👤 **Employee → Jobs** — Find jobs for a named employee
- 💼 **Job → Employees** — Find candidates for a role
- 📋 **Browse All Jobs** — Filter and explore openings

**3 Retrieval Cases:**
- **Case 1:** Employee name → SQL fetches skills →
  FAISS semantic search → Ranked jobs
- **Case 2:** Job ID → SQL fetches requirements →
  FAISS semantic search → Ranked employees
- **Case 3:** General query → SQL keyword search first →
  RAG fallback if insufficient → Merged results

# Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| API | FastAPI |
| LLM | Groq Llama 3.3 70B (Free) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (Free, Local) |
| Vector Store | FAISS (Local, No API Cost) |
| Database | SQLite |
| Orchestration | LangChain |
| Agent Pattern | ReAct |

**Total API Cost: $0** — Fully open source stack

# Data

- **15 bench employees** across skills like Python,
  ML, LangChain, Java, React, DevOps, Data Engineering
- **12 job openings** across AI Engineer, Data Scientist, Frontend, Backend, DevOps, ML Engineer, and more
- **Locations:** Mumbai, Bangalore, Hyderabad, Chennai,  Pune, Delhi, Ahmedabad

# Run Locally

### Prerequisites
- Python 3.9+
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Setup

# 1. Clone the repo
git clone https://github.com/maahii6/rmg-chatbot
cd rmg-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add API key
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Database and vector stores are included
#    No setup needed — just run the app

# 5. Start Streamlit UI
streamlit run file.py

# 6. OR start FastAPI server
uvicorn main:app --reload
# Open: http://localhost:8000/docs

# API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | Smart agent — all cases |
| `/jobs` | GET | List jobs with filters |
| `/search/employee` | POST | Employee → Jobs |
| `/search/job` | POST | Job → Employees |
| `/search/general` | POST | Hybrid search |
| `/health` | GET | Health check |

# Why This Is Different From Keyword Search

SQL keyword search requires exact matches.
This system finds matches by **meaning**:

```
Job needs:  "Python, LLM, RAG, LangChain"
Employee has: "AI development, vector databases,
               language model deployment"

Keyword search: NO MATCH ❌
Semantic search: STRONG MATCH ✅ (Score: 0.82)
```

# Project Structure

rmg-chatbot/
├── main.py              # FastAPI + all endpoints
├── utils.py             # Core logic + agent tools
├── setup_data.py        # Creates DB + vector stores
├── schemas.py           # Pydantic models
├── file.py              # Streamlit UI (4 modes)
├── requirements.txt
├── rmg_database.db      # SQLite (employees + jobs)
├── vectorstore_jobs/    # FAISS job index
├── vectorstore_employees/ # FAISS employee index
└── screenshots/         # Demo screenshots
