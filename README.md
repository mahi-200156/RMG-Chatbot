#  RMG Job Matching Chatbot

> AI-powered internal talent matching system using
> semantic search and hybrid retrieval to match
> bench employees with suitable job openings.

##  Problem Statement
When an employee's process closes and they go on
bench, Resource Management Group (RMG) needs to
quickly find suitable internal roles. Manual CV
screening is slow and inconsistent.

This system automates talent matching using
semantic AI search — replacing manual effort
with intelligent, explainable recommendations.

##  Architecture
User Query
│
FastAPI Backend
├── Case 1: Employee → Jobs
│   SQL (get skills) → FAISS semantic search
│   → LLM match explanation
├── Case 2: Job → Employees
│   SQL (get requirements) → FAISS search
│   → LLM match explanation
└── Case 3: General Query
SQL keyword search first
→ RAG semantic fallback if insufficient
→ LLM natural language answer

##  Features
- **Part 1:** Browse all job openings
  with location and title filters
- **Case 1:** Find best jobs for a bench employee
  using their skill profile
- **Case 2:** Find best employees for an urgent
  job opening (reverse matching)
- **Case 3:** Natural language queries with
  hybrid SQL + RAG retrieval

  ##  Tech Stack

| Component   | Technology                        |
|-------------|-----------------------------------|
| API         | FastAPI                           |
| LLM         | Groq                              |
| Embeddings  | HuggingFace                       | 
| Vector DB   | FAISS                             |
| Database    | SQLite                            |
| Orchestration | LangChain                       |
