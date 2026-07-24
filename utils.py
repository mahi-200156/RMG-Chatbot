
import os
import re
import sqlite3
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# Global vectorstore references for agent tools
job_vs_global = None
emp_vs_global = None

def set_vs_global(job_vs, emp_vs):
    global job_vs_global, emp_vs_global
    job_vs_global = job_vs
    emp_vs_global = emp_vs

load_dotenv()

def init_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )


def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def load_vectorstores(embeddings):
    """Load pre-built FAISS indices from disk."""
    job_vs = FAISS.load_local(
        "vectorstore_jobs", embeddings,
        allow_dangerous_deserialization=True
    )
    emp_vs = FAISS.load_local(
        "vectorstore_employees", embeddings,
        allow_dangerous_deserialization=True
    )
    return job_vs, emp_vs


def get_db_connection():
    conn = sqlite3.connect("rmg_database.db")
    conn.row_factory = sqlite3.Row
    return conn


# PART 1: LIST ALL JOBS

def get_all_jobs(location: str = None, title: str = None) -> dict:
    """Fetch all available jobs with optional filters. No AI needed."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM jobs WHERE 1=1"
    params = []

    if location:
        query += " AND LOWER(location) LIKE LOWER(?)"
        params.append(f"%{location}%")
    if title:
        query += " AND LOWER(title) LIKE LOWER(?)"
        params.append(f"%{title}%")

    query += " ORDER BY title ASC"
    rows = cursor.execute(query, params).fetchall()
    conn.close()

    return {
        "total_jobs": len([dict(r) for r in rows]),
        "filters_applied": {"location": location, "title": title},
        "jobs": [dict(r) for r in rows]
    }


# CASE 1: EMPLOYEE → MATCHING JOBS

def find_jobs_for_employee(employee_name: str, top_k: int, llm, job_vectorstore) -> dict:
    """
    Given employee name → fetch their skills → semantic search jobs.
    """
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM employees WHERE LOWER(name) LIKE LOWER(?)",
        (f"%{employee_name}%",)
    ).fetchone()
    conn.close()

    if not row:
        return {"error": f"No employee found matching '{employee_name}'"}

    employee = dict(row)
    print(f"[Case 1] Employee: {employee['name']} | Skills: {employee['skills']}")

    # Semantic search — employee skills vs job embeddings
    search_query = f"Skills: {employee['skills']} Designation: {employee['designation']}"
    results_with_scores = job_vectorstore.similarity_search_with_score(search_query, k=top_k)

    raw_matches = []
    jobs_context = ""
    for i, (doc, score) in enumerate(results_with_scores):
        similarity = round(1 / (1 + score), 3)
        raw_matches.append({
            "job_id": doc.metadata["job_id"],
            "title": doc.metadata["title"],
            "location": doc.metadata["location"],
            "openings": doc.metadata["openings"],
            "similarity_score": similarity,
            "content": doc.page_content
        })
        jobs_context += f"\nJob {i+1}:\n{doc.page_content}\nSimilarity: {similarity}\nJob ID: {doc.metadata['job_id']}\n---"

    # LLM generates match explanations
    prompt = f"""You are an RMG specialist helping place employees into suitable roles.

EMPLOYEE:
Name: {employee['name']}
Designation: {employee['designation']}
Skills: {employee['skills']}
Experience: {employee['experience']} years
Location: {employee['location']}

MATCHED JOBS:
{jobs_context}

For each job write ONE sentence explaining why it suits this employee.
Then write a 2-sentence overall recommendation.

FORMAT:
JOB_ANALYSIS:
[Job ID]: [reason]
...
SUMMARY:
[recommendation]"""

    response = llm.invoke(prompt).content
    job_reasons = _parse_job_reasons(response, raw_matches)
    summary = _extract_summary(response)

    return {
        "employee_name": employee["name"],
        "employee_skills": employee["skills"],
        "designation": employee["designation"],
        "experience": employee["experience"],
        "matched_jobs": [
            {**m, "match_reason": job_reasons.get(m["job_id"], "Strong skill alignment.")}
            for m in raw_matches
        ],
        "summary": summary
    }


# CASE 2: JOB → MATCHING EMPLOYEES

def find_employees_for_job(job_id: str, top_k: int, llm, emp_vectorstore) -> dict:
    """
    Given job ID → fetch requirements → semantic search employees.
    """
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM jobs WHERE LOWER(job_id) = LOWER(?)", (job_id,)
    ).fetchone()
    conn.close()

    if not row:
        return {"error": f"No job found with ID '{job_id}'"}

    job = dict(row)
    print(f"[Case 2] Job: {job['title']} | Skills needed: {job['required_skills']}")

    # Semantic search — job requirements vs employee embeddings
    search_query = f"Required Skills: {job['required_skills']} Role: {job['title']}"
    results_with_scores = emp_vectorstore.similarity_search_with_score(search_query, k=top_k)

    raw_matches = []
    emp_context = ""
    for i, (doc, score) in enumerate(results_with_scores):
        similarity = round(1 / (1 + score), 3)
        raw_matches.append({
            "emp_id": doc.metadata["emp_id"],
            "name": doc.metadata["name"],
            "designation": doc.metadata["designation"],
            "location": doc.metadata["location"],
            "similarity_score": similarity,
            "content": doc.page_content
        })
        emp_context += f"\nCandidate {i+1}:\n{doc.page_content}\nSimilarity: {similarity}\nEmp ID: {doc.metadata['emp_id']}\n---"

    # LLM generates match explanations
    prompt = f"""You are an RMG specialist filling an urgent job opening.

JOB REQUIREMENT:
Job ID: {job['job_id']}
Title: {job['title']}
Required Skills: {job['required_skills']}
Location: {job['location']}
Openings: {job['openings']}
Experience Needed: {job['experience_required']} years

MATCHED CANDIDATES (all on bench):
{emp_context}

For each candidate write ONE sentence explaining their fit.
Then write a 2-sentence recommendation of the best candidate(s).

FORMAT:
EMP_ANALYSIS:
[Emp ID]: [reason]
...
SUMMARY:
[recommendation]"""

    response = llm.invoke(prompt).content
    emp_reasons = _parse_emp_reasons(response, raw_matches)
    summary = _extract_summary(response)

    # Fetch full skills for each match
    matched_employees = []
    conn = get_db_connection()
    for m in raw_matches:
        skills_row = conn.execute(
            "SELECT skills FROM employees WHERE emp_id=?", (m["emp_id"],)
        ).fetchone()
        matched_employees.append({
            **m,
            "skills": skills_row["skills"] if skills_row else "",
            "match_reason": emp_reasons.get(m["emp_id"], "Strong skill alignment.")
        })
    conn.close()

    return {
        "job_id": job["job_id"],
        "job_title": job["title"],
        "required_skills": job["required_skills"],
        "openings": job["openings"],
        "matched_employees": matched_employees,
        "summary": summary
    }


# CASE 3: GENERAL HYBRID SEARCH
# SQL keyword search → if insufficient → RAG semantic fallback


def general_search(query: str, top_k: int, llm, emp_vectorstore) -> dict:
    """
    Hybrid search for natural language queries.
    SQL first → RAG fallback → LLM synthesis.
    """
    print(f"[Case 3] Query: {query}")

    keywords = _extract_keywords(query)
    print(f"[Case 3] Keywords: {keywords}")

    sql_results = []
    rag_results = []
    search_mode = ""

    # ─ Step 1: SQL keyword search ─
    if keywords:
        conn = get_db_connection()
        conditions, params = [], []
        for kw in keywords:
            conditions.append(
                "(LOWER(skills) LIKE LOWER(?) OR LOWER(designation) LIKE LOWER(?)"
                " OR LOWER(location) LIKE LOWER(?) OR LOWER(name) LIKE LOWER(?))"
            )
            params.extend([f"%{kw}%"] * 4)

        rows = conn.execute(
            f"SELECT * FROM employees WHERE status='bench' AND ({' OR '.join(conditions)}) LIMIT {top_k*2}",
            params
        ).fetchall()
        conn.close()
        sql_results = [dict(r) for r in rows]
        print(f"[Case 3] SQL found: {len(sql_results)}")

    # ─ Step 2: RAG fallback if SQL insufficient ─
    SQL_THRESHOLD = 2
    if len(sql_results) < SQL_THRESHOLD:
        search_mode = "RAG" if not sql_results else "SQL+RAG"
        rag_raw = emp_vectorstore.similarity_search_with_score(query, k=top_k)
        for doc, score in rag_raw:
            rag_results.append({
                "emp_id": doc.metadata["emp_id"],
                "name": doc.metadata["name"],
                "similarity_score": round(1 / (1 + score), 3),
                "source": "RAG"
            })
    else:
        search_mode = "SQL"

    # ─ Step 3: Merge and deduplicate ─
    seen = set()
    merged = []
    for emp in sql_results[:top_k]:
        if emp["emp_id"] not in seen:
            emp["source"] = "SQL"
            emp["similarity_score"] = None
            merged.append(emp)
            seen.add(emp["emp_id"])

    conn = get_db_connection()
    for r in rag_results:
        if r["emp_id"] not in seen:
            full = conn.execute(
                "SELECT * FROM employees WHERE emp_id=?", (r["emp_id"],)
            ).fetchone()
            if full:
                emp = dict(full)
                emp["source"] = "RAG"
                emp["similarity_score"] = r["similarity_score"]
                merged.append(emp)
                seen.add(r["emp_id"])
    conn.close()

    print(f"[Case 3] Merged: {len(merged)} | Mode: {search_mode}")

    if not merged:
        return {
            "query": query, "search_mode": search_mode, "results": [],
            "analysis": "No matching bench employees found for this query."
        }

    # ─ Step 4: LLM generates answer ─
    context = "\n".join([
        f"- {e['name']} | {e['designation']} | Skills: {e['skills']} | "
        f"Location: {e['location']} | Exp: {e['experience']} yrs | Source: {e['source']}"
        for e in merged
    ])

    prompt = f"""You are an RMG assistant helping find talent from the bench pool.

USER QUERY: {query}

MATCHING EMPLOYEES:
{context}

Answer the user's query directly. Highlight top 2-3 best matches and why.
Keep it concise and professional (3-4 sentences max)."""

    analysis = llm.invoke(prompt).content

    return {
        "query": query,
        "search_mode": search_mode,
        "results": merged,
        "analysis": analysis
    }



def _extract_keywords(query: str) -> list:
    stop_words = {
        "find","give","show","get","me","for","the","a","an","in","with",
        "who","has","have","people","person","employee","employees","role",
        "from","and","or","of","on","bench","available","suitable","any","all",
        "is","are","can","do","skills","looking","need","want"
    }
    words = re.sub(r"[^\w\s]", "", query.lower()).split()
    return [w for w in words if w not in stop_words and len(w) > 2]

def _parse_job_reasons(response: str, matches: list) -> dict:
    reasons = {}
    in_analysis = False
    for line in response.split("\n"):
        line = line.strip()
        if "JOB_ANALYSIS:" in line:
            in_analysis = True
            continue
        if "SUMMARY:" in line:
            break
        if in_analysis and ":" in line:
            for m in matches:
                if m["job_id"] in line:
                    reasons[m["job_id"]] = line.split(":", 1)[-1].strip()
                    break
    return reasons

def _parse_emp_reasons(response: str, matches: list) -> dict:
    reasons = {}
    in_analysis = False
    for line in response.split("\n"):
        line = line.strip()
        if "EMP_ANALYSIS:" in line:
            in_analysis = True
            continue
        if "SUMMARY:" in line:
            break
        if in_analysis and ":" in line:
            for m in matches:
                if m["emp_id"] in line:
                    reasons[m["emp_id"]] = line.split(":", 1)[-1].strip()
                    break
    return reasons

def _extract_summary(response: str) -> str:
    if "SUMMARY:" in response:
        return response.split("SUMMARY:")[-1].strip()
    return response[-400:].strip()


# ─────────────────────────────────────────────
# AGENT TOOLS
# ─────────────────────────────────────────────

@tool
def find_jobs_tool(employee_name: str) -> str:
    """
    Find suitable jobs for a named bench employee.
    Use when query contains an employee name and
    asks to find jobs or placement options.
    Input: employee name (partial match works)
    """
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM employees WHERE LOWER(name) LIKE LOWER(?)",
        (f"%{employee_name}%",)
    ).fetchone()
    conn.close()

    if not row:
        return (
            f"No employee found matching '{employee_name}'. "
            "Try names like: Abhay, Priya, Rohan, Sneha, "
            "Vikram, Anita, Karan, Meera, Arjun, Divya"
        )

    emp = dict(row)
    results = job_vs_global.similarity_search_with_score(
        f"Skills: {emp['skills']} "
        f"Designation: {emp['designation']}",
        k=5
    )

    out  = f"Top job matches for {emp['name']}:\n"
    out += f"Skills: {emp['skills']}\n"
    out += f"Experience: {emp['experience']} years\n\n"

    for i, (doc, score) in enumerate(results, 1):
        sim = round(1 / (1 + score), 2)
        out += (
            f"{i}. {doc.metadata['title']} | "
            f"{doc.metadata['location']} | "
            f"Openings: {doc.metadata['openings']} | "
            f"Match Score: {sim}\n"
        )
    return out


@tool
def find_employees_tool(job_query: str) -> str:
    """
    Find bench employees suitable for a job opening.
    Use when query asks to fill a role, find candidates,
    or mentions a job ID like JOB001 or a job title.
    Input: job ID (e.g. JOB001) or job title keywords
    """
    conn = get_db_connection()

    # Try exact job_id first
    row = conn.execute(
        "SELECT * FROM jobs WHERE LOWER(job_id) = LOWER(?)",
        (job_query,)
    ).fetchone()

    # If not found try title match
    if not row:
        row = conn.execute(
            "SELECT * FROM jobs WHERE LOWER(title) LIKE LOWER(?)",
            (f"%{job_query}%",)
        ).fetchone()

    conn.close()

    if not row:
        return (
            f"No job found matching '{job_query}'. "
            "Try job IDs like JOB001 to JOB012 or "
            "titles like 'AI Engineer', 'Data Scientist', "
            "'Frontend Developer'"
        )

    job = dict(row)
    results = emp_vs_global.similarity_search_with_score(
        f"Required Skills: {job['required_skills']} "
        f"Role: {job['title']} "
        f"Experience: {job['experience_required']} years",
        k=5
    )

    out  = f"Top candidates for {job['title']}:\n"
    out += f"Required Skills: {job['required_skills']}\n"
    out += f"Location: {job['location']} | "
    out += f"Openings: {job['openings']}\n\n"

    for i, (doc, score) in enumerate(results, 1):
        sim = round(1 / (1 + score), 2)
        out += (
            f"{i}. {doc.metadata['name']} | "
            f"{doc.metadata['designation']} | "
            f"{doc.metadata['location']} | "
            f"Match Score: {sim}\n"
        )
    return out


@tool
def search_bench_tool(query: str) -> str:
    """
    General search across all bench employees.
    Use for broad queries about skills, location,
    experience, gender, or when no specific employee
    or job is mentioned.
    Input: natural language e.g. 'Python developers in Mumbai'
    """
    stop_words = {
        "find", "show", "get", "me", "for", "the",
        "a", "an", "in", "with", "who", "has", "have",
        "people", "employees", "employee", "skill", "skills",
        "looking", "need", "want", "give", "from", "and",
        "or", "any", "all", "bench", "available"
    }
    keywords = [
        w for w in query.lower().split()
        if w not in stop_words and len(w) > 2
    ]

    # SQL keyword search first
    sql_results = []
    if keywords:
        conn = get_db_connection()
        conditions = []
        params = []
        for kw in keywords:
            conditions.append(
                "(LOWER(skills) LIKE ? OR "
                "LOWER(designation) LIKE ? OR "
                "LOWER(location) LIKE ? OR "
                "LOWER(name) LIKE ? OR "
                "LOWER(gender) LIKE ?)"
            )
            params.extend([f"%{kw}%"] * 5)

        rows = conn.execute(
            f"SELECT * FROM employees "
            f"WHERE status = 'bench' "
            f"AND ({' OR '.join(conditions)}) "
            f"LIMIT 8",
            params
        ).fetchall()
        conn.close()
        sql_results = [dict(r) for r in rows]

    # RAG fallback if SQL finds nothing
    if not sql_results:
        rag_results = emp_vs_global.similarity_search(query, k=5)
        conn = get_db_connection()
        for doc in rag_results:
            row = conn.execute(
                "SELECT * FROM employees WHERE emp_id = ?",
                (doc.metadata["emp_id"],)
            ).fetchone()
            if row:
                d = dict(row)
                d["source"] = "RAG"
                sql_results.append(d)
        conn.close()

    if not sql_results:
        return (
            "No matching bench employees found. "
            "Try different keywords or skills."
        )

    out = f"Bench employees matching '{query}':\n\n"
    for emp in sql_results[:6]:
        out += (
            f"• {emp['name']} | {emp['designation']} | "
            f"{emp['location']} | "
            f"{emp['experience']} yrs | {emp['gender']}\n"
            f"  Skills: {emp['skills'][:80]}...\n\n"
        )
    return out


# ─────────────────────────────────────────────
# AGENT RUNNER — handles ALL 3 cases in one
# ─────────────────────────────────────────────

RMG_TOOLS = [
    find_jobs_tool,
    find_employees_tool,
    search_bench_tool
]

def run_rmg_agent(query: str, llm) -> dict:
    """
    Single agent that reads the query and decides
    which tool to call. No need to choose endpoint.

    Examples of automatic routing:
    "Find a job for Abhay"    → find_jobs_tool
    "Fill JOB001 opening"     → find_employees_tool
    "Python devs in Mumbai"   → search_bench_tool
    """
    prompt = PromptTemplate.from_template("""
You are an intelligent RMG (Resource Management Group)
assistant helping match bench employees to job openings.

You have access to these tools:
{tools}

Use this EXACT format every time:
Question: the input question
Thought: think about which tool fits best
Action: tool name (must be one of [{tool_names}])
Action Input: the input string for the tool
Observation: the tool result
Thought: do I have enough to give a final answer?
Final Answer: your complete helpful response

Important rules:
- If asked about a specific person → use find_jobs_tool
- If asked to fill a role or job ID → use find_employees_tool
- For general skill/location queries → use search_bench_tool
- Always give a helpful final answer summarizing the results

Question: {input}
{agent_scratchpad}
""")

    agent = create_react_agent(
        llm=llm,
        tools=RMG_TOOLS,
        prompt=prompt
    )

    executor = AgentExecutor(
        agent=agent,
        tools=RMG_TOOLS,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    result = executor.invoke({"input": query})

    # Extract which tools were used
    tools_used = []
    for step in result.get("intermediate_steps", []):
        t = step[0].tool
        if t not in tools_used:
            tools_used.append(t)

    return {
        "answer":     result["output"],
        "tools_used": tools_used,
        "query":      query
    }