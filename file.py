"""
file.py — RMG Talent Matcher UI
Run: streamlit run file.py
"""

import streamlit as st
import time
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RMG Talent Matcher",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────
# LOAD RESOURCES ONCE
# ─────────────────────────────────────────────
@st.cache_resource
def load_all():
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from utils import set_vs_global

    # ── FIXED: use only .env, no st.secrets ──
    groq_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=groq_key
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    job_vs = FAISS.load_local(
        "vectorstore_jobs", embeddings,
        allow_dangerous_deserialization=True
    )
    emp_vs = FAISS.load_local(
        "vectorstore_employees", embeddings,
        allow_dangerous_deserialization=True
    )

    set_vs_global(job_vs, emp_vs)

    return llm, job_vs, emp_vs


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🤝 RMG Matcher")
    st.caption("AI-powered talent placement")
    st.divider()

    # ── FIXED: use index number instead of string matching ──
    mode_options = [
        "💬 Smart Chat",
        "👤 Employee → Jobs",
        "💼 Job → Employees",
        "📋 Browse All Jobs"
    ]

    mode_index = st.radio(
        "Choose Mode",
        options=range(len(mode_options)),
        format_func=lambda i: mode_options[i],
        index=0
    )

    # mode_index is now 0, 1, 2, or 3 — no string ambiguity
    selected_mode = mode_options[mode_index]

    st.divider()
    st.subheader("💡 Try These")

    if mode_index == 0:   # Smart Chat
        examples = [
            "Find a job for Abhay Kumar",
            "Fill the AI Engineer opening",
            "Python developers in Mumbai",
            "Who has Power BI skills?",
            "Find female engineers on bench",
            "We need a Data Scientist urgently",
        ]
    elif mode_index == 1:  # Employee → Jobs
        examples = [
            "Abhay Kumar", "Priya Sharma",
            "Rohan Mehta", "Sneha Iyer",
            "Vikram Singh", "Kavya Menon"
        ]
    elif mode_index == 2:  # Job → Employees
        examples = [
            "JOB001", "JOB002", "JOB003",
            "JOB004", "JOB005", "JOB009"
        ]
    else:                  # Browse All Jobs
        examples = [
            "Mumbai", "Bangalore", "Engineer", "Analyst"
        ]

    for ex in examples:
        if st.button(ex, use_container_width=True,
                     key=f"sidebar_{ex}"):
            st.session_state.sidebar_click = ex

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.rmg_messages = []
        st.success("Cleared!")

    st.divider()
    st.caption("📊 15 bench employees")
    st.caption("📋 12 job openings")
    st.caption("🤖 Groq Llama 3.3 (Free)")
    st.caption("🔍 HuggingFace Embeddings (Local)")


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
with st.spinner("Loading AI models..."):
    llm, job_vs, emp_vs = load_all()


# ─────────────────────────────────────────────
# MODE 0: SMART CHAT AGENT
# ─────────────────────────────────────────────
if mode_index == 0:
    st.title("💬 Smart RMG Assistant")
    st.caption(
        "Describe what you need — "
        "agent decides which search to run"
    )

    if "rmg_messages" not in st.session_state:
        st.session_state.rmg_messages = []

    # Welcome message
    if not st.session_state.rmg_messages:
        with st.chat_message("assistant"):
            st.markdown("""
👋 Hello! I'm your RMG AI Assistant.

Tell me what you need:
- **"Find a job for Abhay Kumar"** → searches jobs for that employee
- **"Fill the AI Engineer opening"** → finds candidates for that role
- **"Python developers in Mumbai"** → searches bench pool by skill

Try an example from the sidebar or type below!
            """)

    for msg in st.session_state.rmg_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools_used"):
                st.caption(
                    f"🔧 Tool: {' → '.join(msg['tools_used'])}"
                )
            if msg.get("time"):
                st.caption(f"⏱️ {msg['time']:.1f}s")

    sidebar_val = st.session_state.pop("sidebar_click", None)
    user_input = (
        st.chat_input("What talent do you need?")
        or sidebar_val
    )

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.rmg_messages.append({
            "role": "user", "content": user_input
        })

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                start = time.time()
                try:
                    from utils import run_rmg_agent
                    result = run_rmg_agent(user_input, llm)
                    elapsed = time.time() - start
                    answer = result["answer"]
                    tools = result.get("tools_used", [])

                    st.markdown(answer)
                    col1, col2 = st.columns(2)
                    with col1:
                        if tools:
                            st.caption(
                                f"🔧 Tool: {' → '.join(tools)}"
                            )
                    with col2:
                        st.caption(f"⏱️ {elapsed:.1f}s")

                    st.session_state.rmg_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "tools_used": tools,
                        "time": elapsed
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    if "rate" in str(e).lower():
                        st.warning(
                            "Rate limit hit. "
                            "Wait 30 seconds and try again."
                        )


# ─────────────────────────────────────────────
# MODE 1: EMPLOYEE → JOBS
# ─────────────────────────────────────────────
elif mode_index == 1:
    st.title("👤 Find Jobs for an Employee")
    st.caption(
        "Enter employee name — system fetches their "
        "skills and finds best matching job openings"
    )

    sidebar_val = st.session_state.pop("sidebar_click", None)

    emp_name = st.text_input(
        "Employee Name (partial name works)",
        value=sidebar_val or "",
        placeholder="e.g. Abhay, Priya, Kavya..."
    )
    top_k = st.slider("Number of job matches", 1, 10, 5)

    if st.button("🔍 Find Jobs", type="primary",
                 disabled=not emp_name):
        with st.spinner(f"Finding jobs for {emp_name}..."):
            from utils import find_jobs_for_employee
            result = find_jobs_for_employee(
                employee_name=emp_name,
                top_k=top_k,
                llm=llm,
                job_vectorstore=job_vs
            )

        if "error" in result:
            st.error(result["error"])
            st.info(
                "Available: Abhay Kumar, Priya Sharma, "
                "Rohan Mehta, Sneha Iyer, Vikram Singh, "
                "Anita Desai, Karan Patel, Meera Nair, "
                "Arjun Reddy, Divya Krishnan..."
            )
        else:
            st.success(
                f"✅ Found {len(result['matched_jobs'])} "
                f"matches for {result['employee_name']}"
            )

            with st.expander("👤 Employee Profile",
                             expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Name", result["employee_name"])
                    st.metric("Experience",
                              f"{result['experience']} years")
                with col2:
                    st.metric("Designation",
                              result.get("designation", "—"))
                st.write("**Skills:**",
                         result["employee_skills"])

            st.info(f"💡 {result['summary']}")
            st.subheader("📋 Matched Jobs")

            for i, job in enumerate(
                result["matched_jobs"], 1
            ):
                score = job.get("similarity_score", 0)
                flag = ("🟢" if score >= 0.7
                        else "🟡" if score >= 0.5
                        else "🔴")
                with st.expander(
                    f"{flag} {i}. {job['title']} — "
                    f"{job['location']} "
                    f"(Score: {score})",
                    expanded=(i == 1)
                ):
                    st.write(job.get(
                        "match_reason",
                        "Strong skill alignment"
                    ))
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Job ID", job["job_id"])
                    with col2:
                        st.metric("Openings", job["openings"])

            df = pd.DataFrame([{
                "Job ID": j["job_id"],
                "Title": j["title"],
                "Location": j["location"],
                "Openings": j["openings"],
                "Score": j["similarity_score"]
            } for j in result["matched_jobs"]])

            st.download_button(
                "⬇️ Download Results CSV",
                df.to_csv(index=False),
                f"{result['employee_name']}_jobs.csv",
                "text/csv"
            )


# ─────────────────────────────────────────────
# MODE 2: JOB → EMPLOYEES
# ─────────────────────────────────────────────
elif mode_index == 2:
    st.title("💼 Find Employees for a Job")
    st.caption(
        "Select job opening — system finds best "
        "matching bench employees"
    )

    sidebar_val = st.session_state.pop("sidebar_click", None)

    job_options = {
        "JOB001 — AI Engineer (Mumbai)": "JOB001",
        "JOB002 — Data Scientist (Bangalore)": "JOB002",
        "JOB003 — Frontend Developer (Chennai)": "JOB003",
        "JOB004 — DevOps Engineer (Bangalore)": "JOB004",
        "JOB005 — Business Analyst (Mumbai)": "JOB005",
        "JOB006 — Backend Developer (Pune)": "JOB006",
        "JOB007 — Data Engineer (Bangalore)": "JOB007",
        "JOB008 — Senior Software Engineer (Hyd)": "JOB008",
        "JOB009 — ML Engineer (Mumbai)": "JOB009",
        "JOB010 — Security Analyst (Delhi)": "JOB010",
        "JOB011 — Project Manager (Mumbai)": "JOB011",
        "JOB012 — Prompt Engineer (Hyderabad)": "JOB012",
    }

    default_idx = 0
    if sidebar_val:
        for i, key in enumerate(job_options.keys()):
            if sidebar_val in key:
                default_idx = i
                break

    selected = st.selectbox(
        "Select Job Opening",
        list(job_options.keys()),
        index=default_idx
    )
    job_id = job_options[selected]
    top_k = st.slider("Number of candidates", 1, 10, 5)

    if st.button("🔍 Find Candidates", type="primary"):
        with st.spinner(f"Searching bench..."):
            from utils import find_employees_for_job
            result = find_employees_for_job(
                job_id=job_id,
                top_k=top_k,
                llm=llm,
                emp_vectorstore=emp_vs
            )

        if "error" in result:
            st.error(result["error"])
        else:
            st.success(
                f"✅ Found "
                f"{len(result['matched_employees'])} "
                f"candidates for {result['job_title']}"
            )

            with st.expander("📋 Job Details",
                             expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Job ID", result["job_id"])
                    st.metric("Openings", result["openings"])
                with col2:
                    st.metric("Title", result["job_title"])
                st.write("**Required Skills:**",
                         result["required_skills"])

            st.info(f"💡 {result['summary']}")
            st.subheader("👥 Matched Candidates")

            df = pd.DataFrame([{
                "Name": e["name"],
                "Role": e["designation"],
                "Location": e["location"],
                "Score": e["similarity_score"],
                "Skills": e.get("skills", "")[:60] + "..."
            } for e in result["matched_employees"]])

            st.dataframe(
                df.sort_values("Score", ascending=False),
                use_container_width=True
            )

            for i, emp in enumerate(
                result["matched_employees"], 1
            ):
                with st.expander(
                    f"{i}. {emp['name']} — "
                    f"Score: {emp['similarity_score']}"
                ):
                    st.write(emp.get(
                        "match_reason",
                        "Strong skill alignment"
                    ))
                    st.write("**Skills:**",
                             emp.get("skills", ""))

            st.download_button(
                "⬇️ Download Candidates CSV",
                df.to_csv(index=False),
                f"{job_id}_candidates.csv",
                "text/csv"
            )


# ─────────────────────────────────────────────
# MODE 3: BROWSE ALL JOBS
# ─────────────────────────────────────────────
elif mode_index == 3:
    st.title("📋 All Available Job Openings")
    st.caption("Filter and browse all active positions")

    sidebar_val = st.session_state.pop("sidebar_click", None)

    col1, col2 = st.columns(2)
    with col1:
        loc_filter = st.selectbox(
            "Filter by Location",
            ["All", "Mumbai", "Bangalore", "Hyderabad",
             "Chennai", "Pune", "Delhi", "Ahmedabad"]
        )
    with col2:
        title_filter = st.text_input(
            "Search by Title",
            value=sidebar_val or "",
            placeholder="e.g. Engineer, Analyst..."
        )

    from utils import get_all_jobs
    result = get_all_jobs(
        location=loc_filter if loc_filter != "All" else None,
        title=title_filter if title_filter else None
    )

    st.metric("Total Openings", result["total_jobs"])

    if result["jobs"]:
        df = pd.DataFrame([{
            "Job ID": j["job_id"],
            "Title": j["title"],
            "Department": j["department"],
            "Location": j["location"],
            "Openings": j["openings"],
            "Min Exp": f"{j['experience_required']}+ yrs",
            "Required Skills": j["required_skills"][:50]+"..."
        } for j in result["jobs"]])

        st.dataframe(df, use_container_width=True,
                     height=400)

        st.download_button(
            "⬇️ Download Job List CSV",
            df.to_csv(index=False),
            "available_jobs.csv",
            "text/csv"
        )

        st.subheader("Job Details")
        for job in result["jobs"]:
            with st.expander(
                f"📌 {job['job_id']} — {job['title']} | "
                f"{job['location']} | "
                f"{job['openings']} opening(s)"
            ):
                st.write("**Required Skills:**",
                         job["required_skills"])
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Department",
                              job["department"])
                with col2:
                    st.metric(
                        "Min Experience",
                        f"{job['experience_required']}+ yrs"
                    )
    else:
        st.info("No jobs match the selected filters.")