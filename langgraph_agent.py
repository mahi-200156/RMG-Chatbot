"""
langgraph_agent.py   (RMG Job Matching Chatbot)
-----------------------------------------------
LangGraph version of the RMG agent. It uses the SAME 3 tools as the original
ReAct agent in utils.py (find_jobs_tool, find_employees_tool, search_bench_tool).

Flow:
    START -> classify_intent --(small talk)--> general_answer -> END
                             --(search)------> analyst --(final answer)--> END
                                                 |
                                        (needs a tool)
                                                 v
                                               tools -> check_tools --(ok / retries left)--> analyst
                                                            |
                                                            +--(too many failures)--> give_up -> END
                                                            +--(tool limit reached)--> wrap_up -> END

Used by utils.run_rmg_agent() when the environment variable AGENT_ENGINE=langgraph.
"""

import re
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from utils import RMG_TOOLS   # the same 3 tools the original agent uses

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
SYSTEM = SystemMessage(
    content="""You are an intelligent RMG (Resource Management Group) assistant that matches bench employees to job openings.

Tools:
- find_jobs_tool: use when the user names an employee and wants jobs or placement options.
- find_employees_tool: use when the user wants to fill a role, or gives a job ID (like JOB001) or a job title.
- search_bench_tool: use for broad searches by skill, location, experience or gender.

Rules:
- Always use a tool to get employees or jobs. Never invent names, skills, scores, locations or job IDs.
- You can only search and explain matches. You cannot post, edit or delete jobs or employee records, so never offer to.
- Match Score is a similarity score between 0 and 1 from semantic search. It is not a percentage and not a guarantee of fit.
- Reasons why someone fits must come from the skills and details in the tool results. Present any further reasoning as a possibility ('may', 'could').
- If a tool says nothing was found, say so plainly. Do not make up alternatives.
- If a tool note says no one matched all keywords, tell the user the results are partial matches.
- Give a short, helpful final answer that lists the best matches. Use plain Markdown. Do not use HTML tags such as <br>."""
)

GENERAL_SYSTEM = SystemMessage(
    content=(
        "You are a friendly RMG (Resource Management Group) assistant. "
        "Answer greetings and general questions briefly. "
        "You can only search and match: you cannot post, edit or delete jobs or employee records. "
        "If the user wants talent or jobs, tell them they can ask things like: "
        "'Find a job for Abhay Kumar', 'Fill the AI Engineer opening' or "
        "'Python developers in Mumbai'."
    )
)

MAX_RETRIES = 2        # retries after a failed tool call (3 tries in total)
MAX_TOOL_ROUNDS = 5    # max tool batches per question (the old agent allowed 5 iterations)
HISTORY_TURNS = 3      # past questions sent to the LLM
RECURSION_LIMIT = 40   # last-resort safety stop

# A tool result starting with one of these counts as a FAILED call and is retried.
# ("No employee found" / "No job found" are normal answers, not failures.)
ERROR_PREFIXES = ("error", "database error", "invalid")

# Words that clearly mean "this is a search / matching question"
DATA_KEYWORDS = [
    "job", "employee", "bench", "skill", "candidate", "fill", "opening", "role",
    "hire", "hiring", "position", "vacancy", "developer", "engineer", "analyst",
    "scientist", "manager", "python", "java", "sql", "power bi", "react", "aws",
    "mumbai", "bangalore", "hyderabad", "chennai", "pune", "delhi", "ahmedabad",
    "experience", "female", "male", "gender", "match", "placement", "resource",
]
JOB_ID = re.compile(r"\bjob\d+\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class RMGState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history
    intent: str                              # "search" or "general"
    error: Optional[str]                     # last tool error text
    retries: int                             # failed-tool retries used this turn
    rounds: int                              # tool batches used this turn


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


def _current_turn(messages: list) -> list:
    """Messages after the latest user question."""
    last_human = -1
    for i, m in enumerate(messages):
        if m.type == "human":
            last_human = i
    return messages[last_human + 1:]


def _recent(messages: list, turns: int = HISTORY_TURNS) -> list:
    """Keep only the last few user turns (always starts at a user message)."""
    human_idx = [i for i, m in enumerate(messages) if m.type == "human"]
    if len(human_idx) <= turns:
        return messages
    return messages[human_idx[-turns]:]


def _is_tool_error(msg) -> bool:
    if msg.type != "tool":
        return False
    if getattr(msg, "status", None) == "error":
        return True
    return _text(msg.content).strip().lower().startswith(ERROR_PREFIXES)


# ---------------------------------------------------------------------------
# GRAPH
# ---------------------------------------------------------------------------
_checkpointer = MemorySaver()   # built-in conversation memory (one thread = one session)
_graph = None


def build_graph(llm):
    llm_with_tools = llm.bind_tools(RMG_TOOLS)

    # node 1: search question or small talk?
    def classify_intent(state: RMGState):
        question = _text(state["messages"][-1].content).lower()

        if JOB_ID.search(question) or any(k in question for k in DATA_KEYWORDS):
            intent = "search"
        else:
            try:
                recent = _recent(state["messages"], 2)
                transcript = "\n".join(
                    f"{m.type}: {_text(m.content)[:300]}"
                    for m in recent
                    if m.type in ("human", "ai") and _text(m.content).strip()
                )
                reply = llm.invoke(
                    "Classify the LAST user message. Reply with exactly one word.\n"
                    "search = asks about employees, jobs, skills, candidates, a person's name, "
                    "or is a follow-up to such a question.\n"
                    "general = greeting, thanks, or a question about what you can do.\n\n"
                    f"{transcript}\n\nAnswer:"
                )
                label = _text(reply.content).strip().lower()
                intent = "general" if label.startswith("general") else "search"
            except Exception:
                intent = "search"   # LLM unavailable: the analyst node reports the error

        return {"intent": intent, "retries": 0, "rounds": 0, "error": None}

    # node 2: the LLM thinks and may call tools
    def analyst(state: RMGState):
        prompt = [SYSTEM]
        if state.get("error"):
            prompt.append(
                SystemMessage(
                    content=(
                        f"Your previous tool call failed with: {state['error']}\n"
                        "Fix the arguments and call the tool again."
                    )
                )
            )
        last_err = None
        for _ in range(2):  # Groq sometimes fails on a tool call; try once more
            try:
                response = llm_with_tools.invoke(prompt + _recent(state["messages"]))
                return {"messages": [response]}
            except Exception as e:
                last_err = e
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Sorry, the language model failed to respond "
                        f"({str(last_err)[:150]}). Please try again."
                    )
                )
            ]
        }

    # node 3: runs the tools; handle_tool_errors=True turns a crash into an error message
    tool_node = ToolNode(RMG_TOOLS, handle_tool_errors=True)

    # node 4: look at the tool results, count failures and rounds
    def check_tools(state: RMGState):
        batch = []
        for m in reversed(state["messages"]):
            if m.type == "tool":
                batch.append(m)
            else:
                break
        errors = [m for m in batch if _is_tool_error(m)]
        update = {"rounds": state.get("rounds", 0) + 1, "error": None}
        if errors:
            update["error"] = _text(errors[-1].content)[:500]
            update["retries"] = state.get("retries", 0) + 1
        return update

    # node 5: tools failed too many times, stop politely
    def give_up(state: RMGState):
        first_line = str(state.get("error") or "unknown error").splitlines()[0]
        msg = (
            f"Sorry, I could not complete that search after {MAX_RETRIES + 1} attempts. "
            f"Last error: {first_line} "
            "Please rephrase the question or try again in a moment."
        )
        return {"messages": [AIMessage(content=msg)], "error": None}

    # node 5b: tool limit reached -> answer from the data already gathered
    def wrap_up(state: RMGState):
        turn = _current_turn(state["messages"])
        question = ""
        for m in reversed(state["messages"]):
            if m.type == "human":
                question = _text(m.content)
                break
        chunks = [
            _text(m.content)[:1500]
            for m in turn
            if m.type == "tool" and not _is_tool_error(m)
        ]
        data = "\n\n".join(chunks)[:7000]
        fallback = AIMessage(
            content=(
                "Sorry, I needed too many steps to answer that. "
                "Please ask a more specific question."
            )
        )
        if not data.strip():
            return {"messages": [fallback]}
        try:
            reply = llm.invoke(
                [
                    SYSTEM,
                    HumanMessage(
                        content=(
                            f"Question: {question}\n\n"
                            f"Tool results gathered so far:\n{data}\n\n"
                            "The tool limit has been reached. Answer the question now "
                            "using ONLY the results above. Do not invent names, skills or scores."
                        )
                    ),
                ]
            )
            text = _text(reply.content).strip()
            return {"messages": [AIMessage(content=text)] if text else [fallback]}
        except Exception:
            return {"messages": [fallback]}

    # node 6: small talk, no tools
    def general_answer(state: RMGState):
        plain = [
            m
            for m in _recent(state["messages"])
            if m.type == "human"
            or (m.type == "ai" and not getattr(m, "tool_calls", None) and _text(m.content).strip())
        ]
        try:
            return {"messages": [llm.invoke([GENERAL_SYSTEM] + plain)]}
        except Exception as e:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Sorry, the language model failed to respond "
                            f"({str(e)[:150]}). Please try again."
                        )
                    )
                ]
            }

    # routers
    def route_after_classify(state: RMGState):
        return "analyst" if state["intent"] == "search" else "general_answer"

    def route_after_analyst(state: RMGState):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    def route_after_check(state: RMGState):
        if state.get("error") and state.get("retries", 0) > MAX_RETRIES:
            return "give_up"
        if state.get("rounds", 0) >= MAX_TOOL_ROUNDS:
            return "wrap_up"
        return "analyst"

    builder = StateGraph(RMGState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("analyst", analyst)
    builder.add_node("tools", tool_node)
    builder.add_node("check_tools", check_tools)
    builder.add_node("give_up", give_up)
    builder.add_node("wrap_up", wrap_up)
    builder.add_node("general_answer", general_answer)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent", route_after_classify,
        {"analyst": "analyst", "general_answer": "general_answer"},
    )
    builder.add_conditional_edges(
        "analyst", route_after_analyst, {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "check_tools")
    builder.add_conditional_edges(
        "check_tools", route_after_check,
        {"analyst": "analyst", "give_up": "give_up", "wrap_up": "wrap_up"},
    )
    builder.add_edge("give_up", END)
    builder.add_edge("wrap_up", END)
    builder.add_edge("general_answer", END)

    return builder.compile(checkpointer=_checkpointer)


def get_graph(llm):
    global _graph
    if _graph is None:
        _graph = build_graph(llm)
    return _graph


def clear_langgraph_history(session_id: str):
    """Forget one chat session (used by utils.clear_rmg_history)."""
    try:
        _checkpointer.delete_thread(session_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SAME OUTPUT SHAPE AS utils.run_rmg_agent()
# ---------------------------------------------------------------------------
def run_rmg_langgraph_agent(query: str, llm, session_id: str = "default") -> dict:
    graph = get_graph(llm)
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    try:
        result = graph.invoke({"messages": [HumanMessage(content=query)]}, config)
    except GraphRecursionError:
        clear_langgraph_history(session_id)   # avoid a half-finished turn in memory
        return {
            "answer": "Sorry, I got stuck on that question. Please try rephrasing it.",
            "tools_used": [],
            "query": query,
            "session_id": session_id,
        }

    turn = _current_turn(result["messages"])

    tools_used = []
    for m in turn:
        if m.type == "ai":
            for call in getattr(m, "tool_calls", None) or []:
                if call["name"] not in tools_used:
                    tools_used.append(call["name"])

    last_ai = next((m for m in reversed(turn) if isinstance(m, AIMessage)), None)
    answer = _text(last_ai.content).strip() if last_ai else ""
    if not answer:
        answer = "I could not generate an answer. Please try again."

    return {
        "answer": answer,
        "tools_used": tools_used,
        "query": query,
        "session_id": session_id,
    }
