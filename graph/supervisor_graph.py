"""
graph/supervisor_graph.py  —  the SUPERVISOR (ReAct) multi-agent architecture.

A SUPERVISOR agent reasons about the work and delegates each sub-task to the
specialist sub-agent it needs (Research, Summarization, Analysis, Coding),
looping until done, then synthesizes a report. Built as a stateful LangGraph.

Flow:
    START -> plan -> supervisor --(routes to)--> research / analyze / code
                         ^                              |
                         |______________________________|   (workers return)
    supervisor --(when done)--> synthesize -> END
"""

import re

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from state.agent_state import AgentState
from services.llm_service import invoke_llm
from utils.helpers import message_text
from agents.specialists import (
    run_research_agent,
    run_summarization_agent,
    run_analysis_agent,
    run_coding_agent,
)

MAX_STEPS = 12          # safety bound so the ReAct loop can never run forever
MAX_TASKS = 3           # fewer sub-tasks = fewer model calls = faster runs
MAX_SECTIONS = 4        # cap on synthesized report sections


def _user_query(state: AgentState) -> str:
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            return message_text(m.content)
    return message_text(state["messages"][0].content)


def _parse(text: str, key: str):
    m = re.search(key + r"\s*:\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _mark_first_pending_done(todos, instruction):
    """Mark the matching (or first) pending todo as completed."""
    for t in todos:
        if t["status"] != "completed" and (t["content"] == instruction):
            t["status"] = "completed"
            return t["content"]
    for t in todos:                      # fallback: first pending
        if t["status"] == "pending":
            t["status"] = "completed"
            return t["content"]
    return None


# ====================== 1) PLAN NODE ======================
def plan_node(state: AgentState) -> dict:
    """Decompose the request into 3 tracked sub-tasks (write_todos pattern)."""
    query = _user_query(state)
    prompt = (
        f"Break the following request into exactly 3 clear, focused sub-tasks. "
        "Return ONE sub-task per line, no numbering, no bullets.\n\n"
        f"Request: {query}"
    )
    lines = message_text(invoke_llm(prompt).content).splitlines()
    tasks = []
    for ln in lines:
        c = re.sub(r"^\s*\d+[.)]\s*", "", ln).strip("-*• \t")
        if c and len(c) > 5 and not c.endswith(":"):
            tasks.append(c)
    tasks = tasks[:MAX_TASKS] or [query]
    todos = [{"content": t, "status": "pending"} for t in tasks]
    plan_text = "Plan:\n" + "\n".join(f"  - {t}" for t in tasks)
    print(f"[PLAN] {len(tasks)} sub-tasks created")
    return {
        "todos": todos,
        "messages": [AIMessage(content=plan_text)],
        "research_notes": "",
        "virtual_files": {},
        "completed_tasks": [],
        "next_agent": "",
        "task_instruction": "",
        "step_count": 0,
    }


# ====================== 2) SUPERVISOR NODE (ReAct reasoning) ======================
SUPERVISOR_PROMPT = """You are the SUPERVISOR. You do NOT do the work yourself. You assign the NEXT pending sub-task to the single most appropriate specialist sub-agent.

Specialists:
- research : finds factual information from the web (fact-finding, gathering, exploring a topic)
- analyze  : interprets, compares, or evaluates information that has already been gathered
- code     : writes, implements, or explains code in a programming language

All current sub-tasks and their status:
{todos}

The NEXT pending sub-task to assign is:
"{task}"

Choose the ONE specialist that this specific sub-task needs. Respond EXACTLY in this format and nothing else:
AGENT: <research|analyze|code>"""


def _classify(task: str) -> str:
    """Fast keyword router — no API call. Used first; the LLM is only a backup."""
    t = task.lower()
    if any(k in t for k in ("write code", "write a function", "write a python",
                            "python function", "python script", "code for",
                            "implement a function", "implement an algorithm",
                            "write a program", "coding", "debug", "code snippet")):
        return "code"
    if any(k in t for k in ("analyze", "analyse", "compare", "evaluate", "assess",
                            "interpret", "implication", "trade-off", "tradeoff",
                            "pros and cons", "critically", "synthesi")):
        return "analyze"
    return "research"


def supervisor_node(state: AgentState) -> dict:
    """Assign the next pending sub-task to the specialist it needs.

    Hybrid routing: the free keyword classifier decides first; the LLM is only
    consulted when the keyword guess is the default ('research'), so a plainly
    worded analyze/code task isn't missed. This removes most supervisor LLM
    calls and makes runs much faster while keeping LLM-reasoned routing.
    """
    step = state.get("step_count", 0) + 1
    todos = [dict(t) for t in state.get("todos", [])]
    pending = [t for t in todos if t["status"] == "pending"]

    # safety: bound the loop
    if step > MAX_STEPS:
        print("[SUPERVISOR] step limit reached -> synthesize")
        return {"next_agent": "synthesize", "task_instruction": "", "step_count": step}

    # all tasks handled -> write the report
    if not pending:
        print(f"[SUPERVISOR] step {step}: all sub-tasks complete -> synthesize")
        return {"next_agent": "synthesize", "task_instruction": "", "step_count": step}

    task = pending[0]["content"]
    agent = _classify(task)                         # fast, no API call

    # only ask the LLM when the keyword guess is the generic default
    if agent == "research":
        todo_str = "\n".join(f"  - [{t['status']}] {t['content']}" for t in todos)
        prompt = SUPERVISOR_PROMPT.format(todos=todo_str, task=task)
        try:
            decision = message_text(invoke_llm(prompt, temperature=0).content)
            llm_agent = (_parse(decision, "AGENT") or "").lower().strip()
            if llm_agent in ("research", "analyze", "code"):
                agent = llm_agent
        except Exception:
            pass                                    # keep the keyword guess on error

    print(f"[SUPERVISOR] step {step}: '{task[:45]}' -> {agent}")
    return {"next_agent": agent, "task_instruction": task, "step_count": step}


def route_from_supervisor(state: AgentState) -> str:
    nxt = state.get("next_agent", "synthesize")
    return nxt if nxt in ("research", "analyze", "code", "synthesize") else "synthesize"


# ====================== 3) WORKER NODES (the sub-agents) ======================
def research_node(state: AgentState) -> dict:
    """Research Agent + Summarization Agent: gather, then condense, then offload."""
    todos = [dict(t) for t in state["todos"]]
    instr = state.get("task_instruction") or next(
        (t["content"] for t in todos if t["status"] == "pending"), ""
    )
    print(f"[RESEARCH] {instr}")
    raw = run_research_agent(instr)                 # sub-agent 1: research (Tavily)
    condensed = run_summarization_agent(raw)        # sub-agent 2: summarize
    files = dict(state.get("virtual_files", {}))
    idx = len([k for k in files if k.startswith("findings_")]) + 1
    files[f"findings_{idx}.md"] = f"# {instr}\n\n{condensed}"
    notes = (state.get("research_notes", "") + f"\n\n## {instr}\n{condensed}").strip()
    done = _mark_first_pending_done(todos, instr)
    completed = list(state.get("completed_tasks", [])) + ([done] if done else [])
    print(f"[VFS] saved findings_{idx}.md   [SUMMARIZE] {len(condensed)} chars")
    return {
        "todos": todos,
        "virtual_files": files,
        "research_notes": notes,
        "completed_tasks": completed,
        "messages": [AIMessage(content=f"[research+summarize] {instr}")],
    }


def analyze_node(state: AgentState) -> dict:
    """Analysis Agent: interpret/compare the notes for the assigned task."""
    todos = [dict(t) for t in state["todos"]]
    instr = state.get("task_instruction") or "Analyse the gathered notes."
    notes = state.get("research_notes", "")
    print(f"[ANALYZE] {instr}")
    analysis = run_analysis_agent(notes)
    files = dict(state.get("virtual_files", {}))
    idx = len([k for k in files if k.startswith("analysis_")]) + 1
    files[f"analysis_{idx}.md"] = f"# {instr}\n\n{analysis}"
    notes2 = (notes + f"\n\n## Analysis: {instr}\n{analysis}").strip()
    done = _mark_first_pending_done(todos, instr)
    completed = list(state.get("completed_tasks", [])) + ([done] if done else [])
    return {
        "todos": todos,
        "virtual_files": files,
        "research_notes": notes2,
        "completed_tasks": completed,
        "messages": [AIMessage(content=f"[analyze] {instr}")],
    }


def code_node(state: AgentState) -> dict:
    """Coding Agent: write/explain code for a task that needs it."""
    instr = state.get("task_instruction", "")
    print(f"[CODE] {instr}")
    code = run_coding_agent(instr)
    todos = [dict(t) for t in state["todos"]]
    files = dict(state.get("virtual_files", {}))
    idx = len([k for k in files if k.startswith("code_")]) + 1
    files[f"code_{idx}.md"] = f"# {instr}\n\n{code}"
    notes = (state.get("research_notes", "") + f"\n\n## Code: {instr}\n{code}").strip()
    done = _mark_first_pending_done(todos, instr)
    completed = list(state.get("completed_tasks", [])) + ([done] if done else [])
    return {
        "todos": todos,
        "virtual_files": files,
        "research_notes": notes,
        "completed_tasks": completed,
        "messages": [AIMessage(content=f"[code] {instr}")],
    }


# ====================== 4) SYNTHESIZE NODE (dynamic, section-by-section) ======================
def synthesize_node(state: AgentState) -> dict:
    query = _user_query(state)
    notes = state.get("research_notes", "")
    if len(notes) > 7000:
        notes = notes[:7000] + "\n\n[Notes truncated for length.]"

    # 1) Let the model choose the sections that actually fit THIS query.
    outline_prompt = (
        f"You are planning the structure of a research report on: {query}\n\n"
        f"Based on the research notes below, choose the {MAX_SECTIONS} section titles "
        "that best fit THIS specific topic. Always start with 'Overview' and end with "
        "'Conclusion'. Choose middle sections that genuinely suit the query (e.g. "
        "'Key Findings', 'Comparison', 'Benefits', 'Risks & Challenges', 'Applications', "
        "'Implementation', 'Future Outlook') — only include ones the notes support.\n\n"
        "Return ONLY the section titles, one per line, no numbering or extra text.\n\n"
        f"Notes:\n{notes[:3000]}"
    )
    try:
        raw = message_text(invoke_llm(outline_prompt, temperature=0).content)
        titles = [ln.strip("-*•0123456789. \t") for ln in raw.splitlines() if ln.strip()]
        titles = [t for t in titles if len(t) > 2][:MAX_SECTIONS]
    except Exception:
        titles = []
    if len(titles) < 2:
        titles = ["Overview", "Key Findings", "Analysis", "Conclusion"]
    if titles[0].lower() != "overview":
        titles = ["Overview"] + titles
    if titles[-1].lower() != "conclusion":
        titles = titles + ["Conclusion"]
    titles = titles[:MAX_SECTIONS + 1]              # hard cap

    print(f"[SYNTHESIZE] sections chosen: {', '.join(titles)}")

    # 2) Write each chosen section in its own call (keeps anti-truncation benefit).
    parts = []
    for title in titles:
        prompt = (
            f"You are writing a research report on: {query}\n\n"
            f"Research notes:\n{notes}\n\n"
            f"Write ONLY the '{title}' section. Use the notes and cite sources where "
            f"relevant. If the section calls for points, use 4-6 bullets; if it calls "
            f"for discussion, use 2-4 paragraphs. Write only the body text — do not "
            f"repeat the heading and do not write any other section."
        )
        try:
            body = message_text(invoke_llm(prompt, temperature=0).content).strip()
        except Exception as e:
            body = f"(Section unavailable: {e})"
        parts.append(f"## {title}\n\n{body}")

    final_report = f"# Research Report: {query}\n\n" + "\n\n".join(parts)
    print(f"[SYNTHESIZE] done ({len(final_report)} chars, {len(titles)} sections)")
    return {
        "final_report": final_report,
        "messages": [AIMessage(content="Final report synthesized.")],
    }


# ====================== BUILD THE GRAPH ======================
def build_supervisor_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("research", research_node)
    g.add_node("analyze", analyze_node)
    g.add_node("code", code_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"research": "research", "analyze": "analyze", "code": "code", "synthesize": "synthesize"},
    )
    g.add_edge("research", "supervisor")   # workers return to the supervisor (the loop)
    g.add_edge("analyze", "supervisor")
    g.add_edge("code", "supervisor")
    g.add_edge("synthesize", END)
    return g.compile()