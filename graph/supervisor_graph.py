"""
graph/supervisor_graph.py  —  the SUPERVISOR (ReAct) multi-agent architecture.

This matches the project specification:
  * a SUPERVISOR agent that reasons about the work and decides what to do next
    (a ReAct loop: Reason -> choose an agent -> Act -> Observe -> repeat),
  * which DELEGATES each sub-task to a specialised sub-agent
    (Research, Summarization, Analysis, Coding),
  * all wired together as a stateful LangGraph.

Flow:
    START -> plan -> supervisor --(routes to)--> research / analyze / code
                         ^                              |
                         |______________________________|   (workers return to supervisor)
    supervisor --(when done)--> synthesize -> END

The supervisor decides routing using the LLM (the "reasoning" of ReAct); the
chosen sub-agent is then invoked in code, which keeps the system reliable on
free-tier models while remaining faithful to the supervisor/ReAct pattern.
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

MAX_STEPS = 14   # safety bound so the ReAct loop can never run forever


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
    """Decompose the request into 3-5 tracked sub-tasks (write_todos pattern)."""
    query = _user_query(state)
    prompt = (
        "Break the following request into 3 to 5 clear, focused sub-tasks. "
        "Return ONE sub-task per line, no numbering, no bullets.\n\n"
        f"Request: {query}"
    )
    lines = message_text(invoke_llm(prompt).content).splitlines()
    tasks = []
    for ln in lines:
        c = re.sub(r"^\s*\d+[.)]\s*", "", ln).strip("-*• \t")
        if c and len(c) > 5 and not c.endswith(":"):
            tasks.append(c)
    tasks = tasks[:5] or [query]
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
- research : finds factual information from the web (use for fact-finding, gathering, exploring a topic)
- analyze  : interprets, compares, or evaluates information that has already been gathered
- code     : writes, implements, or explains code in a programming language

All current sub-tasks and their status:
{todos}

The NEXT pending sub-task to assign is:
"{task}"

Choose the ONE specialist that this specific sub-task needs. Do not pick an agent the task does not require. Respond EXACTLY in this format and nothing else:
AGENT: <research|analyze|code>"""


def _classify(task: str) -> str:
    """Keyword fallback so routing is reliable even if the model's reply is unclear."""
    t = task.lower()
    if any(k in t for k in ("write code", "implement", "function", "script", "program",
                            "python code", "code for", "coding", "debug", "snippet")):
        return "code"
    if any(k in t for k in ("analyze", "analyse", "compare", "evaluate", "assess",
                            "interpret", "implication", "trade-off", "tradeoff",
                            "pros and cons", "critically", "synthesi")):
        return "analyze"
    return "research"


def supervisor_node(state: AgentState) -> dict:
    """Assign the next pending sub-task to the specific specialist it needs."""
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
    todo_str = "\n".join(f"  - [{t['status']}] {t['content']}" for t in todos)
    prompt = SUPERVISOR_PROMPT.format(todos=todo_str, task=task)
    decision = message_text(invoke_llm(prompt, temperature=0).content)
    agent = (_parse(decision, "AGENT") or "").lower().strip()

    # robust fallback: classify by keywords if the model's choice is unclear
    if agent not in ("research", "analyze", "code"):
        agent = _classify(task)

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


# ====================== 4) SYNTHESIZE NODE (section-by-section) ======================
def synthesize_node(state: AgentState) -> dict:
    query = _user_query(state)
    notes = state.get("research_notes", "")
    if len(notes) > 7000:
        notes = notes[:7000] + "\n\n[Notes truncated for length.]"
    print("[SYNTHESIZE] writing final report section-by-section...")

    sections = [
        ("Overview",
         "Write ONLY the 'Overview' section (2-3 sentences). Do not write other sections."),
        ("Key Findings",
         "Write ONLY the 'Key Findings' section as 4-6 bullet points, each ending with a "
         "source in parentheses. Do not write other sections."),
        ("Analysis",
         "Write ONLY the 'Analysis' section as 3-4 substantial paragraphs comparing evidence, "
         "trade-offs, and implications. Do not write other sections."),
        ("Conclusion",
         "Write ONLY the 'Conclusion' section (2-3 sentences). Do not write other sections."),
    ]
    parts = []
    for title, instruction in sections:
        prompt = (
            f"You are writing a research report on: {query}\n\n"
            f"Research notes:\n{notes}\n\n{instruction}\n"
            "Do not repeat the section heading; write only the body text."
        )
        try:
            body = message_text(invoke_llm(prompt, temperature=0).content).strip()
        except Exception as e:
            body = f"(Section unavailable: {e})"
        parts.append(f"## {title}\n\n{body}")

    final_report = f"# Research Report: {query}\n\n" + "\n\n".join(parts)
    print(f"[SYNTHESIZE] done ({len(final_report)} chars)")
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