"""
Graph nodes for the Autonomous Cognitive Engine.

Each node receives the current state and returns the fields it wants to update.
The graph (build_graph.py) wires them in a fixed order, which is what enforces
correct sequencing — the LLM no longer chooses the flow, so it can't read files
before they're written or report before it researches.

Web search runs here in CODE, not via an LLM tool call, so the model never has
to format a tool call (the source of the malformed-call errors we saw).

Flow: plan_node -> research_node (looped once per task) -> synthesize_node
"""

import re

from langchain_core.messages import AIMessage, HumanMessage

from agents.summarization_agent import build_summarization_agent
from services.llm_service import get_llm
from state.agent_state import AgentState, Todo
from tools.search_tools import web_search
from utils.helpers import message_text


def _user_query(state: AgentState) -> str:
    """Return the original user request (the first human message)."""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            return message_text(m.content)
    return message_text(state["messages"][0].content)


def plan_node(state: AgentState) -> dict:
    """Break the user's request into a list of research sub-tasks."""
    query = _user_query(state)
    llm = get_llm()

    prompt = (
        "Break the following research request into 3 to 5 clear, focused "
        "sub-tasks. Return ONE sub-task per line. No numbering, no bullets, "
        "no extra text.\n\n"
        f"Request: {query}"
    )
    raw_lines = message_text(llm.invoke(prompt).content).splitlines()

    tasks: list[str] = []
    for line in raw_lines:
        cleaned = re.sub(r"^\s*\d+[.)]\s*", "", line)  # drop "1." or "2)"
        cleaned = cleaned.strip("-*• \t")
        if cleaned and not cleaned.endswith(":") and len(cleaned) > 5:
            tasks.append(cleaned)
    tasks = tasks[:5] or [query]  # fall back to the raw request if parsing fails

    todos: list[Todo] = [{"content": t, "status": "pending"} for t in tasks]
    plan_text = "Plan:\n" + "\n".join(f"  - {t}" for t in tasks)

    return {
        "todos": todos,
        "messages": [AIMessage(content=plan_text)],
        "research_notes": "",
        "virtual_files": {},
        "completed_tasks": [],
    }


def research_node(state: AgentState) -> dict:
    """Research the next pending task: search the web, then summarize."""
    todos = [dict(t) for t in state["todos"]]  # copy so we can edit statuses

    index = next(
        (i for i, t in enumerate(todos) if t["status"] == "pending"), None
    )
    if index is None:
        return {}  # nothing pending (the router normally prevents this)

    task = todos[index]["content"]
    todos[index]["status"] = "running"

    # 1) Search the web in CODE — deterministic, no LLM tool-call to mangle.
    try:
        raw = web_search.invoke({"query": task})
        results = raw.get("results", []) if isinstance(raw, dict) else []
        sources = "\n\n".join(
            f"Source: {r.get('url', '')}\n{r.get('content', '')}"
            for r in results
        ) or "No results found."
    except Exception as exc:
        sources = f"Web search failed: {exc}"

    # 2) Delegate condensing to the summarization specialist (no tools = reliable).
    summarizer = build_summarization_agent()
    summary_input = (
        f"Summarize the findings for this research task: '{task}'. "
        f"Keep the key facts and the source URLs.\n\nFindings:\n{sources}"
    )
    out = summarizer.invoke(
        {"messages": [{"role": "user", "content": summary_input}]}
    )
    findings = message_text(out["messages"][-1].content)

    # 3) Offload findings to the virtual file system and the running notes.
    files = dict(state.get("virtual_files", {}))
    files[f"findings_{index + 1}.md"] = f"# {task}\n\n{findings}"
    notes = f"{state.get('research_notes', '')}\n\n## {task}\n{findings}".strip()

    # 4) Mark the task completed.
    todos[index]["status"] = "completed"
    completed = list(state.get("completed_tasks", [])) + [task]

    return {
        "todos": todos,
        "virtual_files": files,
        "research_notes": notes,
        "completed_tasks": completed,
        "current_task": task,
        "messages": [AIMessage(content=f"Completed research: {task}")],
    }


def synthesize_node(state: AgentState) -> dict:
    """Write the final report from all gathered research notes."""
    query = _user_query(state)
    notes = state.get("research_notes", "")
    llm = get_llm()

    prompt = (
        "Write a final research report using ONLY the research notes below. "
        "Answer the request with these sections: Overview, Key Findings, "
        "Analysis, Conclusion. Preserve any source URLs from the notes.\n\n"
        f"Request: {query}\n\nResearch notes:\n{notes}"
    )
    report = message_text(llm.invoke(prompt).content)

    return {
        "final_report": report,
        "messages": [AIMessage(content=report)],
    }