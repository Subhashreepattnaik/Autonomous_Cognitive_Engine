"""
Milestone evaluators.
Each evaluator inspects a finished run's state and returns:
    {"metric": str, "passed": bool, "score": ..., "reason": str}
Milestones 1-3 are deterministic structural checks (no extra LLM calls — they
verify the mechanism fired, the same evidence a LangSmith trace shows).
Milestone 4 uses an LLM-as-a-judge to grade report quality.
"""
import json
import re
from services.llm_service import get_llm, invoke_llm
from utils.helpers import message_text

def evaluate_planning(state: dict) -> dict:
    """Milestone 1 — Task Decomposition Accuracy."""
    todos = state.get("todos", [])
    count = len(todos)
    substantive = all(len(t.get("content", "")) > 10 for t in todos)
    passed = 3 <= count <= 6 and substantive
    return {
        "metric": "M1 Task Planning",
        "passed": passed,
        "score": count,
        "reason": f"{count} sub-tasks; substantive={substantive}",
    }

def evaluate_offloading(state: dict) -> dict:
    """Milestone 2 — Correct File System / Context Offloading Usage."""
    files = state.get("virtual_files", {})
    todos = state.get("todos", [])
    completed = sum(1 for t in todos if t.get("status") == "completed")
    non_empty = bool(files) and all(len(c) > 20 for c in files.values())
    passed = len(files) >= 1 and non_empty
    return {
        "metric": "M2 Context Offloading",
        "passed": passed,
        "score": len(files),
        "reason": f"{len(files)} files saved for {completed} completed tasks",
    }

def evaluate_delegation(state: dict) -> dict:
    """Milestone 3 — Successful Delegation & Result Integration.

    Each research task delegates summarization to the summarization sub-agent
    and integrates the condensed result into the shared research_notes. This
    checks that delegation produced integrated content and all tasks completed.
    """
    notes = message_text(state.get("research_notes", ""))
    todos = state.get("todos", [])
    all_done = bool(todos) and all(t.get("status") == "completed" for t in todos)
    passed = len(notes) > 100 and all_done
    return {
        "metric": "M3 Delegation & Integration",
        "passed": passed,
        "score": len(notes),
        "reason": f"notes={len(notes)} chars; all_completed={all_done}",
    }

def evaluate_report_quality(state: dict) -> dict:
    """Milestone 4 — Output Quality via LLM-as-a-judge."""
    report = message_text(state.get("final_report", ""))
    if not report.strip():
        return {
            "metric": "M4 Report Quality",
            "passed": False,
            "score": 0,
            "reason": "No report produced.",
        }

    prompt = (
        "You are evaluating a research report produced by an automated web-research "
        "agent (not a human or an academic paper). Judge it ONLY on what such a "
        "report should achieve. Rate it 1 to 5 on:\n"
        "  1. Structure: clear sections (Overview, Key Findings, Analysis, Conclusion).\n"
        "  2. Sourcing: refers to / cites where information came from.\n"
        "  3. Coherence: reads clearly and stays on topic.\n"
        "  4. Completeness: covers the question and is not cut off.\n\n"
        "Do NOT require academic methodology, literature-review rigour, or formal "
        "citation styles — those are out of scope for a web-research summary. A report "
        "that is well-structured, sourced, coherent, and complete is 'good' (4). "
        "Reserve 'excellent' (5) for that PLUS strong analytical depth.\n\n"
        "Respond with ONLY a JSON object, no other text:\n"
        '{"score": <1-5 integer>, "grade": "poor|fair|good|excellent", '
        '"justification": "<one sentence>"}\n\n'
        f"REPORT:\n{report[:8000]}"
    )
    raw = message_text(invoke_llm(prompt, temperature=0).content)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except (json.JSONDecodeError, AttributeError):
        data = {}

    score = data.get("score", 0)
    grade = str(data.get("grade", "")).lower()
    passed = grade in {"good", "excellent"} or (
        isinstance(score, (int, float)) and score >= 4
    )
    return {
        "metric": "M4 Report Quality",
        "passed": passed,
        "score": score,
        "reason": data.get("justification", "Could not parse judge response."),
    }

ALL_EVALUATORS = [
    evaluate_planning,
    evaluate_offloading,
    evaluate_delegation,
    evaluate_report_quality,
]

def score_report_for_display(report: str) -> dict:
    """Judge a single report for the UI. Returns {score:0-10, grade, reason}.
    One LLM call. Used to show a 'Quality Score' badge after a research run.
    """
    report = message_text(report)
    if not report.strip():
        return {"score": 0, "grade": "no report", "reason": "Empty report."}

    prompt = (
        "You are a strict evaluator of research reports. Rate the report below "
        "from 0 to 10, considering completeness, clear structure, whether "
        "sources are cited, and overall coherence.\n\n"
        "Respond with ONLY a JSON object, no other text:\n"
        '{"score": <0-10 number>, "grade": "<Poor|Fair|Good|Excellent>", '
        '"reason": "<one short sentence>"}\n\n'
        f"REPORT:\n{report[:8000]}"
    )
    raw = message_text(invoke_llm(prompt, temperature=0).content)
    import json
    import re
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except (json.JSONDecodeError, AttributeError):
        data = {}

    try:
        score = float(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    return {
        "score": round(score, 1),
        "grade": str(data.get("grade", "N/A")),
        "reason": str(data.get("reason", "Could not parse judge response.")),
    }