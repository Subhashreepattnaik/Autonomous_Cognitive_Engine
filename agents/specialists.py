"""
agents/specialists.py  —  the FOUR specialist sub-agents the supervisor delegates to.

Per the project specification, the main (supervisor) agent invokes specialised
sub-agents, each operating with its own focused context (and, for research, its
own tool). Each function below IS one specialist sub-agent:

    research   -> gathers information using the Tavily web-search tool
    summarize  -> condenses raw findings into clean notes
    analyze    -> interprets/compares notes to produce insight
    code       -> writes or explains code when a task requires it

All model calls route through invoke_llm (your single chokepoint), so they get
automatic rate limiting and 120B -> 20B -> 70B failover for free.

NOTE on reliability: the Research agent calls Tavily IN CODE rather than via a
model-issued tool call, because on free-tier models a model-issued call is the
main source of malformed-call errors. This keeps the sub-agent reliable while
still being a genuine specialist that uses the search tool.
"""

from services.llm_service import invoke_llm
from tools.search_tools import web_search
from utils.helpers import message_text


# ---------- 1) RESEARCH AGENT (uses the Tavily search tool) ----------
def run_research_agent(instruction: str) -> str:
    """Gather factual, sourced information for a sub-task."""
    try:
        raw = web_search.invoke({"query": instruction})
        results = raw.get("results", []) if isinstance(raw, dict) else []
        sources = "\n\n".join(
            f"Source: {r.get('url', '')}\n{r.get('content', '')}" for r in results
        ) or "No results found."
    except Exception as exc:
        sources = f"Web search failed: {exc}"

    prompt = (
        "You are the RESEARCH AGENT. Organise the following web findings for the "
        f"task '{instruction}' into clear, factual bullet points. Keep the source "
        f"URLs next to each point.\n\nFindings:\n{sources}"
    )
    return message_text(invoke_llm(prompt).content)


# ---------- 2) SUMMARIZATION AGENT (no tools) ----------
def run_summarization_agent(text: str) -> str:
    """Condense raw findings into clean, factual notes."""
    prompt = (
        "You are the SUMMARIZATION AGENT. Condense the following into clear, "
        "factual notes. Keep the key points and any source URLs. No fluff.\n\n"
        f"{text}"
    )
    return message_text(invoke_llm(prompt).content)


# ---------- 3) ANALYSIS AGENT (no tools) ----------
def run_analysis_agent(notes: str) -> str:
    """Interpret and compare notes to produce genuine insight."""
    prompt = (
        "You are the ANALYSIS AGENT. Interpret and compare the following research "
        "notes: identify patterns, trade-offs, tensions, and practical "
        "implications. Provide insight, not a restatement.\n\n"
        f"{notes}"
    )
    return message_text(invoke_llm(prompt, temperature=0).content)


# ---------- 4) CODING AGENT (no tools) ----------
def run_coding_agent(instruction: str) -> str:
    """Write or explain code for a task that requires it."""
    prompt = (
        "You are the CODING AGENT. Write correct, well-commented code for the "
        "task below, in a single fenced code block, followed by a brief "
        "explanation. If the task does not actually need code, say so clearly.\n\n"
        f"Task: {instruction}"
    )
    return message_text(invoke_llm(prompt, temperature=0).content)