"""
Web search agent — a focused specialist that answers a question using web
search alone. Lighter than the research agent: no file system, just search.
"""
from langchain.agents import create_agent
from services.llm_service import get_llm
from tools.search_tools import web_search

SEARCH_SYSTEM_PROMPT = """You are a focused web search specialist.
Given a question or topic, use the `tavily_search` tool to find current,
relevant information, then return a concise, factual answer.
Rules:
- Search the web rather than relying on memory.
- Include the source URL for each key fact.
- Keep your answer tight and on-topic. Do not pad or speculate."""

def build_search_agent():
    """Create and return the compiled web search agent."""
    return create_agent(
        model=get_llm(),
        tools=[web_search],
        system_prompt=SEARCH_SYSTEM_PROMPT,
        name="search_agent",
    )