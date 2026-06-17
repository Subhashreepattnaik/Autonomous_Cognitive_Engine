"""
Web search tool for the Autonomous Cognitive Engine.

Wraps Tavily, a search engine built for AI agents: instead of raw HTML, it
returns clean, extracted, relevant content that an LLM can use directly.

Building the tool reads TAVILY_API_KEY from the environment, which
config.settings has already loaded from your .env file.
"""

from langchain_tavily import TavilySearch

from config import settings

# Fail fast with a clear message if the key is missing — a research agent
# simply cannot function without search, so we stop here rather than crash
# cryptically mid-run.
if not settings.TAVILY_API_KEY:
    raise EnvironmentError(
        "TAVILY_API_KEY is missing. Add it to your .env file (see .env.example)."
    )

# The configured search tool the agents will use.
#   max_results: how many sources to return per search (more = richer context,
#                but more tokens and Tavily credits per call).
#   topic="general": standard web search (Tavily also offers "news", "finance").
web_search = TavilySearch(
    max_results=3,
    topic="general",
)