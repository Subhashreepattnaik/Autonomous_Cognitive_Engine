"""
Research agent — a specialist that gathers information from the web and saves
its findings to the virtual file system.

Built with LangChain's `create_agent`, which wraps an LLM and a set of tools
into a ready-to-run ReAct loop (reason -> act -> observe -> repeat).
"""

from langchain.agents import AgentState as BaseAgentState
from langchain.agents import create_agent

from services.llm_service import get_llm
from tools.file_system_tools import edit_file, ls, read_file, write_file
from tools.search_tools import web_search


class ResearchAgentState(BaseAgentState):
    """The research agent's state: LangChain's base agent state (which already
    has `messages`) plus a virtual file system, so the file tools work here."""

    virtual_files: dict[str, str]


RESEARCH_SYSTEM_PROMPT = """You are a meticulous research specialist.

Your job: thoroughly investigate the topic or question you are given and
produce well-sourced findings.

How to work:
- Use the `tavily_search` tool to find current, factual information. Run
  several searches from different angles if one is not enough.
- Always note the source URL next to any important fact.
- When you have gathered substantial findings, save them with `write_file`
  using a clear filename (e.g. "findings_agi_definition.md"). This keeps your
  working context clean and preserves the information for later.
- Use `ls` and `read_file` to review what you have already saved.
- When you have enough, write a clear, organized summary as your final
  response, citing the source URLs you used.

Be accurate and specific. Never invent facts or sources."""


def build_research_agent():
    """Create and return the compiled research agent.

    Returns:
        A runnable LangGraph agent that researches a topic using web search
        and a virtual file system.
    """
    tools = [web_search, write_file, read_file, ls, edit_file]
    return create_agent(
        model=get_llm(),
        tools=tools,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        state_schema=ResearchAgentState,
        name="research_agent",
    )