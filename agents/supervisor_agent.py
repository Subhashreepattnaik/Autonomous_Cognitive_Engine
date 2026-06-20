"""
Supervisor agent — plans the work, delegates sub-tasks to specialist agents,
and synthesizes the final report.
The `task` tool is the delegation mechanism: calling it runs an entire
specialist agent and returns its result to the Supervisor.
"""
from typing import Literal
from langchain.agents import AgentState as BaseAgentState, create_agent
from langchain.tools import tool
from agents.research_agent import build_research_agent
from agents.search_agent import build_search_agent
from agents.summarization_agent import build_summarization_agent
from services.llm_service import get_llm
from state.agent_state import Todo
from tools.file_system_tools import ls, read_file, write_file
from tools.planning_tools import write_todos
from utils.helpers import message_text

# The specialists the Supervisor is allowed to delegate to.
AgentName = Literal["research_agent", "search_agent", "summarization_agent"]

class SupervisorState(BaseAgentState):
    """Supervisor's state: the base agent state plus our plan and file system,
    so `write_todos` and the file tools have channels to write to."""
    todos: list[Todo]
    virtual_files: dict[str, str]

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor of an autonomous research \
team. You coordinate specialist agents to produce a thorough, well-sourced \
research report.

ALWAYS begin by calling the `write_todos` tool to break the user's request into
a short list of clear sub-tasks. Do this before anything else.

Then work through the tasks. Delegate using the `task` tool:
- "research_agent": deep web research that gathers detailed, sourced findings.
  Use this for the main investigation work.
- "search_agent": a quick, focused web lookup for a single fact or question.
- "summarization_agent": condenses long text you give it into a tight summary.

For each sub-task: delegate it to the best specialist with `task`, then save the
important results with `write_file` using a descriptive filename.

When every sub-task is done, use `ls` and `read_file` to gather what you saved,
then write the final report as your response. Structure it with clear sections:
Overview, Key Findings, Analysis, and Conclusion. Always preserve the source
URLs the specialists gave you.

Rules:
- Always call `write_todos` first.
- Delegate the research — do not try to do it all yourself.
- Never invent facts or sources."""


def build_supervisor_agent():
    """Create and return the compiled Supervisor agent, with its specialists
    wired in behind the `task` delegation tool."""
    # Build each specialist once; the task tool closes over this registry.
    sub_agents = {
        "research_agent": build_research_agent(),
        "search_agent": build_search_agent(),
        "summarization_agent": build_summarization_agent(),
    }

    @tool
    def task(description: str, agent: AgentName) -> str:
        """Delegate a sub-task to a specialist agent and return its result.

        Args:
            description: A clear, self-contained description of the sub-task.
            agent: Which specialist to use — "research_agent" for deep web
                research, "search_agent" for a quick lookup, or
                "summarization_agent" to condense text you provide.
        """
        sub = sub_agents.get(agent)
        if sub is None:
            return f"Error: unknown agent '{agent}'."
        try:
            result = sub.invoke(
                {"messages": [{"role": "user", "content": description}]}
            )
        except Exception as exc:  # keep the Supervisor alive if a specialist fails
            return f"The {agent} failed on this sub-task: {exc}"
        return message_text(result["messages"][-1].content)

    tools = [write_todos, task, write_file, read_file, ls]
    return create_agent(
        model=get_llm(),
        tools=tools,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        state_schema=SupervisorState,
        name="supervisor",
    )