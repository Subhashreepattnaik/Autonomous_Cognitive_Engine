"""
Planning tools for the Autonomous Cognitive Engine.

Contains `write_todos`: the tool the agent uses to create its plan — a list of
sub-tasks stored in the shared graph state. New tasks start as "pending"; the
graph updates their statuses as work progresses.
"""

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from state.agent_state import Todo


@tool
def write_todos(tasks: list[str], runtime: ToolRuntime) -> Command:
    """Create the plan: a list of sub-task descriptions to work through.

    Call this FIRST, before doing anything else, to break the user's request
    into clear sub-tasks. Each task you pass becomes a to-do with status
    "pending".

    Args:
        tasks: Short descriptions of the sub-tasks, in the order to do them.
    """
    todos: list[Todo] = [{"content": t, "status": "pending"} for t in tasks]

    summary = "\n".join(f"  [pending] {t}" for t in tasks)
    message = f"Plan created with {len(todos)} task(s):\n{summary}"

    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(content=message, tool_call_id=runtime.tool_call_id)
            ],
        }
    )