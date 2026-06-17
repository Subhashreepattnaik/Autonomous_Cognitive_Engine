"""
Shared state for the Autonomous Cognitive Engine.

This is the 'whiteboard' that flows through every node in the LangGraph.
Each node reads the current state and returns the fields it wants to change.
By default a returned field OVERWRITES the old value; the `messages` field is
the exception — its `add_messages` reducer APPENDS instead.
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# A single task in the agent's plan.
class Todo(TypedDict):
    """One item on the agent's to-do list."""

    content: str
    status: Literal["pending", "running", "completed"]


# The full shared state passed between all nodes.
class AgentState(TypedDict):
    """The shared state ('whiteboard') for the whole agent workflow."""

    # Conversation history: user input, AI replies, tool results.
    # The add_messages reducer APPENDS new messages instead of overwriting.
    messages: Annotated[list[AnyMessage], add_messages]

    # The plan: a list of sub-tasks, each with a status.
    todos: list[Todo]

    # The virtual file system: filename -> file contents.
    virtual_files: dict[str, str]

    # Short descriptions of tasks already finished (handy for progress %).
    completed_tasks: list[str]

    # The single task the agent is focused on right now.
    current_task: str

    # Accumulated findings, gathered before writing the final report.
    research_notes: str

    # The finished report text, filled in at the very end.
    final_report: str