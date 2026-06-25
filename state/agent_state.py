"""
state/agent_state.py  —  shared state for the supervisor / multi-agent version.

This extends your original AgentState with three fields the supervisor needs to
route work: next_agent, task_instruction, and step_count (a safety counter that
bounds the ReAct loop so it can never run forever).
"""

from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Todo(TypedDict):
    content: str
    status: str           # "pending" | "running" | "completed"


class AgentState(TypedDict):
    # --- conversation / memory (unchanged from your original) ---
    messages: Annotated[list[AnyMessage], add_messages]
    todos: list[Todo]
    virtual_files: dict[str, str]      # the Virtual File System (VFS)
    completed_tasks: list[str]
    research_notes: str
    final_report: str

    # --- NEW: supervisor routing fields ---
    next_agent: str          # which sub-agent runs next: research | analyze | code | synthesize
    task_instruction: str    # the specific instruction the supervisor passes to that sub-agent
    step_count: int          # safety counter to bound the supervisor (ReAct) loop