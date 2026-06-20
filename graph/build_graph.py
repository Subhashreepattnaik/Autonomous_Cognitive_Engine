"""
Assembles the agent nodes into a stateful LangGraph workflow.
The graph enforces the order: plan -> research (looped once per task) ->
synthesize -> END. Because the EDGES decide what runs next (not the LLM), the
system always plans first, researches every task in turn, and only synthesizes
once all tasks are complete.
"""

from langgraph.graph import END, START, StateGraph
from graph.nodes import plan_node, research_node, synthesize_node
from state.agent_state import AgentState

def _route_after_research(state: AgentState) -> str:
    """Loop back to research while pending tasks remain, else synthesize."""
    has_pending = any(t["status"] == "pending" for t in state["todos"])
    return "research" if has_pending else "synthesize"

def build_graph():
    """Build and compile the full cognitive-engine graph."""
    builder = StateGraph(AgentState)

    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "research")
    builder.add_conditional_edges(
        "research",
        _route_after_research,
        {"research": "research", "synthesize": "synthesize"},
    )
    builder.add_edge("synthesize", END)

    return builder.compile()