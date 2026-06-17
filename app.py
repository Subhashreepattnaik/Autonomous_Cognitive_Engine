"""
Streamlit UI for the Autonomous Cognitive Engine.

Run with:  streamlit run app.py
"""

import streamlit as st

from config import settings
from graph.build_graph import build_graph
from ui.components import (
    render_current_task,
    render_memory,
    render_progress,
    render_stats,
    render_todo_board,
)
from ui.styles import load_css
from utils.helpers import message_text

st.set_page_config(
    page_title="Autonomous Cognitive Engine",
    page_icon="🧠",
    layout="wide",
)
st.markdown(load_css(), unsafe_allow_html=True)


@st.cache_resource
def get_graph():
    settings.validate_settings()
    return build_graph()


if "result" not in st.session_state:
    st.session_state.result = None
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# ---- Handle a reset request (must run BEFORE widgets are drawn) ----
if st.session_state.get("do_reset"):
    st.session_state.result = None
    st.session_state.query_input = ""
    st.session_state.do_reset = False
    st.rerun()

EXAMPLE_PROMPTS = [
    "Generate a research report on Artificial General Intelligence",
    "Research the main risks and benefits of solar energy",
    "Analyze the current state of quantum computing in 2026",
]

# ---- Hero / landing ----
st.markdown(
    """
    <div class="hero">
      <div class="hero-badge">⚡ Powered by LangGraph</div>
      <h1 class="hero-title">Autonomous Cognitive Engine</h1>
      <p class="hero-subtitle">
        Give it a complex question. It plans, researches the live web,
        organizes its findings, and writes you a sourced report — on its own.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Input section ----
st.markdown("#### Try an example, or write your own:")
cols = st.columns(len(EXAMPLE_PROMPTS))
for col, example in zip(cols, EXAMPLE_PROMPTS):
    if col.button(example, use_container_width=True):
        st.session_state.query_input = example

query = st.text_area(
    "Research request",
    key="query_input",
    placeholder="e.g. Generate a research report on the future of nuclear fusion",
    height=120,
    label_visibility="collapsed",
)

start = st.button("🚀 Start Research", type="primary", use_container_width=True)

# ---- Run the research with a live dashboard ----
if start and query.strip():
    graph = get_graph()
    inputs = {"messages": [{"role": "user", "content": query.strip()}]}

    st.markdown("### 🔄 Live Execution")
    dashboard = st.empty()
    final_state = None

    for chunk in graph.stream(inputs, stream_mode="values"):
        final_state = chunk
        todos = chunk.get("todos", [])
        with dashboard.container():
            render_progress(todos)
            render_current_task(todos)
            render_todo_board(todos, live=True)

    dashboard.empty()  # clear the live view; settled view shows below
    st.session_state.result = final_state

elif start:
    st.warning("Please enter a research request first.")

# ---- Persistent results view (survives reruns) ----
result = st.session_state.result
if result:
    st.markdown("---")

    header_col, reset_col = st.columns([4, 1])
    with header_col:
        st.markdown("### Results")
    with reset_col:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.do_reset = True
            st.rerun()

    render_stats(result)
    # ... the tabs block stays exactly as it is below ...

    report_tab, memory_tab, plan_tab = st.tabs(
        ["📄 Report", "🧠 Memory", "📋 Plan"]
    )

    with report_tab:
        if result.get("final_report"):
            from services.pdf_service import generate_report_pdf

            pdf_bytes = generate_report_pdf(
                query=message_text(result["messages"][0].content),
                report=message_text(result["final_report"]),
            )
            st.download_button(
                "⬇️ Download PDF",
                data=pdf_bytes,
                file_name="research_report.pdf",
                mime="application/pdf",
                type="primary",
            )
            st.markdown(message_text(result["final_report"]))
        else:
            st.info("No report was produced.")

    with memory_tab:
        render_memory(result.get("virtual_files", {}))

    with plan_tab:
        render_todo_board(result.get("todos", []), live=False)