"""
Streamlit UI for the Autonomous Cognitive Engine.

Run with:  streamlit run app.py
"""

import streamlit as st

from config import settings
from ui.components import (
    render_current_task,
    render_memory,
    render_progress,
    render_quality_badge,
    render_stats,
    render_todo_board,
)
from evaluation.evaluators import score_report_for_display
from ui.styles import load_css
from utils.helpers import clean_for_display, message_text

st.set_page_config(
    page_title="Autonomous Cognitive Engine",
    page_icon="◆",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* hide Streamlit default chrome for a cleaner, app-like look */
#MainMenu, footer, header { visibility: hidden; }

/* page width + spacing */
.block-container { padding-top: 2rem; max-width: 1080px; }

/* ---------- HERO ---------- */
.hero {
    background: radial-gradient(120% 140% at 0% 0%, #16313A 0%, #0F1A24 55%);
    border: 1px solid #1E3540; border-radius: 20px;
    padding: 2.4rem 2.6rem; margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
}
.hero-logo {
    display: inline-flex; align-items: center; justify-content: center;
    width: 46px; height: 46px; border-radius: 12px;
    background: linear-gradient(135deg, #0D9488, #2DD4BF);
    color: #07221E; font-weight: 800; font-size: 1.35rem;
    margin-bottom: 1rem; box-shadow: 0 6px 18px rgba(13,148,136,0.35);
}
.hero-badge {
    display: inline-block; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #2DD4BF; background: rgba(45,212,191,0.10);
    border: 1px solid rgba(45,212,191,0.25); border-radius: 999px;
    padding: 0.28rem 0.85rem; margin-bottom: 0.9rem;
}
.hero-title {
    font-size: 2.5rem !important; font-weight: 800 !important;
    letter-spacing: -1px; color: #F1F5F9 !important; margin: 0.2rem 0 0.5rem 0;
}
.hero-subtitle {
    font-size: 1.02rem; color: #94A3B8; max-width: 640px; line-height: 1.6; margin: 0;
}

/* headings */
h1 { font-weight: 800; letter-spacing: -0.5px; color: #F1F5F9; }
h2, h3 { font-weight: 700; color: #CBD5E1; }

/* primary buttons */
.stButton > button {
    background: linear-gradient(135deg, #0D9488, #14B8A6);
    color: #FFFFFF; border: none; border-radius: 10px;
    padding: 0.55rem 1.4rem; font-weight: 600; font-size: 0.95rem;
    transition: all 0.18s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(13,148,136,0.35);
}

/* secondary (example) buttons read lighter */
.stButton > button[kind="secondary"] {
    background: #16242F; border: 1px solid #223240; color: #CBD5E1;
    font-weight: 500;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #2DD4BF; color: #2DD4BF; box-shadow: none;
}

/* cards / containers */
[data-testid="stMetric"], .stTabs [data-baseweb="tab-panel"] {
    background: #16242F; border: 1px solid #223240;
    border-radius: 14px; padding: 1rem 1.2rem;
}

/* metric numbers */
[data-testid="stMetricValue"] { color: #2DD4BF; font-weight: 700; }

/* tabs */
.stTabs [data-baseweb="tab"] { font-weight: 600; color: #94A3B8; }
.stTabs [aria-selected="true"] { color: #2DD4BF !important; }

/* progress bar */
.stProgress > div > div > div { background: linear-gradient(90deg,#0D9488,#2DD4BF); }

/* inputs */
.stTextInput textarea, .stTextArea textarea, .stTextInput input {
    background: #0F1A24; border: 1px solid #223240; border-radius: 10px;
    color: #E8EEF2;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_graph():
    settings.validate_settings()
    from graph.supervisor_graph import build_supervisor_graph
    return build_supervisor_graph()


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
     <div class="hero-logo">◆</div>
      <div class="hero-badge">Powered by LangGraph</div>
      <h1 class="hero-title">Autonomous Cognitive Engine</h1>
      <p class="hero-subtitle">
        Give it a complex question. It plans, researches the live web,
        organizes its findings, and writes you a sourced report — on its own.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("ℹ️  New here? What this app does & how it works", expanded=False):
    st.markdown("""
**Autonomous Cognitive Engine (ACE)** turns one research question into a complete, sourced report — automatically.

**How it works**
1. **Plan** — it breaks your question into 3–5 focused sub-tasks.
2. **Delegate** — a *supervisor agent* sends each task to the specialist it needs: **Research** (live web search), **Summarization**, **Analysis**, or **Coding**.
3. **Remember** — findings are saved to a **Virtual File System** so nothing is lost between steps.
4. **Synthesize** — it writes a structured, sourced report and scores its own quality (0–10).

**Try it:** type a question below (or pick an example) and click **Start Research**. Open the **Memory** tab afterward to see what each agent saved.
    """)

# ---- Input section ----
st.markdown("#### Try an example, or write your own")
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

start = st.button("Start Research", type="primary", use_container_width=True)

# ---- Run the research with a live dashboard ----
if start and query.strip():
    graph = get_graph()
    inputs = {"messages": [{"role": "user", "content": query.strip()}]}

    st.markdown("### Live Execution")
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

    # Judge the report quality (one extra LLM call) for the badge.
    if final_state and final_state.get("final_report"):
        with st.spinner("Scoring report quality…"):
            try:
                final_state["quality"] = score_report_for_display(
                    final_state["final_report"]
                )
            except Exception:
                final_state["quality"] = None

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
        if st.button("Reset", use_container_width=True):
            st.session_state.do_reset = True
            st.rerun()

    render_stats(result)

    report_tab, memory_tab, plan_tab = st.tabs(
        ["Report", "Memory", "Plan"]
    )

    with report_tab:
        if result.get("final_report"):
            if result.get("quality"):
                render_quality_badge(result["quality"])

            from services.pdf_service import generate_report_pdf
            pdf_bytes = generate_report_pdf(
                query=message_text(result["messages"][0].content),
                report=message_text(result["final_report"]),
            )
            st.download_button(
                "⬇ Download PDF",
                data=pdf_bytes,
                file_name="research_report.pdf",
                mime="application/pdf",
                type="primary",
            )
            st.markdown(clean_for_display(message_text(result["final_report"])))
        else:
            st.info("No report was produced.")

    with memory_tab:
        render_memory(result.get("virtual_files", {}))

    with plan_tab:
        render_todo_board(result.get("todos", []), live=False)