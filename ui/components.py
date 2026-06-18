"""Reusable Streamlit UI components for the cognitive engine dashboard."""

import streamlit as st

# Visual style per task status: (icon, color, label)
STATUS_STYLES = {
    "completed": ("✅", "#22c55e", "Completed"),
    "running": ("🔄", "#3b82f6", "Running"),
    "pending": ("⏳", "#94a3b8", "Pending"),
}


def render_progress(todos: list[dict]) -> None:
    """Show a progress bar + percentage based on completed tasks."""
    if not todos:
        return
    total = len(todos)
    done = sum(1 for t in todos if t["status"] == "completed")
    fraction = done / total
    st.progress(
        fraction, text=f"Progress: {done}/{total} tasks ({int(fraction * 100)}%)"
    )


def render_current_task(todos: list[dict]) -> None:
    """Show what the engine is working on right now."""
    next_task = next(
        (t["content"] for t in todos if t["status"] != "completed"), None
    )
    if next_task:
        st.info(f"🔬 Researching: {next_task}")
    else:
        st.info("🧩 Synthesizing the final report…")


def render_todo_board(todos: list[dict], live: bool = False) -> None:
    """Render the task plan as colored status cards.

    When live=True (graph still running), the first not-yet-completed task is
    shown as 'running' to reflect what the engine is working on right now.
    """
    if not todos:
        return
    st.markdown("##### 📋 Task Plan")

    marked_running = False
    for t in todos:
        status = t["status"]
        if live and status != "completed" and not marked_running:
            status = "running"  # highlight the current task
            marked_running = True
        icon, color, label = STATUS_STYLES.get(status, ("•", "#94a3b8", status))
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.6rem;
                        padding:0.55rem 0.8rem; margin-bottom:0.4rem;
                        border-radius:10px; background:rgba(148,163,184,0.08);
                        border-left:4px solid {color};">
              <span>{icon}</span>
              <span style="flex:1; color:#e2e8f0;">{t['content']}</span>
              <span style="font-size:0.75rem; font-weight:600;
                           color:{color};">{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_memory(virtual_files: dict[str, str]) -> None:
    """Show the agent's saved files (its offloaded context) for inspection."""
    if not virtual_files:
        st.caption("No files were saved during this run.")
        return

    st.caption(
        f"The agent offloaded its findings to {len(virtual_files)} file(s) "
        "to keep its working context small. Open any to inspect the raw notes."
    )
    for name, content in virtual_files.items():
        with st.expander(f"📄 {name}  ·  {len(content)} chars"):
            st.markdown(content)


def render_stats(result: dict) -> None:
    """Show a small row of run metrics."""
    todos = result.get("todos", [])
    files = result.get("virtual_files", {})
    done = sum(1 for t in todos if t["status"] == "completed")

    c1, c2, c3 = st.columns(3)
    c1.metric("Tasks completed", f"{done}/{len(todos)}" if todos else "0")
    c2.metric("Files saved", len(files))
    c3.metric("Report length", f"{len(result.get('final_report', ''))} chars")

def render_quality_badge(result: dict) -> None:
    """Show a colored 'Quality Score: X/10' badge for the report."""
    score = result.get("score", 0)
    grade = result.get("grade", "N/A")
    reason = result.get("reason", "")

    # Color by score band.
    if score >= 8:
        color, bg = "#22c55e", "rgba(34,197,94,0.12)"
    elif score >= 6:
        color, bg = "#eab308", "rgba(234,179,8,0.12)"
    else:
        color, bg = "#ef4444", "rgba(239,68,68,0.12)"

    st.markdown(
        f"""
        <div style="display:inline-flex; align-items:center; gap:0.5rem;
                    padding:0.5rem 1rem; border-radius:999px; background:{bg};
                    border:1px solid {color};">
          <span style="font-weight:700; color:{color};">
            ✅ Quality Score: {score}/10 — {grade}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if reason:
        st.caption(reason)