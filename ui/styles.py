"""Custom CSS for the Streamlit UI — gives the app a modern SaaS look."""


def load_css() -> str:
    """Return the CSS to inject into the Streamlit page."""
    return """
    <style>
      /* Hide Streamlit's default chrome for a cleaner app feel */
      #MainMenu, footer {visibility: hidden;}

      .hero { text-align: center; padding: 2.5rem 1rem 1rem 1rem; }
      .hero-badge {
        display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px;
        background: rgba(99, 102, 241, 0.12); color: #818cf8;
        font-size: 0.85rem; font-weight: 600; margin-bottom: 1rem;
      }
      .hero-title {
        font-size: 3rem; font-weight: 800; margin: 0;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
      }
      .hero-subtitle {
        font-size: 1.1rem; color: #94a3b8; max-width: 640px;
        margin: 0.75rem auto 0 auto; line-height: 1.6;
      }
      .stButton button[kind="primary"] {
        font-size: 1.05rem; padding: 0.6rem 1rem; border-radius: 10px;
      }
    </style>
    """