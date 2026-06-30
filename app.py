"""Adaptive RAG AI — Streamlit shell that serves the HTML/JS frontend inline."""
import pathlib
import streamlit as st

st.set_page_config(
    page_title="Adaptive RAG AI",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide ALL Streamlit chrome — header, footer, sidebar, padding
st.markdown("""
<style>
  #MainMenu, header, footer, [data-testid="stToolbar"],
  [data-testid="stDecoration"], [data-testid="stStatusWidget"],
  [data-testid="collapsedControl"] { display: none !important; visibility: hidden !important; }

  .stApp { background: transparent !important; }
  .block-container { padding: 0 !important; max-width: 100% !important; }
  [data-testid="stAppViewContainer"] { padding: 0 !important; }
  [data-testid="stVerticalBlock"] { gap: 0 !important; padding: 0 !important; }
  section[data-testid="stSidebar"] { display: none !important; }

  /* Remove iframe border */
  iframe { border: none !important; display: block !important; }
</style>
""", unsafe_allow_html=True)

# Load frontend files
_here = pathlib.Path(__file__).parent
_html_path = _here / "frontend" / "index.html"
_js_path   = _here / "frontend" / "app.js"

html_content = _html_path.read_text(encoding="utf-8")
js_content   = _js_path.read_text(encoding="utf-8")

# Inline app.js so the iframe doesn't need to fetch /app.js (would 404 on Streamlit Cloud)
html_content = html_content.replace(
    '<script src="/app.js"></script>',
    f"<script>\n{js_content}\n</script>"
)

# Force API_BASE to Render (Streamlit Cloud is not the API server)
# Already handled in app.js auto-detection — hostname won't be localhost on Streamlit Cloud
# But override explicitly to be safe:
html_content = html_content.replace(
    "const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'\n    ? ''\n    : 'https://adaptive-rag-knowledge-retrieval.onrender.com';",
    "const API_BASE = 'https://adaptive-rag-knowledge-retrieval.onrender.com';"
)

# Render the full app inside a full-viewport iframe
st.components.v1.html(html_content, height=900, scrolling=False)
