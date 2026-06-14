"""Adaptive RAG AI — Streamlit frontend (dark sidebar + white main area)."""
import html
import json
import os
import time
import uuid

import requests
import streamlit as st
import streamlit.components.v1 as components

# ── AUTO-START BACKEND ────────────────────────────────────────────────────────
# On Streamlit Cloud only one process runs, so we launch the FastAPI server
# in a background daemon thread before anything else.
try:
    from backend_runner import ensure_backend_running
    ensure_backend_running(port=8080, wait_seconds=10.0)
except Exception:
    pass  # graceful: if import fails we still try to connect via API_URL


def _make_logo_pil(size: int = 80):
    """Three radiating signal arcs + source dot — Grok AI style, deep dark bg."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Very dark navy — matches Grok AI's near-black aesthetic
        r_bg = size // 5
        bg = (6, 9, 32)
        try:
            draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=r_bg, fill=bg)
        except (AttributeError, TypeError):
            draw.rectangle([0, 0, size, size], fill=bg)
            for cx2, cy2 in [(r_bg, r_bg), (size-r_bg, r_bg), (r_bg, size-r_bg), (size-r_bg, size-r_bg)]:
                draw.ellipse([cx2-r_bg, cy2-r_bg, cx2+r_bg, cy2+r_bg], fill=bg)

        s = size / 24.0
        cx, cy = 7.0 * s, 12.0 * s   # source dot centre

        # Outer glow halo behind dot
        gh = 3.2 * s
        draw.ellipse([(cx-gh, cy-gh), (cx+gh, cy+gh)], fill=(80, 120, 255, 55))

        # Source dot — full bright white
        dr = 1.6 * s
        draw.ellipse([(cx-dr, cy-dr), (cx+dr, cy+dr)], fill=(255, 255, 255, 255))

        # Three arcs: small / medium / large, decreasing brightness
        arc_cfg = [
            (3.5, (255, 255, 255, 255), max(2, int(2.0 * s))),
            (6.0, (200, 210, 255, 165), max(2, int(1.7 * s))),
            (9.0, (150, 165, 230,  90), max(1, int(1.4 * s))),
        ]
        for r_arc, color, lw in arc_cfg:
            rp = r_arc * s
            # PIL arc: 270°→90° clockwise = right-facing D-shape
            draw.arc([(cx-rp, cy-rp), (cx+rp, cy+rp)],
                     start=270, end=90, fill=color, width=lw)

        return img
    except Exception:
        return "◉"


_LOGO_PIL = _make_logo_pil(80)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive RAG AI",
    page_icon=_LOGO_PIL,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a Bug": None,
                "About": "Adaptive RAG AI — Intelligent Document Intelligence"},
)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MAX_UPLOADS = 5
GAP_COMPLETENESS_THRESHOLD = 8


def _resolve_api_base() -> str:
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except Exception:
        pass
    return os.getenv("API_URL", "http://localhost:8080")


API_BASE = _resolve_api_base()


# ── LOGO ──────────────────────────────────────────────────────────────────────
def _logo_html(size: int = 36, radius: int = 10) -> str:
    """Three radiating signal arcs + source dot — Grok AI aesthetic."""
    ic = int(size * 0.82)
    return (
        f"<div style='width:{size}px;height:{size}px;"
        "background:linear-gradient(150deg,#06091e 0%,#0b1038 100%);"
        f"border-radius:{radius}px;"
        "display:flex;align-items:center;justify-content:center;"
        "box-shadow:0 4px 24px rgba(37,99,235,0.30),0 0 0 1px rgba(255,255,255,0.07);"
        "flex-shrink:0'>"

        f"<svg width='{ic}' height='{ic}' viewBox='0 0 24 24' fill='none'"
        " xmlns='http://www.w3.org/2000/svg'>"

        # Soft glow halo behind source dot
        "<circle cx='7' cy='12' r='3.2' fill='rgba(80,130,255,0.18)'/>"

        # Source dot — bright white
        "<circle cx='7' cy='12' r='1.6' fill='white'/>"

        # Inner arc (r=3.5) — full white
        "<path d='M7,8.5 A3.5,3.5 0 0,1 7,15.5'"
        " stroke='white' stroke-width='2.0' stroke-linecap='round'/>"

        # Middle arc (r=6) — 60% opacity
        "<path d='M7,6 A6,6 0 0,1 7,18'"
        " stroke='white' stroke-width='1.8' stroke-linecap='round' opacity='0.55'/>"

        # Outer arc (r=9) — 28% opacity
        "<path d='M7,3 A9,9 0 0,1 7,21'"
        " stroke='white' stroke-width='1.5' stroke-linecap='round' opacity='0.28'/>"

        "</svg></div>"
    )


# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ════════════════════════════════════════
   WHITE MAIN AREA
   ════════════════════════════════════════ */
.stApp                               { background-color: #F8FAFC !important; }
[data-testid="stMain"]               { background-color: #F8FAFC !important; }
[data-testid="stMainBlockContainer"] {
    background-color: #F8FAFC !important;
    max-width: 920px !important;
    padding: 2rem 2.5rem 3rem !important;
}
.main  { background-color: #F8FAFC !important; }
.block-container { background-color: #F8FAFC !important; }

/* ════════════════════════════════════════
   DARK SIDEBAR — base
   ════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* All sidebar text defaults to slate */
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] div { color: #94A3B8 !important; }

/* Headings stay white */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #FFFFFF !important; }

/* Sidebar hr */
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; }

/* ════════════════════════════════════════
   SIDEBAR — FILE UPLOADER
   The visible white box is stFileUploaderDropzone,
   NOT stFileUploader (which is just a layout wrapper).
   ════════════════════════════════════════ */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px dashed rgba(255,255,255,0.18) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(37,99,235,0.55) !important;
    background: rgba(37,99,235,0.06) !important;
}
/* The inner content wrapper — must also be dark */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div {
    background: transparent !important;
}
/* Upload cloud icon */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg {
    fill: #475569 !important;
    opacity: 0.6 !important;
}
/* "Drag and drop" label */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span {
    color: #64748B !important;
    font-size: 13px !important;
}
/* "Limit 10MB" small text */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
    color: #475569 !important;
    font-size: 11px !important;
}
/* "Browse files" button */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: rgba(37,99,235,0.18) !important;
    color: #93C5FD !important;
    border: 1px solid rgba(37,99,235,0.40) !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 5px 14px !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(37,99,235,0.30) !important;
    color: #BFDBFE !important;
}
/* Already-uploaded file chips in the uploader */
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] p {
    color: #94A3B8 !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDeleteBtn"] button {
    color: #475569 !important;
    background: transparent !important;
    border: none !important;
}

/* ════════════════════════════════════════
   SIDEBAR — TOGGLES
   Labels are <p> inside .stToggle
   ════════════════════════════════════════ */
[data-testid="stSidebar"] [data-testid="stToggle"] p,
[data-testid="stSidebar"] [data-testid="stToggle"] label,
[data-testid="stSidebar"] .stToggle p,
[data-testid="stSidebar"] .stToggle label,
[data-testid="stSidebar"] .stToggle span {
    color: #94A3B8 !important;
    font-size: 13px !important;
    font-weight: 400 !important;
}

/* ════════════════════════════════════════
   SIDEBAR — BUTTONS
   ════════════════════════════════════════ */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background-color: rgba(255,255,255,0.06) !important;
    color: #CBD5E1 !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(255,255,255,0.11) !important;
    color: #FFFFFF !important;
    border-color: rgba(255,255,255,0.18) !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button {
    background-color: transparent !important;
    color: #64748B !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    width: 100% !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button:hover {
    background-color: rgba(255,255,255,0.06) !important;
    color: #CBD5E1 !important;
}

/* ════════════════════════════════════════
   CHAT INPUT — light style
   ════════════════════════════════════════ */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div {
    background: #FFFFFF !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] {
    border: 1.5px solid #E2E8F0 !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.06) !important;
}
[data-testid="stChatInput"] > div { border: none !important; }
[data-testid="stChatInput"] textarea {
    color: #0F172A !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #94A3B8 !important; }
[data-testid="stChatInput"]:focus-within {
    border-color: #93C5FD !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.10) !important;
}
[data-testid="stChatInput"] button {
    background: #2563EB !important;
    border-radius: 8px !important;
    transition: background 0.15s !important;
}
[data-testid="stChatInput"] button:hover { background: #1D4ED8 !important; }

/* ════════════════════════════════════════
   CHAT MESSAGES
   ════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
}

/* ════════════════════════════════════════
   EXPANDERS — light
   ════════════════════════════════════════ */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05) !important;
    margin-top: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #64748B !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
}
[data-testid="stExpander"] summary:hover { color: #0F172A !important; }

/* ════════════════════════════════════════
   MAIN AREA BUTTONS
   ════════════════════════════════════════ */
.stButton > button {
    background-color: #FFFFFF !important;
    color: #334155 !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.15s !important;
    width: 100%;
}
.stButton > button:hover {
    border-color: #2563EB !important;
    color: #2563EB !important;
    background-color: #EFF6FF !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08) !important;
}
[data-testid="stDownloadButton"] > button {
    background-color: transparent !important;
    color: #64748B !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    width: 100%;
    transition: all 0.15s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}

/* ════════════════════════════════════════
   SCROLLBAR
   ════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* ════════════════════════════════════════
   NESTED DETAILS (source chunks / raw meta)
   ════════════════════════════════════════ */
details.rag-details {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    margin-top: 8px;
}
details.rag-details > summary {
    color: #64748B;
    font-size: 12px;
    font-weight: 500;
    padding: 8px 14px;
    cursor: pointer;
    list-style: none;
    user-select: none;
    transition: color 0.15s;
}
details.rag-details > summary::marker { display: none; }
details.rag-details > summary::before { content: "▸  "; color: #CBD5E1; }
details.rag-details[open] > summary::before { content: "▾  "; color: #94A3B8; }
details.rag-details > summary:hover { color: #0F172A; }
details.rag-details .rag-body { padding: 4px 14px 14px; }

/* ════════════════════════════════════════
   KEYFRAMES
   ════════════════════════════════════════ */
@keyframes shimmer-sweep {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(250%); }
}
@keyframes pulse-blue {
    0%,100% { opacity: 0.45; transform: scale(0.82); }
    50%      { opacity: 1;    transform: scale(1.12); }
}
@keyframes pulse-green {
    0%,100% { opacity: 0.5; transform: scale(0.85); }
    50%      { opacity: 1;   transform: scale(1.0);  }
}
@keyframes spin {
    100% { transform: rotate(360deg); }
}
@keyframes typing-bounce {
    0%, 80%, 100% { transform: scale(0.55); opacity: 0.35; }
    40%           { transform: scale(1.0);  opacity: 1; }
}
</style>
""", unsafe_allow_html=True)


# ── SESSION ───────────────────────────────────────────────────────────────────

def init_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        st.session_state.total_cost = 0.0
        st.session_state.total_tokens = 0
        st.session_state.backend_ok = None
        st.session_state.last_health_check = 0.0
        st.session_state.last_metadata = None
        st.session_state.suggested_prompts = []
        st.session_state.doc_summary = ""
        st.session_state.doc_topics = []
        _load_history()


def _load_history() -> None:
    try:
        r = requests.get(f"{API_BASE}/api/sessions/{st.session_state.session_id}", timeout=5)
        if r.ok:
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"], "meta": None}
                for m in r.json()
            ]
    except requests.RequestException:
        pass


def check_backend() -> bool:
    now = time.time()
    if now - st.session_state.last_health_check > 30 or st.session_state.backend_ok is None:
        try:
            st.session_state.backend_ok = requests.get(f"{API_BASE}/health", timeout=3).ok
        except requests.RequestException:
            st.session_state.backend_ok = False
        st.session_state.last_health_check = now
    return st.session_state.backend_ok


def format_time(ms: int) -> str:
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms}ms"


# ── SIDEBAR HELPER ────────────────────────────────────────────────────────────

def _sidebar_section(label: str) -> None:
    st.sidebar.markdown(
        f"<p style='font-size:10px;font-weight:700;color:#475569;"
        f"text-transform:uppercase;letter-spacing:0.08em;margin:14px 0 6px'>{label}</p>",
        unsafe_allow_html=True,
    )


# ── RENDER HELPERS ────────────────────────────────────────────────────────────

def render_pipeline_animation(
    placeholder,
    stage: str,
    route: str = "",
    loop: int = 1,
    relevance: float = 0.0,
    message: str = "",
) -> None:

    # ── SVG icon library — 24×24 viewbox, stroke-based, clean ────────────
    def _icon(kind: str, clr: str) -> str:
        return {
            # Magnifying glass → Query
            "search": (
                f"<circle cx='10.5' cy='10.5' r='5.8' stroke='{clr}' stroke-width='2' fill='none'/>"
                f"<line x1='15.2' y1='15.2' x2='20' y2='20' stroke='{clr}'"
                " stroke-width='2.4' stroke-linecap='round'/>"
            ),
            # Branch fork → Router
            "fork": (
                f"<circle cx='5' cy='12' r='2.3' fill='{clr}'/>"
                f"<circle cx='19' cy='6.5' r='2.0' fill='{clr}' opacity='0.80'/>"
                f"<circle cx='19' cy='17.5' r='2.0' fill='{clr}' opacity='0.80'/>"
                f"<path d='M7.2 11.2 Q13 9 17 7.3' stroke='{clr}' stroke-width='1.6'"
                " fill='none' stroke-linecap='round'/>"
                f"<path d='M7.2 12.8 Q13 15 17 16.7' stroke='{clr}' stroke-width='1.6'"
                " fill='none' stroke-linecap='round'/>"
            ),
            # Document page → Retrieve (index)
            "document": (
                f"<path d='M6 2.5h9l5 5v14H6V2.5z' stroke='{clr}' stroke-width='1.9'"
                " fill='none' stroke-linejoin='round'/>"
                f"<path d='M15 2.5v5h5' stroke='{clr}' stroke-width='1.4' fill='none'/>"
                f"<line x1='9' y1='12.5' x2='16' y2='12.5' stroke='{clr}'"
                " stroke-width='1.5' stroke-linecap='round'/>"
                f"<line x1='9' y1='16' x2='13' y2='16' stroke='{clr}'"
                " stroke-width='1.5' stroke-linecap='round'/>"
            ),
            # Globe → Web search
            "globe": (
                f"<circle cx='12' cy='12' r='8.5' stroke='{clr}' stroke-width='1.9' fill='none'/>"
                f"<path d='M12 3.5 Q16.3 8 16.3 12 Q16.3 16 12 20.5"
                f" Q7.7 16 7.7 12 Q7.7 8 12 3.5z' stroke='{clr}' stroke-width='1.3' fill='none'/>"
                f"<line x1='3.5' y1='9' x2='20.5' y2='9' stroke='{clr}'"
                " stroke-width='1.3' stroke-linecap='round'/>"
                f"<line x1='3.5' y1='15' x2='20.5' y2='15' stroke='{clr}'"
                " stroke-width='1.3' stroke-linecap='round'/>"
            ),
            # 4-pointed star ✦ → LLM / Generate
            "spark": (
                f"<path d='M12 2.5L13.75 10.25L21.5 12L13.75 13.75L12 21.5"
                f"L10.25 13.75L2.5 12L10.25 10.25Z' fill='{clr}'/>"
            ),
            # Circle checkmark → Done
            "check": (
                f"<circle cx='12' cy='12' r='8.5' stroke='{clr}' stroke-width='1.9' fill='none'/>"
                f"<polyline points='7.5,12 10.8,15.5 16.5,8.5' stroke='{clr}'"
                " stroke-width='2.3' fill='none' stroke-linecap='round' stroke-linejoin='round'/>"
            ),
        }.get(kind, "")

    # ── Node list ─────────────────────────────────────────────────────────
    if route == "index":
        node_defs = [
            ("search",   "Query"),
            ("fork",     "Route"),
            ("document", "Retrieve"),
            ("spark",    "Generate"),
            ("check",    "Done"),
        ]
    elif route == "search":
        node_defs = [
            ("search", "Query"),
            ("fork",   "Route"),
            ("globe",  "Web"),
            ("spark",  "Generate"),
            ("check",  "Done"),
        ]
    else:
        node_defs = [
            ("search", "Query"),
            ("fork",   "Route"),
            ("spark",  "Generate"),
            ("check",  "Done"),
        ]

    in_middle = route in ("index", "search")
    stage_map = {
        "routing": 1, "routed": 1,
        "retrieving": 2, "retrieved": 2, "rewriting": 2,
        "searching_web": 2, "web_done": 2,
        "generating": 3 if in_middle else 2,
        "done": len(node_defs) - 1,
    }
    active = stage_map.get(stage, 1)
    prog   = int(active / max(len(node_defs) - 1, 1) * 100)

    default_msg = {
        "routing":       "Classifying your question...",
        "routed":        f"Route selected — {route or 'detecting'}",
        "retrieving":    f"Searching your documents (attempt {loop} of 2)...",
        "retrieved":     f"Chunks retrieved · {int(relevance * 100)}% relevance",
        "rewriting":     "Rewriting query for better results...",
        "searching_web": "Querying live web sources...",
        "web_done":      "Web results retrieved",
        "generating":    "Composing your answer...",
        "done":          "Complete",
    }
    status = html.escape(message or default_msg.get(stage, "Processing..."))

    # ── Build node cards ──────────────────────────────────────────────────
    nodes_html = ""
    for i, (kind, label) in enumerate(node_defs):
        is_active = i == active
        is_done   = i < active

        if is_active:
            bg      = "#EFF6FF"
            border  = "1.5px solid #2563EB"
            shadow  = "0 0 0 3px rgba(37,99,235,0.11),0 2px 8px rgba(37,99,235,0.13)"
            clr     = "#2563EB"
            lc, lw  = "#1D4ED8", "700"
            sweep   = (
                "<div style='position:absolute;inset:0;"
                "background:linear-gradient(90deg,transparent,"
                "rgba(37,99,235,0.07),transparent);"
                "animation:shimmer-sweep 1.6s ease-in-out infinite'></div>"
            )
        elif is_done:
            bg      = "#F0FDF4"
            border  = "1.5px solid #BBF7D0"
            shadow  = "none"
            clr     = "#16A34A"
            lc, lw  = "#15803D", "600"
            sweep   = ""
        else:
            bg      = "#F8FAFC"
            border  = "1px solid #E2E8F0"
            shadow  = "none"
            clr     = "#CBD5E1"
            lc, lw  = "#CBD5E1", "500"
            sweep   = ""

        nodes_html += (
            "<div style='display:flex;flex-direction:column;align-items:center;gap:5px'>"
            f"<div style='width:54px;height:50px;border-radius:12px;"
            f"background:{bg};border:{border};box-shadow:{shadow};"
            "display:flex;align-items:center;justify-content:center;"
            "position:relative;overflow:hidden'>"
            f"{sweep}"
            f"<svg width='20' height='20' viewBox='0 0 24 24' fill='none'"
            " style='position:relative;z-index:1'>"
            f"{_icon(kind, clr)}</svg>"
            "</div>"
            f"<span style='font-size:8.5px;color:{lc};font-weight:{lw};"
            "letter-spacing:0.04em;font-family:Inter,sans-serif'>"
            f"{label}</span>"
            "</div>"
        )

        if i < len(node_defs) - 1:
            fp = "100%" if is_done else ("55%" if is_active else "0%")
            fc = "#BBF7D0" if is_done else "#BFDBFE"
            nodes_html += (
                "<div style='display:flex;align-items:center;"
                "margin-top:-16px;width:20px;flex-shrink:0'>"
                "<div style='width:100%;height:2px;background:#E2E8F0;"
                "border-radius:1px;overflow:hidden'>"
                f"<div style='height:100%;width:{fp};background:{fc};"
                "border-radius:1px;"
                "transition:width .5s cubic-bezier(.4,0,.2,1)'></div>"
                "</div></div>"
            )

    # ── Attempt progress bar (replaces tiny dots) ─────────────────────────
    attempt_html = ""
    if route == "index" and stage in ("retrieving", "retrieved", "rewriting"):
        ap = 50 if loop == 1 else 100
        ac = "#BFDBFE" if loop == 1 else "#FCA5A5"
        attempt_html = (
            "<div style='display:flex;align-items:center;gap:10px;"
            "background:#F8FAFC;border:1px solid #F1F5F9;"
            "border-radius:8px;padding:6px 12px;margin-bottom:12px'>"
            f"<span style='color:#94A3B8;font-size:10px;font-weight:700;"
            "letter-spacing:0.05em;min-width:68px'>ATTEMPT {loop}/2</span>"
            "<div style='flex:1;background:#E2E8F0;height:3px;"
            "border-radius:999px;overflow:hidden'>"
            f"<div style='width:{ap}%;height:100%;background:{ac};"
            "border-radius:999px;transition:width .5s ease'></div></div>"
            "</div>"
        )

    # ── Document relevance bar ─────────────────────────────────────────────
    rel_html = ""
    if relevance > 0 and route == "index":
        rc = "#16A34A" if relevance >= 0.7 else "#D97706" if relevance >= 0.45 else "#EF4444"
        rl = "Strong match" if relevance >= 0.7 else "Moderate" if relevance >= 0.45 else "Weak match"
        rel_html = (
            "<div style='display:flex;align-items:center;gap:10px;margin-top:12px;"
            "background:#F8FAFC;border:1px solid #E2E8F0;"
            "border-radius:8px;padding:7px 12px'>"
            "<span style='color:#94A3B8;font-size:10px;font-weight:700;"
            "letter-spacing:0.05em;min-width:70px'>DOC MATCH</span>"
            "<div style='flex:1;background:#E2E8F0;height:3px;"
            "border-radius:999px;overflow:hidden'>"
            f"<div style='width:{int(relevance * 100)}%;height:100%;background:{rc};"
            "border-radius:999px'></div></div>"
            f"<span style='color:{rc};font-size:10px;font-weight:700;"
            f"min-width:88px;text-align:right'>{int(relevance * 100)}% · {rl}</span>"
            "</div>"
        )

    # ── Compose card ──────────────────────────────────────────────────────
    placeholder.markdown(
        # Card wrapper — position:relative for the gradient progress strip
        "<div style='background:#FFFFFF;border:1px solid #E2E8F0;"
        "border-radius:14px;padding:14px 18px 16px;margin-bottom:12px;"
        "box-shadow:0 1px 4px rgba(15,23,42,0.07);"
        "position:relative;overflow:hidden'>"

        # Gradient progress strip along top edge
        "<div style='position:absolute;top:0;left:0;right:0;height:2.5px;background:#F1F5F9'>"
        f"<div style='height:100%;width:{prog}%;"
        "background:linear-gradient(90deg,#2563EB,#7C3AED);"
        "transition:width .5s cubic-bezier(.4,0,.2,1)'></div></div>"

        # Header: pulsing dot + status message + step counter
        "<div style='display:flex;align-items:center;gap:10px;"
        "margin-bottom:14px;padding-top:4px'>"
        "<div style='width:7px;height:7px;border-radius:50%;background:#2563EB;"
        "flex-shrink:0;animation:pulse-blue 1.2s ease-in-out infinite'></div>"
        f"<span style='color:#334155;font-size:12px;font-weight:500;flex:1;"
        f"font-family:Inter,sans-serif'>{status}</span>"
        f"<span style='color:#94A3B8;font-size:10px;font-weight:600;"
        f"white-space:nowrap'>{active + 1} / {len(node_defs)}</span>"
        "</div>"

        # Attempt bar (only when looping on doc retrieval)
        f"{attempt_html}"

        # Node row
        f"<div style='display:flex;align-items:center;"
        f"justify-content:center;gap:0'>{nodes_html}</div>"

        # Relevance bar
        f"{rel_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def _grounding_breakdown_html(results: list) -> str:
    styles = {
        "GROUNDED":   {"icon": "✅", "color": "#059669", "bg": "#ECFDF5", "border": "#059669"},
        "INFERRED":   {"icon": "⚠️", "color": "#D97706", "bg": "#FFFBEB", "border": "#D97706"},
        "UNGROUNDED": {"icon": "❌", "color": "#DC2626", "bg": "#FEF2F2", "border": "#DC2626"},
    }
    rows = []
    for item in results:
        label = item.get("label", "INFERRED")
        sentence = html.escape(item.get("sentence", ""))
        s = styles.get(label, styles["INFERRED"])
        rows.append(
            f"<div style='background:{s['bg']};border-left:3px solid {s['border']};"
            "padding:7px 12px;margin-bottom:4px;border-radius:0 6px 6px 0;"
            f"font-size:12px;color:#374151;line-height:1.55'>{sentence}"
            f"<span style='color:{s['color']};font-size:10px;font-weight:600;margin-left:8px'>"
            f"{s['icon']} {label}</span></div>"
        )
    legend = (
        "<div style='font-size:10px;color:#94A3B8;margin-bottom:8px;line-height:1.8'>"
        "✅ Grounded — directly supported &nbsp;·&nbsp; "
        "⚠️ Inferred — implied &nbsp;·&nbsp; "
        "❌ Unsupported — no document evidence</div>"
    )
    return (
        "<details class='rag-details'><summary>See sentence breakdown</summary>"
        f"<div class='rag-body'>{legend}{''.join(rows)}</div></details>"
    )


def _process_citations(text: str, source_chunks: list) -> str:
    """Convert [N] citation markers in answer text to superscripts and append footnotes."""
    import re
    if not source_chunks:
        return text
    cited_html = re.sub(
        r'\[(\d+)\]',
        lambda m: f'<sup style="color:#2563EB;font-size:10px;font-weight:700">[{m.group(1)}]</sup>',
        text,
    )
    refs_used = sorted({int(m.group(1)) for m in re.finditer(r'\[(\d+)\]', text)})
    valid_refs = [n for n in refs_used if 1 <= n <= len(source_chunks)]
    if valid_refs:
        notes = " &nbsp;·&nbsp; ".join(
            f'<sup>[{n}]</sup> <span style="color:#64748B">{html.escape(source_chunks[n-1].get("filename","Doc"))}</span>'
            for n in valid_refs
        )
        cited_html += (
            "<div style='border-top:1px solid #F1F5F9;margin-top:14px;"
            "padding-top:8px;font-size:11px;color:#94A3B8'>"
            f"<span style='font-weight:700;letter-spacing:0.05em'>SOURCES</span>"
            f"&nbsp;&nbsp;{notes}</div>"
        )
    return cited_html


def render_source_chunks(chunks: list) -> None:
    if not chunks:
        return
    rows = []
    for idx, c in enumerate(chunks):
        if isinstance(c, dict):
            text = c.get("text", "")
            filename = c.get("filename", "")
        else:
            text, filename = str(c), ""
        rows.append(
            f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;"
            "border-left:3px solid #2563EB;border-radius:0 8px 8px 0;"
            "padding:10px 14px;margin-bottom:6px'>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
            f"<span style='font-size:10px;font-weight:700;color:#2563EB;"
            f"background:#EFF6FF;border:1px solid #BFDBFE;border-radius:4px;"
            f"padding:1px 6px'>[{idx + 1}]</span>"
            f"<span style='font-size:10px;font-weight:600;color:#64748B'>"
            f"{html.escape(filename)}</span></div>"
            "<div style='font-size:12px;color:#475569;line-height:1.7'>"
            f"{html.escape(text)}</div></div>"
        )
    count = len(rows)
    label = f"SOURCES ({count})"
    st.markdown(
        "<details style='margin-top:8px'>"
        "<summary style='cursor:pointer;display:inline-flex;align-items:center;gap:6px;"
        "font-size:10px;font-weight:700;color:#64748B;letter-spacing:0.08em;"
        "list-style:none;user-select:none;padding:4px 0'>"
        "<span style='font-size:9px'>▶</span>"
        f"<span>{label}</span>"
        "</summary>"
        "<div style='margin-top:6px'>"
        f"{''.join(rows)}</div></details>",
        unsafe_allow_html=True,
    )


def render_confidence_bar(grounding: dict) -> None:
    """Prominent colour bar shown below every document-sourced answer."""
    if not grounding or grounding.get("skipped") or not grounding.get("summary"):
        return
    summ = grounding["summary"]
    score = summ.get("trust_score", 0)
    level = summ.get("trust_level", "")
    pct = int(score * 100)
    cfg = {
        "HIGH":     ("#16A34A", "#F0FDF4", "#BBF7D0", "Well-supported by document"),
        "MODERATE": ("#D97706", "#FFFBEB", "#FDE68A", "Partially supported — some inference"),
        "LOW":      ("#EF4444", "#FEF2F2", "#FECACA", "Limited document support"),
    }.get(level, ("#94A3B8", "#F8FAFC", "#E2E8F0", "Unverified"))
    bar_color, bg, border, label = cfg
    st.markdown(
        f"<div style='background:{bg};border:1px solid {border};border-radius:8px;"
        "padding:9px 14px;margin:8px 0 4px;display:flex;align-items:center;gap:12px'>"
        "<div style='flex:1;background:#E2E8F0;height:5px;border-radius:999px;overflow:hidden'>"
        f"<div style='width:{pct}%;height:100%;background:{bar_color};"
        "border-radius:999px'></div></div>"
        f"<span style='color:{bar_color};font-size:12px;font-weight:700;"
        f"white-space:nowrap'>{pct}%</span>"
        f"<span style='color:#64748B;font-size:11px;white-space:nowrap'>{label}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_copy_button(answer_text: str) -> None:
    """Clipboard copy button rendered in an iframe so JS executes freely."""
    escaped = json.dumps(answer_text)
    components.html(
        f"""<div style="font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif;
                        padding:2px 0;display:flex;gap:6px">
        <button id="cb" onclick="navigator.clipboard.writeText({escaped})
            .then(function(){{
                var b=document.getElementById('cb');
                b.textContent='✓ Copied';
                b.style.background='#EFF6FF';
                b.style.borderColor='#93C5FD';
                b.style.color='#1D4ED8';
                setTimeout(function(){{
                    b.textContent='⧉ Copy answer';
                    b.style.background='';
                    b.style.borderColor='';
                    b.style.color='';
                }},1600);
            }})
            .catch(function(){{ document.getElementById('cb').textContent='Copy failed'; }});"
            style="background:#FFFFFF;border:1.5px solid #CBD5E1;border-radius:7px;
                   padding:5px 14px;font-size:12px;font-weight:600;color:#475569;
                   cursor:pointer;font-family:inherit;transition:all .15s ease;
                   display:inline-flex;align-items:center;gap:5px;
                   box-shadow:0 1px 3px rgba(15,23,42,0.06)"
            onmouseover="this.style.borderColor='#2563EB';this.style.color='#2563EB';this.style.background='#EFF6FF'"
            onmouseout="this.style.borderColor='#CBD5E1';this.style.color='#475569';this.style.background='#FFFFFF'">
            ⧉ Copy answer
        </button></div>""",
        height=36,
    )


def render_agent_panel(metadata: dict) -> None:
    route = metadata.get("route_taken", "general")
    loops = metadata.get("loops_executed", 0) or 0
    time_ms = metadata.get("processing_ms", 0)
    cost = metadata.get("estimated_cost_usd", 0)
    relevance_scores = metadata.get("relevance_scores") or []
    grounding = metadata.get("grounding") or {}
    answer_versions = metadata.get("answer_versions") or []
    answer_improvement = metadata.get("answer_improvement") or {}
    knowledge_gaps = metadata.get("knowledge_gaps")
    source_chunks = metadata.get("source_chunks") or []

    # Prominent confidence bar — visible immediately below the answer
    if st.session_state.get("enable_grounding", True):
        render_confidence_bar(grounding)

    route_cfg = {
        "index":   {"label": "Document",  "bg": "#EFF6FF", "color": "#1D4ED8", "border": "#BFDBFE", "icon": "📄"},
        "search":  {"label": "Web Search", "bg": "#F5F3FF", "color": "#6D28D9", "border": "#DDD6FE", "icon": "🌐"},
        "general": {"label": "General AI", "bg": "#FFFBEB", "color": "#92400E", "border": "#FDE68A", "icon": "⚡"},
    }
    rc = route_cfg.get(route, {"label": route, "bg": "#F8FAFC", "color": "#64748B", "border": "#E2E8F0", "icon": "⚡"})

    trust_html = ""
    if grounding and not grounding.get("skipped") and grounding.get("summary"):
        t_level = grounding["summary"].get("trust_level", "UNVERIFIED")
        t_score = grounding["summary"].get("trust_score", 0)
        trust_styles = {
            "HIGH":     {"bg": "#ECFDF5", "color": "#065F46", "border": "#A7F3D0"},
            "MODERATE": {"bg": "#FFFBEB", "color": "#92400E", "border": "#FDE68A"},
            "LOW":      {"bg": "#FEF2F2", "color": "#991B1B", "border": "#FECACA"},
        }
        ts = trust_styles.get(t_level, {"bg": "#F8FAFC", "color": "#64748B", "border": "#E2E8F0"})
        trust_html = (
            f"<span style='background:{ts['bg']};border:1px solid {ts['border']};color:{ts['color']};"
            f"font-size:11px;font-weight:600;padding:2px 10px;border-radius:999px'>"
            f"{t_level} · {int(t_score * 100)}%</span>"
        )

    loops_html = (
        "<span style='color:#CBD5E1;font-size:11px'>·</span>"
        f"<span style='color:#94A3B8;font-size:11px'>{loops} loop{'s' if loops != 1 else ''}</span>"
        if loops > 1 else ""
    )

    st.markdown(
        "<div style='display:flex;align-items:center;gap:8px;margin-top:5px;margin-bottom:3px;flex-wrap:wrap'>"
        f"<span style='background:{rc['bg']};border:1px solid {rc['border']};color:{rc['color']};"
        f"font-size:11px;font-weight:600;padding:3px 10px;border-radius:999px'>{rc['icon']} {rc['label']}</span>"
        f"<span style='background:#F8FAFC;border:1px solid #E2E8F0;color:#64748B;"
        f"font-size:11px;padding:3px 10px;border-radius:999px'>{format_time(time_ms)}</span>"
        f"{loops_html}{trust_html}</div>",
        unsafe_allow_html=True,
    )

    if route == "index" and source_chunks:
        render_source_chunks(source_chunks)

    with st.expander("View AI reasoning", expanded=False):
        # Retrieval quality bars
        if route == "index" and relevance_scores:
            bars = []
            for i, score in enumerate(relevance_scores):
                bc = "#059669" if score >= 0.7 else "#D97706" if score >= 0.45 else "#EF4444"
                lbl = "Strong" if score >= 0.7 else "Fair" if score >= 0.45 else "Weak"
                bars.append(
                    "<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>"
                    f"<div style='color:#94A3B8;font-size:11px;min-width:54px'>Chunk {i+1}</div>"
                    "<div style='flex:1;background:#E2E8F0;border-radius:999px;height:4px;overflow:hidden'>"
                    f"<div style='width:{int(score*100)}%;height:100%;background:{bc};border-radius:999px'></div></div>"
                    f"<div style='color:{bc};font-size:11px;font-weight:600;min-width:72px'>{score:.2f} · {lbl}</div></div>"
                )
            st.markdown(
                "<div style='color:#94A3B8;font-size:10px;text-transform:uppercase;"
                "letter-spacing:0.07em;margin-bottom:10px'>Retrieval Quality</div>"
                + "".join(bars) + "<div style='height:8px'></div>",
                unsafe_allow_html=True,
            )

        # Answer Evolution
        if st.session_state.get("enable_versioning", True) and len(answer_versions) > 1:
            improved = answer_improvement.get("improved", False)
            reason = html.escape(answer_improvement.get("reason", ""))
            status_line = (
                f"<div style='color:#059669;font-size:12px;margin-bottom:10px'>✓ Answer refined · {reason}</div>"
                if improved else
                f"<div style='color:#94A3B8;font-size:12px;margin-bottom:10px'>Answer consistent · {reason}</div>"
            )
            attempts = []
            for idx, v in enumerate(answer_versions):
                q = v.get("retrieval_quality", 0) or 0
                qc = "#059669" if q >= 0.7 else "#D97706" if q >= 0.45 else "#EF4444"
                rewrite = " · rewritten" if v.get("query_was_rewritten") else ""
                is_last = idx == len(answer_versions) - 1
                final_chip = (
                    "<span style='color:#2563EB;font-size:10px;margin-left:6px;font-weight:600'>FINAL</span>"
                    if is_last else ""
                )
                attempts.append(
                    f"<div style='border-left:2px solid {'#2563EB' if is_last else '#E2E8F0'};"
                    "padding-left:12px;margin-bottom:4px'>"
                    "<div style='display:flex;align-items:center;gap:8px'>"
                    f"<span style='color:#475569;font-size:11px;font-weight:600'>"
                    f"Attempt {v.get('loop_number', idx+1)}</span>"
                    f"<span style='color:{qc};font-size:11px'>{q:.2f}{rewrite}</span>{final_chip}</div>"
                    "<div style='color:#94A3B8;font-size:11px;font-family:monospace;margin-top:2px'>"
                    f"\"{html.escape(v.get('query_used') or '—')}\"</div></div>"
                )
                if not is_last:
                    attempts.append("<div style='color:#E2E8F0;padding-left:14px;font-size:12px'>↓</div>")
            st.markdown(
                "<div style='border-top:1px solid #F1F5F9;padding-top:12px;margin-bottom:10px'></div>"
                "<div style='color:#94A3B8;font-size:10px;text-transform:uppercase;"
                "letter-spacing:0.07em;margin-bottom:8px'>Answer Evolution</div>"
                + status_line + "".join(attempts),
                unsafe_allow_html=True,
            )

        # Answer Verification
        if st.session_state.get("enable_grounding", True) and grounding:
            hdr = (
                "<div style='border-top:1px solid #F1F5F9;padding-top:12px;"
                "margin-bottom:10px;margin-top:4px'></div>"
                "<div style='color:#94A3B8;font-size:10px;text-transform:uppercase;"
                "letter-spacing:0.07em;margin-bottom:10px'>Answer Verification</div>"
            )
            if grounding.get("skipped"):
                st.markdown(
                    hdr + f"<div style='color:#94A3B8;font-size:12px'>Skipped · {html.escape(grounding.get('reason',''))}</div>",
                    unsafe_allow_html=True,
                )
            elif grounding.get("summary"):
                summ = grounding["summary"]
                results = grounding.get("results") or []
                t_score = summ.get("trust_score", 0)
                t_level = summ.get("trust_level", "—")
                t_color = {"HIGH": "#059669", "MODERATE": "#D97706", "LOW": "#EF4444"}.get(t_level, "#94A3B8")
                body = (
                    "<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>"
                    "<div style='flex:1;background:#E2E8F0;border-radius:999px;height:5px;overflow:hidden'>"
                    f"<div style='width:{int(t_score*100)}%;height:100%;background:{t_color};"
                    "border-radius:999px'></div></div>"
                    f"<div style='color:{t_color};font-size:12px;font-weight:700;min-width:76px'>"
                    f"{int(t_score*100)}% trusted</div></div>"
                    "<div style='display:flex;gap:14px;margin-bottom:10px'>"
                    f"<span style='color:#059669;font-size:11px'>✅ {summ.get('grounded_count',0)} verified</span>"
                    f"<span style='color:#D97706;font-size:11px'>⚠️ {summ.get('inferred_count',0)} inferred</span>"
                    f"<span style='color:#EF4444;font-size:11px'>❌ {summ.get('ungrounded_count',0)} unsupported</span></div>"
                )
                if results:
                    body += _grounding_breakdown_html(results)
                st.markdown(hdr + body, unsafe_allow_html=True)

        # Raw metadata
        raw = html.escape(
            f"route: {route}\nloops: {loops}\nprocessing_ms: {time_ms}\n"
            f"estimated_cost_usd: {cost:.5f}\nrelevance_scores: {relevance_scores}"
        )
        st.markdown(
            "<details class='rag-details'><summary>Raw metadata</summary>"
            f"<div class='rag-body'><pre style='color:#64748B;font-size:11px;margin:0;"
            f"background:#F8FAFC;border-radius:6px;padding:8px'>{raw}</pre></div></details>",
            unsafe_allow_html=True,
        )

    if st.session_state.get("enable_gaps", True) and route == "index" and knowledge_gaps:
        render_knowledge_gap_card(knowledge_gaps)


def render_knowledge_gap_card(knowledge_gaps: dict) -> None:
    score = knowledge_gaps.get("completeness_score", 10)
    if score >= GAP_COMPLETENESS_THRESHOLD:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:6px;margin-top:8px'>"
            "<span style='color:#059669;font-size:12px;font-weight:500'>"
            "✓ Answer well-supported by your document</span></div>",
            unsafe_allow_html=True,
        )
        return

    missing = knowledge_gaps.get("missing_info") or []
    suggested = knowledge_gaps.get("suggested_documents") or []
    summary = html.escape(knowledge_gaps.get("gap_summary", ""))
    score_color = "#DC2626" if score <= 3 else "#D97706" if score <= 5 else "#D97706"

    missing_items = "".join(
        f"<div style='color:#64748B;font-size:12px;padding:2px 0;display:flex;gap:6px'>"
        f"<span style='color:#CBD5E1'>·</span>{html.escape(str(m))}</div>"
        for m in missing
    )
    suggested_items = "".join(
        f"<div style='color:#64748B;font-size:12px;padding:2px 0;display:flex;gap:6px'>"
        f"<span style='color:#2563EB'>→</span>{html.escape(str(d))}</div>"
        for d in suggested
    )
    mc = (
        "<div><div style='color:#94A3B8;font-size:10px;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.05em;margin-bottom:4px'>Missing</div>{missing_items}</div>"
        if missing_items else ""
    )
    sc = (
        "<div><div style='color:#94A3B8;font-size:10px;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.05em;margin-bottom:4px'>Try uploading</div>{suggested_items}</div>"
        if suggested_items else ""
    )
    grid = (
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>{mc}{sc}</div>"
        if (mc or sc) else ""
    )

    st.markdown(
        "<div style='background:#FFFBEB;border:1px solid rgba(217,119,6,0.3);border-radius:10px;"
        "padding:14px 16px;margin-top:10px;"
        "box-shadow:0 1px 3px rgba(15,23,42,0.04)'>"
        "<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px'>"
        "<span style='color:#92400E;font-size:13px;font-weight:600'>💡 Improve this answer</span>"
        f"<span style='color:{score_color};font-size:10px;font-weight:700;"
        f"background:rgba(220,38,38,0.08);border:1px solid rgba(220,38,38,0.2);"
        f"padding:2px 9px;border-radius:999px'>{score}/10 complete</span></div>"
        f"<div style='color:#78350F;font-size:12px;margin-bottom:10px;line-height:1.55'>{summary}</div>"
        f"{grid}</div>",
        unsafe_allow_html=True,
    )


def _friendly_error(raw: str) -> str:
    """Map raw API / HTTP error strings to clean user-facing messages."""
    r = raw.lower()
    if "429" in raw or "rate limit" in r or "rate_limit" in r or "quota" in r:
        return (
            "⏱ Rate limit reached — the AI model has a daily usage quota on the free tier. "
            "Wait 1–2 minutes and try again. (Groq free tier: 6 000 tokens/minute)"
        )
    if "401" in raw or "unauthorized" in r or "invalid api key" in r or "api_key" in r:
        return "🔑 Invalid API key — check your GROQ_API_KEY value in the .env file."
    if "context" in r and ("length" in r or "window" in r or "token" in r):
        return (
            "📄 Conversation too long for the model's context window. "
            "Click 'Clear conversation' and start fresh."
        )
    if "timeout" in r or "timed out" in r:
        return "⏳ The request timed out — the model took too long. Try a shorter question."
    if "connection" in r or "refused" in r or "cannot connect" in r or "unreachable" in r:
        return "🔌 Cannot reach the backend. Make sure it's running: python start.py"
    if "500" in raw or "internal server error" in r:
        return "⚡ The AI server encountered an internal error. Try again in a moment."
    return raw


def render_error_state(error_message: str) -> None:
    st.markdown(
        "<div style='background:linear-gradient(135deg,#FEF2F2,#FFF5F5);"
        "border:1px solid #FECACA;border-radius:12px;padding:20px 24px;margin-top:8px'>"
        "<h3 style='color:#DC2626;margin:0 0 6px;font-size:16px'>⚠ Something went wrong</h3>"
        f"<p style='color:#64748B;margin:0 0 8px;font-size:13px'>{html.escape(error_message)}</p>"
        "<p style='color:#94A3B8;margin:0;font-size:12px'>"
        "Try rephrasing your question, or check that your document uploaded correctly.</p></div>",
        unsafe_allow_html=True,
    )


# ── APP INIT ──────────────────────────────────────────────────────────────────
init_session()

# Ctrl+K → focus chat input (runs inside an iframe, accesses parent document)
components.html(
    """<script>
    (function() {
        var doc = window.parent.document;
        doc.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                var ta = doc.querySelector('[data-testid="stChatInputTextArea"]');
                if (ta) { ta.focus(); ta.select(); }
            }
        });
    })();
    </script>""",
    height=0,
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # Brand block
    st.markdown(
        "<div style='display:flex;align-items:center;gap:12px;padding:20px 0 16px'>"
        + _logo_html(40, 10) +
        "<div>"
        "<div style='font-size:15px;font-weight:800;color:#FFFFFF;letter-spacing:-0.3px'>"
        "Adaptive RAG AI</div>"
        "<div style='font-size:10px;color:#64748B;margin-top:2px'>"
        "Intelligent Document Intelligence</div>"
        "</div></div>"
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:0 0 4px'>",
        unsafe_allow_html=True,
    )

    # Documents
    _existing = {f["name"] for f in st.session_state.uploaded_files}
    _slots = MAX_UPLOADS - len(st.session_state.uploaded_files)
    _sidebar_section(f"YOUR DOCUMENTS ({len(st.session_state.uploaded_files)}/{MAX_UPLOADS})")

    if _slots > 0:
        uploaded_list = st.file_uploader(
            label="Upload documents",
            type=["pdf", "txt", "docx", "md", "csv"],
            accept_multiple_files=True,
            help=f"Up to {MAX_UPLOADS} docs · PDF, TXT, DOCX, MD, CSV · 10 MB max each",
            label_visibility="collapsed",
        )
        new_files = [f for f in (uploaded_list or []) if f.name not in _existing]
        for uf in new_files[:_slots]:
            _upload_ph = st.empty()
            _upload_ph.markdown(
                "<div style='display:flex;align-items:center;gap:10px;"
                "background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);"
                "border-radius:10px;padding:10px 14px;margin:4px 0'>"
                "<div style='width:16px;height:16px;border:2.5px solid rgba(37,99,235,0.25);"
                "border-top-color:#60A5FA;border-radius:50%;flex-shrink:0;"
                "animation:spin 0.75s linear infinite'></div>"
                f"<span style='font-size:12px;color:#94A3B8'>"
                f"Indexing {html.escape(uf.name)}…</span></div>",
                unsafe_allow_html=True,
            )
            try:
                r = requests.post(
                    f"{API_BASE}/api/upload",
                    files={"file": (uf.name, uf.getvalue())},
                    data={"session_id": st.session_state.session_id},
                    timeout=300,
                )
                if r.ok:
                    result = r.json()
                    st.session_state.uploaded_files.append({
                        "name": uf.name,
                        "chunks": result.get("child_count", 0),
                    })
                    try:
                        sug = requests.get(
                            f"{API_BASE}/api/suggestions/{st.session_state.session_id}",
                            timeout=15,
                        )
                        if sug.ok:
                            st.session_state.suggested_prompts = sug.json().get("prompts", [])
                    except requests.RequestException:
                        pass
                    try:
                        ins = requests.get(
                            f"{API_BASE}/api/insight/{st.session_state.session_id}",
                            timeout=20,
                        )
                        if ins.ok:
                            ins_data = ins.json()
                            st.session_state.doc_summary = ins_data.get("summary", "")
                            st.session_state.doc_topics = ins_data.get("topics", [])
                    except requests.RequestException:
                        pass
                else:
                    try:
                        err_detail = r.json().get("detail", r.text)
                    except Exception:
                        err_detail = r.text[:300] if r.text else f"HTTP {r.status_code} — server returned no body"
                    st.error(f"Upload failed ({uf.name}): {err_detail}")
            except Exception as e:
                st.error(f"Upload error: {_friendly_error(str(e))}")
            finally:
                _upload_ph.empty()
    else:
        st.markdown(
            f"<div style='color:#475569;font-size:12px;padding:4px 0 8px'>"
            f"Maximum of {MAX_UPLOADS} documents reached.</div>",
            unsafe_allow_html=True,
        )

    # File chips
    FILE_ICONS = {".pdf": "📄", ".txt": "📝", ".docx": "📋", ".md": "#️⃣", ".csv": "📊"}
    for finfo in st.session_state.uploaded_files:
        ext = ("." + finfo["name"].rsplit(".", 1)[-1].lower()) if "." in finfo["name"] else ""
        icon = FILE_ICONS.get(ext, "📄")
        chunk_note = f"{finfo['chunks']} chunks" if finfo.get("chunks") else "Indexed"
        st.markdown(
            "<div style='display:flex;align-items:center;gap:10px;"
            "background:rgba(255,255,255,0.05);"
            "border:1px solid rgba(255,255,255,0.09);"
            "border-radius:8px;padding:8px 12px;margin:4px 0'>"
            f"<span style='font-size:16px'>{icon}</span>"
            "<div style='flex:1;min-width:0'>"
            f"<div style='font-size:12px;font-weight:600;color:#F1F5F9;"
            f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{html.escape(finfo['name'])}</div>"
            f"<div style='font-size:10px;color:#64748B;margin-top:1px'>{chunk_note}</div>"
            "</div>"
            "<span style='font-size:10px;background:rgba(5,150,105,0.18);"
            "color:#34D399;border:1px solid rgba(5,150,105,0.28);"
            "border-radius:10px;padding:2px 8px'>✓</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    if not st.session_state.uploaded_files:
        st.markdown(
            "<div style='color:#475569;font-size:12px;padding:6px 0 4px;line-height:1.5'>"
            "No documents yet — questions will use web search or general knowledge.</div>",
            unsafe_allow_html=True,
        )

    # Transparency
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:14px 0 4px'>",
        unsafe_allow_html=True,
    )
    _sidebar_section("TRANSPARENCY")
    st.toggle("Grounding Check", value=True, key="enable_grounding",
              help="Verify each sentence against your document.")
    st.toggle("Answer Evolution", value=True, key="enable_versioning",
              help="Show how the answer improved across retrieval attempts.")
    st.toggle("Knowledge Gap Alerts", value=True, key="enable_gaps",
              help="Flag when the document lacks enough info and suggest uploads.")

    # Session stats
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:14px 0 4px'>",
        unsafe_allow_html=True,
    )
    _sidebar_section("SESSION")

    if st.session_state.get("last_metadata"):
        meta = st.session_state.last_metadata
        route_val = meta.get("route_taken", "—")
        route_label = {"index": "📄 Doc", "search": "🌐 Web", "general": "⚡ AI"}.get(route_val, route_val)
        route_color = {"index": "#93C5FD", "search": "#C4B5FD", "general": "#6EE7B7"}.get(route_val, "#94A3B8")

        grounding_m = meta.get("grounding") or {}
        trust_val, trust_color = "—", "#94A3B8"
        if not grounding_m.get("skipped") and grounding_m.get("summary"):
            t_lv = grounding_m["summary"].get("trust_level", "—")
            trust_val = t_lv
            trust_color = {"HIGH": "#6EE7B7", "MODERATE": "#FDE68A", "LOW": "#FCA5A5"}.get(t_lv, "#94A3B8")

        def _stat(label, color, value):
            return (
                "<div style='background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.14);"
                "border-radius:8px;padding:9px 11px'>"
                f"<div style='font-size:9px;font-weight:700;color:#64748B;"
                f"text-transform:uppercase;letter-spacing:0.07em'>{label}</div>"
                f"<div style='font-size:13px;font-weight:800;color:{color};margin-top:3px'>{value}</div></div>"
            )

        cards = (
            _stat("ROUTE", route_color, route_label)
            + _stat("TIME", "#FFFFFF", format_time(meta.get("processing_ms", 0)))
            + _stat("LOOPS", "#FFFFFF", str(meta.get("loops_executed", 0)))
            + _stat("COST", "#FFFFFF", f"${meta.get('estimated_cost_usd', 0):.4f}")
            + _stat("TRUST", trust_color, trust_val)
            + _stat("TOKENS", "#FFFFFF", f"{(meta.get('token_usage') or {}).get('completion', 0):,}")
        )
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:8px 0'>{cards}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='color:#475569;font-size:11px;margin-bottom:10px'>"
        f"Total: <span style='color:#94A3B8'>${st.session_state.total_cost:.4f}</span>"
        f" · <span style='color:#94A3B8'>{st.session_state.total_tokens:,} tokens</span></div>",
        unsafe_allow_html=True,
    )

    if st.button("Clear conversation", use_container_width=True):
        msgs = st.session_state.messages
        total_q = sum(1 for m in msgs if m["role"] == "user")
        doc_q   = sum(1 for m in msgs if m["role"] == "assistant"
                      and (m.get("meta") or {}).get("route_taken") == "index")
        web_q   = sum(1 for m in msgs if m["role"] == "assistant"
                      and (m.get("meta") or {}).get("route_taken") == "search")
        cost    = st.session_state.total_cost
        if total_q:
            st.toast(
                f"Session: {total_q} question{'s' if total_q != 1 else ''} · "
                f"{doc_q} from document · {web_q} from web · ${cost:.4f} spent",
                icon="📊",
            )
        st.session_state.messages = []
        st.session_state.last_metadata = None
        st.session_state.total_cost = 0.0
        st.session_state.total_tokens = 0
        st.rerun()

    # Export conversation
    if st.session_state.messages:
        _lines = [
            "Adaptive RAG AI — Conversation Export",
            f"Session: {st.session_state.session_id}",
            "=" * 60, "",
        ]
        _cur_q = None
        for _m in st.session_state.messages:
            if _m["role"] == "user":
                _cur_q = _m["content"]
            elif _m["role"] == "assistant" and _cur_q is not None:
                _lines.append(f"Q: {_cur_q}")
                _lines.append("")
                _lines.append(f"A: {_m['content']}")
                _met = _m.get("meta") or {}
                if _met:
                    _lines.append(
                        f"   [{_met.get('route_taken','—')} · "
                        f"{format_time(_met.get('processing_ms',0))} · "
                        f"${_met.get('estimated_cost_usd',0):.4f}]"
                    )
                _lines += ["", "-" * 60, ""]
                _cur_q = None
        _lines += [
            f"Total cost: ${st.session_state.total_cost:.4f}",
            f"Total tokens: {st.session_state.total_tokens:,}",
        ]
        st.download_button(
            label="⬇ Export conversation",
            data="\n".join(_lines),
            file_name=f"adaptive_rag_{st.session_state.session_id[:8]}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ── Query history ─────────────────────────────────────────────────
    _sidebar_section("QUERY HISTORY")
    _route_icon = {"index": "📄", "search": "🌐", "general": "⚡"}
    _qh_msgs = st.session_state.messages
    _qh_pairs = []
    _i = 0
    while _i < len(_qh_msgs) - 1:
        if _qh_msgs[_i]["role"] == "user":
            _q = _qh_msgs[_i]["content"]
            _a_meta = (_qh_msgs[_i + 1].get("meta") or {}) if _qh_msgs[_i + 1]["role"] == "assistant" else {}
            _rt = _a_meta.get("route_taken", "general")
            _qh_pairs.append((_q, _rt))
            _i += 2
        else:
            _i += 1

    if _qh_pairs:
        for _qi, (_q, _rt) in enumerate(reversed(_qh_pairs[-6:])):
            _ico = _route_icon.get(_rt, "⚡")
            _label = (_q[:34] + "…") if len(_q) > 34 else _q
            if st.button(f"{_ico} {_label}", key=f"qhist_{_qi}", use_container_width=True):
                st.session_state.pending_prompt = _q
                st.rerun()
    else:
        st.markdown(
            "<div style='font-size:11px;color:#475569;padding:4px 0 8px'>"
            "No queries yet.</div>",
            unsafe_allow_html=True,
        )

    # Backend status
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if check_backend():
        st.markdown(
            "<span style='display:inline-flex;align-items:center;gap:6px;"
            "font-size:11px;font-weight:600;"
            "background:rgba(5,150,105,0.15);color:#34D399;"
            "border:1px solid rgba(5,150,105,0.25);"
            "border-radius:20px;padding:4px 12px'>"
            "<span style='width:6px;height:6px;border-radius:50%;"
            "background:#34D399;display:inline-block'></span>"
            "Backend connected</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span style='display:inline-flex;align-items:center;gap:6px;"
            "font-size:11px;font-weight:600;"
            "background:rgba(220,38,38,0.15);color:#F87171;"
            "border:1px solid rgba(220,38,38,0.25);"
            "border-radius:20px;padding:4px 12px'>"
            "<span style='width:6px;height:6px;border-radius:50%;"
            "background:#F87171;display:inline-block'></span>"
            "Backend offline</span>",
            unsafe_allow_html=True,
        )


# ── MAIN AREA ─────────────────────────────────────────────────────────────────

# Hero banner — dark gradient card sitting on the white page (FinGuard style)
_HERO_BADGES = ["Groq LLaMA 3.3", "Adaptive RAG", "Qdrant Vector DB", "FastAPI", "Multi-Document"]
_badge_pills = "".join(
    f"<span style='background:rgba(37,99,235,0.28);border:1px solid rgba(147,197,253,0.40);"
    f"border-radius:999px;color:#BAE6FD;font-size:12px;font-weight:600;padding:5px 15px;"
    f"font-family:Inter,sans-serif;white-space:nowrap;letter-spacing:0.01em'>{b}</span>"
    for b in _HERO_BADGES
)
st.markdown(
    "<div style='background:linear-gradient(135deg,#0F172A 0%,#0F3460 50%,#1E40AF 100%);"
    "border-radius:16px;padding:2.25rem 2.5rem;margin-bottom:1.5rem;"
    "position:relative;overflow:hidden;"
    "box-shadow:0 4px 24px rgba(15,23,42,0.18)'>"
    # 3px gradient top border — blue → purple → cyan
    "<div style='position:absolute;top:0;left:0;right:0;height:3px;"
    "background:linear-gradient(90deg,#2563EB 0%,#7C3AED 50%,#06B6D4 100%)'></div>"
    # radial glow
    "<div style='position:absolute;inset:0;"
    "background:radial-gradient(ellipse at 75% 50%,rgba(37,99,235,0.30) 0%,transparent 68%);"
    "pointer-events:none'></div>"
    # content
    "<div style='position:relative;z-index:1'>"
    # title row
    "<div style='display:flex;align-items:center;gap:14px;margin-bottom:10px'>"
    + _logo_html(52, 14) +
    "<div>"
    "<div style='margin:0;font-size:26px;font-weight:800;color:#FFFFFF;letter-spacing:-0.5px;"
    "font-family:Inter,sans-serif;line-height:1.15'>"
    "Adaptive <span style='color:#93C5FD'>RAG AI</span></div>"
    "<div style='font-size:12px;color:#64748B;margin-top:3px;font-family:Inter,sans-serif'>"
    "Powered by Groq · LLaMA 3.3 · Qdrant</div>"
    "</div></div>"
    # subtitle
    "<p style='color:#CBD5E1;font-size:14px;line-height:1.7;margin:0 0 18px;"
    "max-width:580px;font-family:Inter,sans-serif'>"
    "Ask anything about your documents — grounded answers, real sources, zero hallucinations.</p>"
    # badges
    f"<div style='display:flex;flex-wrap:wrap;gap:8px'>{_badge_pills}</div>"
    "</div></div>",
    unsafe_allow_html=True,
)

# Backend offline banner
if not check_backend():
    st.markdown(
        "<div style='background:linear-gradient(135deg,#FEF2F2,#FFF5F5);"
        "border:1px solid #FECACA;border-radius:12px;padding:20px 24px;margin-bottom:20px'>"
        "<h3 style='color:#DC2626;margin:0 0 6px;font-size:16px'>⚠ Backend Offline</h3>"
        "<p style='color:#64748B;margin:0 0 10px;font-size:13px'>"
        "The backend server is not running. Start it with:</p>"
        "<code style='background:#F1F5F9;border:1px solid #E2E8F0;border-radius:6px;"
        "padding:8px 14px;display:block;font-size:12px;color:#0F172A;margin:4px 0'>"
        "cd adaptive_rag &amp;&amp; python start.py</code>"
        "</div>",
        unsafe_allow_html=True,
    )

# Empty states
if not st.session_state.messages:
    if not st.session_state.uploaded_files:
        # Section header
        st.markdown(
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:16px'>"
            "<div style='width:4px;height:22px;background:#2563EB;border-radius:2px'></div>"
            "<h2 style='margin:0;font-size:18px;font-weight:700;color:#0F172A'>"
            "Get started</h2></div>",
            unsafe_allow_html=True,
        )
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.markdown(
                # Card wrapper
                "<div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;"
                "padding:1.75rem 1.75rem 1.5rem;"
                "box-shadow:0 1px 6px rgba(15,23,42,0.07)'>"
                # Icon + heading block (centered)
                "<div style='text-align:center;margin-bottom:14px'>"
                "<div style='display:inline-flex;align-items:center;justify-content:center;"
                "width:56px;height:56px;border-radius:14px;"
                "background:linear-gradient(135deg,#EFF6FF,#DBEAFE);margin-bottom:10px'>"
                "<svg width='28' height='28' viewBox='0 0 24 24' fill='none'"
                " xmlns='http://www.w3.org/2000/svg'>"
                "<path d='M12 16V4M12 4L8 8M12 4L16 8' stroke='#2563EB'"
                " stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/>"
                "<path d='M3 17v1a3 3 0 003 3h12a3 3 0 003-3v-1' stroke='#2563EB'"
                " stroke-width='2' stroke-linecap='round'/>"
                "</svg></div>"
                "<div style='font-size:15px;font-weight:700;color:#2563EB;margin-bottom:5px'>"
                "Upload a document in the sidebar</div>"
                "<div style='font-size:11px;color:#94A3B8;letter-spacing:0.05em'>"
                "PDF · TXT · DOCX · MD · CSV · up to 5 files</div>"
                "</div>"
                # Divider
                "<div style='border-top:1px solid #F1F5F9;margin:14px 0 12px'></div>"
                # Use-cases section label
                "<div style='font-size:10px;font-weight:700;color:#94A3B8;"
                "text-transform:uppercase;letter-spacing:0.09em;margin-bottom:10px'>Use cases</div>"
                # Left-aligned list
                "<div style='display:flex;flex-direction:column;gap:8px'>"
                "<div style='display:flex;align-items:center;gap:10px'>"
                "<span style='font-size:15px;flex-shrink:0'>📄</span>"
                "<span style='font-size:13px;color:#475569'>Analyze a research paper or report</span></div>"
                "<div style='display:flex;align-items:center;gap:10px'>"
                "<span style='font-size:15px;flex-shrink:0'>📋</span>"
                "<span style='font-size:13px;color:#475569'>Review a contract or policy document</span></div>"
                "<div style='display:flex;align-items:center;gap:10px'>"
                "<span style='font-size:15px;flex-shrink:0'>👤</span>"
                "<span style='font-size:13px;color:#475569'>Deep-dive into a resume or portfolio</span></div>"
                "<div style='display:flex;align-items:center;gap:10px'>"
                "<span style='font-size:15px;flex-shrink:0'>📊</span>"
                "<span style='font-size:13px;color:#475569'>Summarize a CSV data report</span></div>"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            # Hint callout — softer left border
            st.markdown(
                "<div style='background:#EFF6FF;border-left:3px solid #BFDBFE;"
                "border-radius:0 8px 8px 0;padding:11px 15px;margin-top:12px'>"
                "<span style='font-size:13px;color:#3B82F6;font-weight:500'>"
                "💬 Or ask a general question below — I'll search the web or use my own knowledge."
                "</span></div>",
                unsafe_allow_html=True,
            )
    else:
        docs_str = ", ".join(f["name"] for f in st.session_state.uploaded_files)
        latest = st.session_state.uploaded_files[-1] if st.session_state.uploaded_files else {}
        suggested = st.session_state.get("suggested_prompts") or [
            "Summarize this document's key points",
            "What are the main topics covered?",
            "What conclusions does this document reach?",
            "What information is missing from this document?",
        ]
        st.markdown(
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:16px'>"
            "<div style='width:4px;height:22px;background:#2563EB;border-radius:2px'></div>"
            f"<h2 style='margin:0;font-size:18px;font-weight:700;color:#0F172A'>"
            "Document ready — ask anything</h2></div>",
            unsafe_allow_html=True,
        )

        # ── Auto-generated summary card — always shown after upload ──────
        doc_summary = st.session_state.get("doc_summary", "")
        chunk_count = latest.get("chunks", 0)
        doc_name = html.escape(latest.get("name", ""))
        _summary_body = (
            f"<div style='font-size:13px;color:#334155;line-height:1.65'>"
            f"{html.escape(doc_summary)}</div>"
            if doc_summary else
            "<div style='font-size:13px;color:#94A3B8;font-style:italic'>"
            "Summary not available — the AI is currently rate-limited. "
            "Ask a question and it will analyse the document for you.</div>"
        )
        st.markdown(
            "<div style='background:#FFFFFF;border:1px solid #E2E8F0;"
            "border-radius:12px;padding:16px 20px;margin-bottom:14px;"
            "box-shadow:0 1px 4px rgba(15,23,42,0.06)'>"
            "<div style='display:flex;align-items:flex-start;gap:12px'>"
            "<div style='width:38px;height:38px;border-radius:8px;"
            "background:linear-gradient(135deg,#EFF6FF,#DBEAFE);"
            "display:flex;align-items:center;justify-content:center;flex-shrink:0'>"
            "<svg width='20' height='20' viewBox='0 0 24 24' fill='none'>"
            "<path d='M6 2.5h9l5 5v14H6V2.5z' stroke='#2563EB' stroke-width='1.9'"
            " fill='none' stroke-linejoin='round'/>"
            "<path d='M15 2.5v5h5' stroke='#2563EB' stroke-width='1.4' fill='none'/>"
            "<line x1='9' y1='12.5' x2='16' y2='12.5' stroke='#2563EB'"
            " stroke-width='1.5' stroke-linecap='round'/>"
            "<line x1='9' y1='16' x2='13' y2='16' stroke='#2563EB'"
            " stroke-width='1.5' stroke-linecap='round'/>"
            "</svg></div>"
            "<div style='flex:1'>"
            "<div style='font-size:10px;font-weight:700;color:#94A3B8;"
            "letter-spacing:0.08em;margin-bottom:5px'>DOCUMENT SUMMARY</div>"
            f"{_summary_body}"
            f"<div style='font-size:11px;color:#94A3B8;margin-top:7px'>"
            f"{doc_name} · {chunk_count} chunks indexed"
            "</div></div></div></div>",
            unsafe_allow_html=True,
        )

        # ── Key topic chips ───────────────────────────────────────────────
        doc_topics = st.session_state.get("doc_topics") or []
        _fallback_topics = ["Summarize document", "Key findings", "Main conclusions", "What's missing?"]
        _display_topics = doc_topics if doc_topics else _fallback_topics
        st.markdown(
            "<div style='font-size:10px;font-weight:700;color:#94A3B8;"
            "letter-spacing:0.08em;margin-bottom:8px'>"
            + ("KEY TOPICS — click to explore" if doc_topics else "QUICK QUESTIONS") +
            "</div>",
            unsafe_allow_html=True,
        )
        topic_cols = st.columns(min(len(_display_topics), 4))
        for ti, topic in enumerate(_display_topics):
            with topic_cols[ti % 4]:
                _prompt = (
                    f"Explain {topic} as described in the document"
                    if doc_topics else topic
                )
                if st.button(topic, key=f"topic_{ti}", use_container_width=True):
                    st.session_state.pending_prompt = _prompt
                    st.rerun()
        st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

        # ── Compare documents button (shown when 2+ docs are loaded) ─────
        if len(st.session_state.uploaded_files) >= 2:
            _doc_names = " and ".join(f["name"] for f in st.session_state.uploaded_files[:2])
            st.markdown(
                "<div style='display:flex;align-items:center;gap:10px;"
                "background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;"
                "padding:12px 16px;margin-bottom:14px;"
                "box-shadow:0 1px 4px rgba(15,23,42,0.05)'>"
                "<span style='font-size:18px'>🔀</span>"
                "<div style='flex:1'>"
                "<div style='font-size:12px;font-weight:700;color:#0F172A'>"
                "Compare your documents</div>"
                "<div style='font-size:11px;color:#94A3B8;margin-top:1px'>"
                "Ask how your uploaded documents relate or differ</div>"
                "</div></div>",
                unsafe_allow_html=True,
            )
            _cmp_q = f"Compare {_doc_names}: how do they differ in approach and conclusions?"
            if st.button("→ Compare documents", key="compare_docs", use_container_width=False):
                st.session_state.pending_prompt = _cmp_q
                st.rerun()

        # ── Suggested questions ───────────────────────────────────────────
        st.markdown(
            "<div style='font-size:10px;font-weight:700;color:#94A3B8;"
            "letter-spacing:0.08em;margin-bottom:8px'>SUGGESTED QUESTIONS</div>",
            unsafe_allow_html=True,
        )
        _, center, _ = st.columns([1, 3, 1])
        with center:
            cols = st.columns(2)
            for i, prompt_text in enumerate(suggested[:4]):
                with cols[i % 2]:
                    if st.button(prompt_text, key=f"sug_{i}", use_container_width=True):
                        st.session_state.pending_prompt = prompt_text
                        st.rerun()

# Chat history
for msg_idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    with st.chat_message(role, avatar="👤" if role == "user" else _LOGO_PIL):
        _content = message["content"]
        _hist_meta = message.get("meta") or {}
        _hist_src = _hist_meta.get("source_chunks") or []
        if role == "assistant" and _hist_src and _hist_meta.get("route_taken") == "index":
            st.markdown(_process_citations(_content, _hist_src), unsafe_allow_html=True)
        else:
            st.markdown(_content)
        if role == "assistant":
            if message.get("meta"):
                render_agent_panel(message["meta"])
            render_copy_button(message["content"])
            # Regenerate button — re-runs the preceding user question
            if msg_idx > 0 and st.session_state.messages[msg_idx - 1]["role"] == "user":
                _regen_q = st.session_state.messages[msg_idx - 1]["content"]
                if st.button("↺  Regenerate", key=f"regen_{msg_idx}",
                             help="Re-run this question for a fresh answer"):
                    st.session_state.messages = st.session_state.messages[:msg_idx - 1]
                    st.session_state.pending_prompt = _regen_q
                    st.rerun()

# Voice input mic button — injected into parent document body so position:fixed works
# (components.html iframes have height:0, making fixed children invisible)
components.html(
    """<script>
(function() {
    var pd = window.parent.document;
    if (pd.getElementById('micBtn')) return;  // already injected this session

    var btn = pd.createElement('button');
    btn.id = 'micBtn';
    btn.title = 'Voice input (Chrome/Edge)';
    btn.textContent = '🎤';
    btn.style.cssText = [
        'position:fixed','bottom:14px','right:58px','z-index:2147483647',
        'width:36px','height:36px','border-radius:50%',
        'background:rgba(255,255,255,0.08)','border:1.5px solid rgba(255,255,255,0.18)',
        'box-shadow:none',
        'cursor:pointer','font-size:16px',
        'display:flex','align-items:center','justify-content:center',
        'transition:all .15s ease','outline:none','backdrop-filter:blur(4px)'
    ].join(';');

    var listening = false;
    btn.addEventListener('click', function() {
        var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) { alert('Voice input requires Chrome or Edge'); return; }
        if (listening) return;
        var r = new SR(); r.lang = 'en-US'; r.interimResults = false;
        listening = true;
        btn.textContent = '🔴';
        btn.style.borderColor = '#EF4444';
        btn.style.background = 'rgba(239,68,68,0.15)';
        btn.style.boxShadow = '0 0 0 2px rgba(239,68,68,0.35)';
        r.start();
        r.onresult = function(e) {
            var text = e.results[0][0].transcript;
            var ta = pd.querySelector('[data-testid="stChatInputTextArea"]');
            if (ta) {
                Object.getOwnPropertyDescriptor(
                    window.parent.HTMLTextAreaElement.prototype, 'value'
                ).set.call(ta, text);
                ta.dispatchEvent(new Event('input', {bubbles: true}));
                ta.focus();
            }
            listening = false;
            btn.textContent = '🎤';
            btn.style.borderColor = 'rgba(255,255,255,0.18)';
            btn.style.background = 'rgba(255,255,255,0.08)';
            btn.style.boxShadow = 'none';
        };
        r.onerror = function() {
            listening = false;
            btn.textContent = '🎤';
            btn.style.borderColor = 'rgba(255,255,255,0.18)';
            btn.style.background = 'rgba(255,255,255,0.08)';
            btn.style.boxShadow = 'none';
        };
    });
    pd.body.appendChild(btn);
})();
</script>""",
    height=0,
)

# Chat input + SSE streaming handler
user_input = st.chat_input(
    placeholder="Ask anything about your documents, or any general question..."
)
pending = st.session_state.pop("pending_prompt", None)
if pending and not user_input:
    user_input = pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "meta": None})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=_LOGO_PIL):
        pipeline_ph = st.empty()
        answer_ph = st.empty()

        stage, route, loop, relevance = "routing", "", 1, 0.0
        full_answer = ""
        data = None
        stream_error = None

        render_pipeline_animation(pipeline_ph, "routing")

        # Build conversation history from all prior turns (exclude the current
        prior_turns = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m.get("content")
        ][-40:]

        payload = {
            "question": user_input,
            "session_id": st.session_state.session_id,
            "doc_available": bool(st.session_state.uploaded_files),
            "doc_filename": ", ".join(f["name"] for f in st.session_state.uploaded_files),
            "conversation_history": prior_turns,
        }

        try:
            with requests.post(
                f"{API_BASE}/api/chat/stream", json=payload, stream=True, timeout=180
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith(b"data: "):
                        continue
                    try:
                        event = json.loads(line[6:].decode("utf-8"))
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "")
                    if etype == "stage":
                        stage = event.get("stage", stage)
                        route = event.get("route", route) or route
                        loop = event.get("loop", loop)
                        relevance = event.get("relevance", relevance)
                        render_pipeline_animation(
                            pipeline_ph, stage=stage, route=route,
                            loop=loop, relevance=relevance, message=event.get("message", ""),
                        )
                        # Typing dots appear as soon as LLM starts generating
                        if stage == "generating" and not full_answer:
                            answer_ph.markdown(
                                "<div style='display:flex;align-items:center;gap:5px;padding:10px 0'>"
                                "<span style='width:7px;height:7px;border-radius:50%;background:#CBD5E1;"
                                "display:inline-block;animation:typing-bounce 1.2s ease-in-out infinite;"
                                "animation-delay:0s'></span>"
                                "<span style='width:7px;height:7px;border-radius:50%;background:#CBD5E1;"
                                "display:inline-block;animation:typing-bounce 1.2s ease-in-out infinite;"
                                "animation-delay:0.2s'></span>"
                                "<span style='width:7px;height:7px;border-radius:50%;background:#CBD5E1;"
                                "display:inline-block;animation:typing-bounce 1.2s ease-in-out infinite;"
                                "animation-delay:0.4s'></span>"
                                "</div>",
                                unsafe_allow_html=True,
                            )
                    elif etype == "token":
                        full_answer += event.get("content", "")
                        answer_ph.markdown(full_answer + "▌")
                    elif etype == "metadata":
                        data = event
                    elif etype == "error":
                        stream_error = _friendly_error(
                            event.get("message", "Unknown backend error")
                        )
                        break
        except requests.RequestException as exc:
            stream_error = _friendly_error(str(exc))

        # Show the completed pipeline state briefly before it disappears on rerun
        if not stream_error and data is not None:
            render_pipeline_animation(pipeline_ph, "done", route=route,
                                      message="Complete")
        else:
            pipeline_ph.empty()

        if stream_error or data is None:
            answer_ph.empty()
            render_error_state(stream_error or "No response received from the backend.")
            # Remove the user message we just appended — without an answer it
            # would leave a permanent empty gap in the chat history on next render.
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()
        else:
            final_answer = data.get("answer", full_answer)
            _src_chunks = data.get("source_chunks") or []
            # Render citations as superscripts if LLM used [N] markers
            if _src_chunks and data.get("route_taken") == "index":
                answer_ph.markdown(_process_citations(final_answer, _src_chunks),
                                   unsafe_allow_html=True)
            else:
                answer_ph.markdown(final_answer)
            render_agent_panel(data)
            render_copy_button(final_answer)

            usage = data.get("token_usage", {})
            st.session_state.total_cost += data.get("estimated_cost_usd", 0.0)
            st.session_state.total_tokens += usage.get("prompt", 0) + usage.get("completion", 0)
            st.session_state.last_metadata = data
            st.session_state.messages.append(
                {"role": "assistant", "content": data.get("answer", full_answer), "meta": data}
            )
            st.rerun()
