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
    """3-orbit atom mark — three elliptical orbits (doc/web/general routes), electrons at tips."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Deep dark background
        r_bg = size // 5
        bg = (6, 8, 28)
        try:
            draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=r_bg, fill=bg)
        except (AttributeError, TypeError):
            draw.rectangle([0, 0, size, size], fill=bg)
            for bx, by in [(r_bg,r_bg),(size-r_bg,r_bg),(r_bg,size-r_bg),(size-r_bg,size-r_bg)]:
                draw.ellipse([bx-r_bg,by-r_bg,bx+r_bg,by+r_bg], fill=bg)

        s  = size / 24.0
        cx = 12.0 * s
        cy = 12.0 * s
        rx = int(9.0 * s)
        ry = int(3.5 * s)
        lw = max(1, int(1.3 * s))

        # Draw three orbits at 0°, 60°, -60° by rotating on a temp layer
        try:
            resamp = Image.Resampling.BILINEAR
        except AttributeError:
            resamp = Image.BILINEAR  # Pillow < 9

        for angle in (0, 60, -60):
            layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            ld.arc(
                [(int(cx-rx), int(cy-ry)), (int(cx+rx), int(cy+ry))],
                0, 360, fill=(255, 255, 255, 128), width=lw,
            )
            if angle != 0:
                layer = layer.rotate(-angle, center=(cx, cy), resample=resamp)
            img = Image.alpha_composite(img, layer)

        draw = ImageDraw.Draw(img)

        # Electron dots at rightmost tip of each orbit (rotated positions)
        import math
        for angle_deg in (0, 60, -60):
            a = math.radians(angle_deg)
            ex = cx + 9.0 * s * math.cos(a)
            ey = cy + 9.0 * s * math.sin(a)
            er = 1.4 * s
            draw.ellipse([(ex-er, ey-er), (ex+er, ey+er)], fill=(255, 255, 255, 255))

        # Centre glow then dot
        gh = 3.5 * s
        draw.ellipse([(cx-gh, cy-gh), (cx+gh, cy+gh)], fill=(90, 140, 255, 65))
        dr = 2.1 * s
        draw.ellipse([(cx-dr, cy-dr), (cx+dr, cy+dr)], fill=(255, 255, 255, 255))

        return img
    except Exception:
        return "⚛"


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

# (label, question, expected behaviour) — curated to showcase each capability.
SHOWCASE_QUESTIONS = [
    ("🧭 Test Routing", "What is the latest news about GPT-5 today?",
     "→ routes to 🌐 Web Search automatically"),
    ("📄 Test Doc Retrieval", "What are the 3 novel features of this system?",
     "→ finds the exact answer from the document"),
    ("🛡️ Test Grounding", "Does this system guarantee zero hallucinations?",
     "→ watch the sentence-by-sentence grounding check"),
    ("💡 Test Gap Detection", "What is the monthly server cost breakdown?",
     "→ detects missing info and suggests what to upload"),
    ("⚡ Test General AI", "Explain cosine similarity in simple terms",
     "→ no retrieval — answers from LLM knowledge"),
]


def _hiw_step(num: str, num_bg: str, num_border: str, num_color: str,
              title: str, body: str) -> str:
    return (
        "<div style='display:flex;gap:12px;margin-bottom:10px'>"
        f"<div style='width:28px;height:28px;background:{num_bg};border:1.5px solid {num_border};"
        "border-radius:50%;display:flex;align-items:center;justify-content:center;"
        f"flex-shrink:0;font-size:12px;font-weight:700;color:{num_color}'>{num}</div>"
        f"<div><div style='font-size:12px;font-weight:700;color:#0F172A;margin-bottom:2px'>{title}</div>"
        f"<div style='font-size:12px;color:#64748B;line-height:1.5'>{body}</div></div></div>"
        "<div style='margin-left:14px;width:1px;height:10px;background:#E3E0D8;margin-bottom:10px'></div>"
    )


_HOW_IT_WORKS_HTML = (
    "<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-radius:14px;"
    "padding:22px 24px;margin-bottom:14px;box-shadow:none'>"
    "<div style='font-size:10px;font-weight:700;color:#94A3B8;text-transform:uppercase;"
    "letter-spacing:0.08em;margin-bottom:14px'>HOW IT WORKS — WORKED EXAMPLE</div>"
    "<div style='background:#FAF9F6;border:1px solid #E3E0D8;border-left:3px solid #059669;"
    "border-radius:8px;padding:11px 16px;margin-bottom:16px;font-size:13px;color:#334155'>"
    "💬 \"What are the key technical skills in this document?\"</div>"
    + _hiw_step("1", "#ECFDF5", "#A7F3D0", "#059669", "🧭 Route Classification",
                "LLM decides: <span style='color:#059669;font-weight:600'>INDEX</span> "
                "(document uploaded + question is doc-specific) · ~234ms")
    + _hiw_step("2", "#ECFDF5", "#A7F3D0", "#059669", "🔍 Semantic Retrieval + Grading",
                "Found 3 chunks · scored "
                "<span style='color:#059669'>0.89</span>, "
                "<span style='color:#059669'>0.82</span>, "
                "<span style='color:#059669'>0.76</span> · all ≥ 0.60 threshold ✅")
    + _hiw_step("3", "#ECFDF5", "#A7F3D0", "#059669", "✨ Generate + Stream",
                "LLaMA 3.3 70B · 847 tokens · $0.00018 · streamed word-by-word")
    # final step has no trailing connector
    + "<div style='display:flex;gap:12px'>"
      "<div style='width:28px;height:28px;background:#ECFDF5;border:1.5px solid #BBF7D0;"
      "border-radius:50%;display:flex;align-items:center;justify-content:center;"
      "flex-shrink:0;font-size:12px;font-weight:700;color:#059669'>✓</div>"
      "<div><div style='font-size:12px;font-weight:700;color:#0F172A;margin-bottom:2px'>"
      "🛡️ Grounding Verification</div>"
      "<div style='font-size:12px;color:#64748B;line-height:1.5'>Trust Score: "
      "<span style='color:#059669;font-weight:700'>87% HIGH</span> · "
      "✅ 5 grounded · ⚠️ 1 inferred · ❌ 0 ungrounded</div></div></div>"
    "</div>"
)


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
    """3-orbit atom mark — doc / web / general routes as orbits, electrons at tips."""
    ic = int(size * 0.86)
    return (
        f"<div style='width:{size}px;height:{size}px;"
        "background:linear-gradient(150deg,#06081c 0%,#0c1138 100%);"
        f"border-radius:{radius}px;"
        "display:flex;align-items:center;justify-content:center;"
        "box-shadow:0 4px 28px rgba(37,99,235,0.28),0 0 0 1px rgba(255,255,255,0.08);"
        "flex-shrink:0'>"

        f"<svg width='{ic}' height='{ic}' viewBox='0 0 24 24' fill='none'"
        " xmlns='http://www.w3.org/2000/svg'>"

        # Outer boundary ring — subtle frame
        "<circle cx='12' cy='12' r='11' stroke='white' stroke-width='0.6' opacity='0.10'/>"

        # Orbit 1 — horizontal
        "<ellipse cx='12' cy='12' rx='9' ry='3.5'"
        " stroke='white' stroke-width='1.25' opacity='0.55'/>"

        # Orbit 2 — rotated 60°
        "<ellipse cx='12' cy='12' rx='9' ry='3.5'"
        " stroke='white' stroke-width='1.25' opacity='0.55'"
        " transform='rotate(60 12 12)'/>"

        # Orbit 3 — rotated -60°
        "<ellipse cx='12' cy='12' rx='9' ry='3.5'"
        " stroke='white' stroke-width='1.25' opacity='0.55'"
        " transform='rotate(-60 12 12)'/>"

        # Electron on orbit 1  (rightmost: 21, 12)
        "<circle cx='21' cy='12' r='1.45' fill='white'/>"
        # Electron on orbit 2  (21,12 rotated 60° around 12,12 → 16.5, 19.8)
        "<circle cx='16.5' cy='19.8' r='1.45' fill='white'/>"
        # Electron on orbit 3  (21,12 rotated -60° around 12,12 → 16.5, 4.2)
        "<circle cx='16.5' cy='4.2' r='1.45' fill='white'/>"

        # Centre glow
        "<circle cx='12' cy='12' r='3.5' fill='rgba(90,145,255,0.22)'/>"
        # Centre dot
        "<circle cx='12' cy='12' r='2.1' fill='white'/>"
        # Inner highlight
        "<circle cx='11.1' cy='11.1' r='0.75' fill='white' opacity='0.35'/>"

        "</svg></div>"
    )


# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; height: 0 !important; }

/* Reclaim the big empty gap at the top: zero-height helper-script iframes
   (Ctrl+K / mic) and their containers shouldn't reserve a row. */
iframe[height="0"] { display: none !important; }
div[data-testid="stElementContainer"]:has(iframe[height="0"]),
div[data-testid="element-container"]:has(iframe[height="0"]) {
    display: none !important; height: 0 !important; margin: 0 !important;
}

/* ════════════════════════════════════════
   WHITE MAIN AREA
   ════════════════════════════════════════ */
.stApp                               { background-color: #FAF9F6 !important; }
[data-testid="stMain"]               { background-color: #FAF9F6 !important; }
[data-testid="stMainBlockContainer"] {
    background-color: #FAF9F6 !important;
    max-width: 920px !important;
    padding: 1rem 2.5rem 3rem !important;
}
.main  { background-color: #FAF9F6 !important; }
.block-container { background-color: #FAF9F6 !important; }

/* ════════════════════════════════════════
   DARK SIDEBAR — base (emerald accents)
   ════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid #1E293B !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] div { color: #94A3B8 !important; }

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #F1F5F9 !important; }

[data-testid="stSidebar"] hr { border-color: #1E293B !important; }

/* ════════════════════════════════════════
   SIDEBAR — FILE UPLOADER (dark)
   ════════════════════════════════════════ */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #1E293B !important;
    border: 1.5px dashed #334155 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #059669 !important;
    background: #1E293B !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div {
    background: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg {
    fill: #475569 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span {
    color: #94A3B8 !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
    color: #475569 !important;
    font-size: 11px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: #1E293B !important;
    color: #CBD5E1 !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 5px 14px !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
    background: #059669 !important;
    color: #FFFFFF !important;
    border-color: #059669 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
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
   SIDEBAR — TOGGLES (dark, emerald on)
   ════════════════════════════════════════ */
[data-testid="stSidebar"] [data-testid="stToggle"] p,
[data-testid="stSidebar"] [data-testid="stToggle"] label,
[data-testid="stSidebar"] .stToggle p,
[data-testid="stSidebar"] .stToggle label,
[data-testid="stSidebar"] .stToggle span {
    color: #CBD5E1 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stToggle"] [role="switch"][aria-checked="true"],
[data-testid="stSidebar"] .stToggle [role="switch"][aria-checked="true"] {
    background-color: #059669 !important;
}

/* ════════════════════════════════════════
   SIDEBAR — BUTTONS (dark)
   ════════════════════════════════════════ */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background-color: #1E293B !important;
    color: #CBD5E1 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #059669 !important;
    color: #FFFFFF !important;
    border-color: #059669 !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button {
    background-color: transparent !important;
    color: #64748B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    width: 100% !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button:hover {
    background-color: #1E293B !important;
    color: #059669 !important;
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
    border: 1.5px solid #E3E0D8 !important;
    box-shadow:none !important;
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
    border-color: #059669 !important;
    box-shadow: 0 0 0 3px rgba(5,150,105,0.12) !important;
}
[data-testid="stChatInput"] button {
    background: #059669 !important;
    border-radius: 8px !important;
    transition: background 0.15s !important;
}
[data-testid="stChatInput"] button:hover { background: #047857 !important; }

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
    border: 1px solid #E3E0D8 !important;
    border-radius: 10px !important;
    box-shadow:none !important;
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
    border: 1.5px solid #E3E0D8 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.15s !important;
    width: 100%;
}
.stButton > button:hover {
    border-color: #059669 !important;
    color: #059669 !important;
    background-color: #ECFDF5 !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08) !important;
}
[data-testid="stDownloadButton"] > button {
    background-color: transparent !important;
    color: #64748B !important;
    border: 1px solid #E3E0D8 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    width: 100%;
    transition: all 0.15s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: #FAF9F6 !important;
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
    background: #FAF9F6;
    border: 1px solid #E3E0D8;
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
@keyframes pulse-skeleton {
    0%, 100% { opacity: 0.4; }
    50%      { opacity: 1; }
}

/* ════════════════════════════════════════
   TABS — pill style
   ════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #F1F5F9;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #E3E0D8;
}
.stTabs [data-baseweb="tab"] {
    height: 38px;
    padding: 0 18px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    color: #64748B;
    background: transparent;
    border: none;
}
.stTabs [data-baseweb="tab"]:hover { color: #0F172A; background: rgba(255,255,255,0.6); }
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #059669 !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.10);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* Sidebar bordered containers (toggle group) — match the card style */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 2px 12px !important;
}

/* ════════════════════════════════════════
   INSTRUMENTATION-PANEL THEME — fonts
   ════════════════════════════════════════ */
/* Fraunces for main-area headings ONLY (never body/numbers) */
[data-testid="stMainBlockContainer"] h1,
[data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}
/* IBM Plex Mono for every number surface: metric values, code, and .num spans */
[data-testid="stMetricValue"],
[data-testid="stMainBlockContainer"] code,
.num {
    font-family: 'IBM Plex Mono', Consolas, 'SFMono-Regular', monospace !important;
}
/* Off-white page already set via palette swap; reinforce + ink text colour */
[data-testid="stMainBlockContainer"] { color: #1C2128 !important; }
/* Primary buttons (main area) → solid teal */
[data-testid="stMainBlockContainer"] [data-testid="stBaseButton-primary"],
[data-testid="stMainBlockContainer"] .stButton > button[kind="primary"] {
    background: #059669 !important;
    color: #FFFFFF !important;
    border: 1px solid #059669 !important;
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
        st.session_state.last_pipeline_trace = None   # Tab 3: Retrieval Inspector
        st.session_state.grounding_lab_result = None  # Tab 4: Grounding Lab
        st.session_state.demo_mode = False            # Instant-demo flow
        st.session_state.pending_demo_upload = False
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


def estimate_query_complexity(question: str, doc_available: bool) -> dict:
    """Heuristically predict route + complexity before a query is sent (Innovation #1)."""
    q = question.lower()
    search_kw = ["latest", "today", "current", "news", "2024", "2025", "2026", "price", "stock", "recent"]
    general_kw = ["what is", "explain", "define", "who invented", "history of"]
    if any(k in q for k in search_kw):
        route = "search"
    elif not doc_available or any(k in q for k in general_kw):
        route = "general"
    else:
        route = "index"
    words = len(question.split())
    heavy = any(w in q for w in ["compare", "difference", "versus", " vs ", "better",
                                 "all", "every", "total", "how many"])
    if words > 20 or heavy:
        complexity, est = "HIGH", "2–4s"
    elif words > 8:
        complexity, est = "MEDIUM", "1–2s"
    else:
        complexity, est = "LOW", "0.5–1s"
    icon = {"index": "📄 Document", "search": "🌐 Web", "general": "⚡ General"}[route]
    return {"predicted_route": route, "route_label": icon, "complexity": complexity, "est_time": est}


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
            bg      = "#ECFDF5"
            border  = "1.5px solid #059669"
            shadow  = "0 0 0 3px rgba(37,99,235,0.11),0 2px 8px rgba(37,99,235,0.13)"
            clr     = "#059669"
            lc, lw  = "#047857", "700"
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
            bg      = "#FAF9F6"
            border  = "1px solid #E3E0D8"
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
            fc = "#BBF7D0" if is_done else "#A7F3D0"
            nodes_html += (
                "<div style='display:flex;align-items:center;"
                "margin-top:-16px;width:20px;flex-shrink:0'>"
                "<div style='width:100%;height:2px;background:#E3E0D8;"
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
        ac = "#A7F3D0" if loop == 1 else "#FCA5A5"
        attempt_html = (
            "<div style='display:flex;align-items:center;gap:10px;"
            "background:#FAF9F6;border:1px solid #F1F5F9;"
            "border-radius:8px;padding:6px 12px;margin-bottom:12px'>"
            f"<span style='color:#94A3B8;font-size:10px;font-weight:700;"
            "letter-spacing:0.05em;min-width:68px'>ATTEMPT {loop}/2</span>"
            "<div style='flex:1;background:#E3E0D8;height:3px;"
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
            "background:#FAF9F6;border:1px solid #E3E0D8;"
            "border-radius:8px;padding:7px 12px'>"
            "<span style='color:#94A3B8;font-size:10px;font-weight:700;"
            "letter-spacing:0.05em;min-width:70px'>DOC MATCH</span>"
            "<div style='flex:1;background:#E3E0D8;height:3px;"
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
        "<div style='background:#FFFFFF;border:1px solid #E3E0D8;"
        "border-radius:14px;padding:14px 18px 16px;margin-bottom:12px;"
        "box-shadow:none;"
        "position:relative;overflow:hidden'>"

        # Gradient progress strip along top edge
        "<div style='position:absolute;top:0;left:0;right:0;height:2.5px;background:#F1F5F9'>"
        f"<div style='height:100%;width:{prog}%;"
        "background:linear-gradient(90deg,#059669,#7C3AED);"
        "transition:width .5s cubic-bezier(.4,0,.2,1)'></div></div>"

        # Header: pulsing dot + status message + step counter
        "<div style='display:flex;align-items:center;gap:10px;"
        "margin-bottom:14px;padding-top:4px'>"
        "<div style='width:7px;height:7px;border-radius:50%;background:#059669;"
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
        lambda m: f'<sup style="color:#059669;font-size:10px;font-weight:700">[{m.group(1)}]</sup>',
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
            f"<div style='background:#FAF9F6;border:1px solid #E3E0D8;"
            "border-left:3px solid #059669;border-radius:0 8px 8px 0;"
            "padding:10px 14px;margin-bottom:6px'>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
            f"<span style='font-size:10px;font-weight:700;color:#059669;"
            f"background:#ECFDF5;border:1px solid #A7F3D0;border-radius:4px;"
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
    }.get(level, ("#94A3B8", "#FAF9F6", "#E3E0D8", "Unverified"))
    bar_color, bg, border, label = cfg
    st.markdown(
        f"<div style='background:{bg};border:1px solid {border};border-radius:8px;"
        "padding:9px 14px;margin:8px 0 4px;display:flex;align-items:center;gap:12px'>"
        "<div style='flex:1;background:#E3E0D8;height:5px;border-radius:999px;overflow:hidden'>"
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
                b.style.background='#ECFDF5';
                b.style.borderColor='#93C5FD';
                b.style.color='#047857';
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
                   box-shadow:none"
            onmouseover="this.style.borderColor='#059669';this.style.color='#059669';this.style.background='#ECFDF5'"
            onmouseout="this.style.borderColor='#CBD5E1';this.style.color='#475569';this.style.background='#FFFFFF'">
            ⧉ Copy answer
        </button></div>""",
        height=36,
    )


def render_quick_stats_chips(metadata: dict) -> None:
    """Tiny chip row below every answer — instant visual summary of the run."""
    route = metadata.get("route_taken", "general")
    ms = metadata.get("processing_ms", 0)
    loops = metadata.get("loops_executed", 0) or 0
    cost = metadata.get("estimated_cost_usd", 0)
    trust_level = ((metadata.get("grounding") or {}).get("summary") or {}).get("trust_level", "")
    route_cfg = {
        "index": ("📄 Document", "#059669", "#ECFDF5"),
        "search": ("🌐 Web Search", "#7C3AED", "#F5F3FF"),
        "general": ("⚡ General AI", "#D97706", "#FFFBEB"),
    }
    rlabel, rcolor, rbg = route_cfg.get(route, (route, "#94A3B8", "#FAF9F6"))
    chips = [
        (rlabel, rcolor, rbg),
        (f"⏱ {format_time(ms)}", "#475569", "#FAF9F6"),
        (f"💰 ${cost:.5f}", "#475569", "#FAF9F6"),
    ]
    if loops > 1:
        chips.append((f"🔄 {loops} loops", "#7C3AED", "#F5F3FF"))
    trust_cfg = {"HIGH": ("🛡️", "#059669", "#ECFDF5"), "MODERATE": ("⚠️", "#D97706", "#FFFBEB"),
                 "LOW": ("❗", "#DC2626", "#FEF2F2")}
    if trust_level in trust_cfg:
        ti, tc, tb = trust_cfg[trust_level]
        chips.append((f"{ti} {trust_level}", tc, tb))
    chip_html = "".join(
        f"<span style='background:{bg};color:{color};font-size:10px;font-weight:600;"
        f"padding:3px 10px;border-radius:999px;border:1px solid {color}22;"
        f"white-space:nowrap'>{html.escape(text)}</span>"
        for text, color, bg in chips
    )
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 4px'>{chip_html}</div>",
        unsafe_allow_html=True,
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
        "index":   {"label": "Document",  "bg": "#ECFDF5", "color": "#047857", "border": "#A7F3D0", "icon": "📄"},
        "search":  {"label": "Web Search", "bg": "#F5F3FF", "color": "#6D28D9", "border": "#DDD6FE", "icon": "🌐"},
        "general": {"label": "General AI", "bg": "#FFFBEB", "color": "#92400E", "border": "#FDE68A", "icon": "⚡"},
    }
    rc = route_cfg.get(route, {"label": route, "bg": "#FAF9F6", "color": "#64748B", "border": "#E3E0D8", "icon": "⚡"})

    trust_html = ""
    if grounding and not grounding.get("skipped") and grounding.get("summary"):
        t_level = grounding["summary"].get("trust_level", "UNVERIFIED")
        t_score = grounding["summary"].get("trust_score", 0)
        trust_styles = {
            "HIGH":     {"bg": "#ECFDF5", "color": "#065F46", "border": "#A7F3D0"},
            "MODERATE": {"bg": "#FFFBEB", "color": "#92400E", "border": "#FDE68A"},
            "LOW":      {"bg": "#FEF2F2", "color": "#991B1B", "border": "#FECACA"},
        }
        ts = trust_styles.get(t_level, {"bg": "#FAF9F6", "color": "#64748B", "border": "#E3E0D8"})
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
        f"<span style='background:#FAF9F6;border:1px solid #E3E0D8;color:#64748B;"
        f"font-size:11px;padding:3px 10px;border-radius:999px'>⏱ {format_time(time_ms)}</span>"
        f"<span style='background:#FAF9F6;border:1px solid #E3E0D8;color:#64748B;"
        f"font-size:11px;padding:3px 10px;border-radius:999px'>💰 ${cost:.5f}</span>"
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
                    "<div style='flex:1;background:#E3E0D8;border-radius:999px;height:4px;overflow:hidden'>"
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
                    "<span style='color:#059669;font-size:10px;margin-left:6px;font-weight:600'>FINAL</span>"
                    if is_last else ""
                )
                attempts.append(
                    f"<div style='border-left:2px solid {'#059669' if is_last else '#E3E0D8'};"
                    "padding-left:12px;margin-bottom:4px'>"
                    "<div style='display:flex;align-items:center;gap:8px'>"
                    f"<span style='color:#475569;font-size:11px;font-weight:600'>"
                    f"Attempt {v.get('loop_number', idx+1)}</span>"
                    f"<span style='color:{qc};font-size:11px'>{q:.2f}{rewrite}</span>{final_chip}</div>"
                    "<div style='color:#94A3B8;font-size:11px;font-family:monospace;margin-top:2px'>"
                    f"\"{html.escape(v.get('query_used') or '—')}\"</div></div>"
                )
                if not is_last:
                    attempts.append("<div style='color:#E3E0D8;padding-left:14px;font-size:12px'>↓</div>")
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
                    "<div style='flex:1;background:#E3E0D8;border-radius:999px;height:5px;overflow:hidden'>"
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
            f"background:#FAF9F6;border-radius:6px;padding:8px'>{raw}</pre></div></details>",
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
        f"<span style='color:#059669'>→</span>{html.escape(str(d))}</div>"
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
        "<div style='background:#ECFDF5;border:1px solid #A7F3D0;border-radius:10px;"
        "padding:14px 16px;margin-top:10px;"
        "box-shadow:none'>"
        "<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px'>"
        "<span style='color:#047857;font-size:13px;font-weight:600'>💡 Make this answer more complete</span>"
        f"<span style='color:#047857;font-size:10px;font-weight:700;"
        f"background:#DBEAFE;border:1px solid #A7F3D0;"
        f"padding:2px 9px;border-radius:999px'>{score}/10 complete</span></div>"
        f"<div style='color:#334155;font-size:12px;margin-bottom:10px;line-height:1.55'>{summary}</div>"
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

# ── INSTANT DEMO — auto-upload the sample doc and fire the first question ──────
if st.session_state.get("pending_demo_upload"):
    st.session_state.pending_demo_upload = False
    from pathlib import Path as _Path
    _sample = _Path(__file__).parent / "evaluate" / "sample_doc.md"
    if _sample.exists() and not st.session_state.uploaded_files:
        with open(_sample, "rb") as _f:
            _content = _f.read()
        # Retry: the embedding service occasionally drops a connection (transient).
        _ok = False
        for _attempt in range(3):
            try:
                _r = requests.post(
                    f"{API_BASE}/api/upload",
                    files={"file": ("Adaptive_RAG_Overview.md", _content, "text/markdown")},
                    data={"session_id": st.session_state.session_id},
                    timeout=90,
                )
                if _r.ok:
                    _res = _r.json()
                    st.session_state.uploaded_files.append({
                        "name": "Adaptive_RAG_Overview.md",
                        "chunks": _res.get("child_count", 0),
                    })
                    st.session_state.demo_mode = True
                    st.session_state.pending_prompt = (
                        "What makes this Adaptive RAG system different from a standard RAG chatbot?"
                    )
                    _ok = True
                    break
            except Exception:
                pass  # transient — retry
        if not _ok:
            st.session_state.demo_error = (
                "Demo upload hit a transient error — click Launch Demo again."
            )
    st.rerun()

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
        "<div style='font-size:15px;font-weight:800;color:#F1F5F9;letter-spacing:-0.3px'>"
        "Adaptive RAG AI</div>"
        "<div style='font-size:10px;color:#64748B;margin-top:2px'>"
        "Intelligent Document Intelligence</div>"
        "</div></div>"
        "<hr style='border:none;border-top:1px solid #1E293B;margin:0 0 4px'>",
        unsafe_allow_html=True,
    )

    # Mode indicator
    if st.session_state.uploaded_files:
        _mc, _mbg, _mbd, _mi, _mt, _ms = (
            "#4ADE80", "#052E16", "#166534", "📄",
            "Document Mode", f"{len(st.session_state.uploaded_files)} doc(s) indexed")
    else:
        _mc, _mbg, _mbd, _mi, _mt, _ms = (
            "#34D399", "#0B2E22", "#155E47", "⚡",
            "General Mode", "No doc — web search + AI knowledge")
    st.markdown(
        f"<div style='background:{_mbg};border:1px solid {_mbd};border-radius:8px;"
        "padding:8px 12px;margin-bottom:12px;display:flex;align-items:center;gap:10px'>"
        f"<span style='font-size:18px'>{_mi}</span><div>"
        f"<div style='color:{_mc};font-size:12px;font-weight:700'>{_mt}</div>"
        f"<div style='color:#64748B;font-size:10px'>{_ms}</div></div></div>",
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
                "background:#1E293B;border:1px solid #334155;"
                "border-radius:10px;padding:10px 14px;margin:4px 0'>"
                "<div style='width:16px;height:16px;border:2.5px solid #166534;"
                "border-top-color:#4ADE80;border-radius:50%;flex-shrink:0;"
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
            "background:#1E293B;"
            "border:1px solid #334155;"
            "border-radius:8px;padding:8px 12px;margin:4px 0'>"
            f"<span style='font-size:16px'>{icon}</span>"
            "<div style='flex:1;min-width:0'>"
            f"<div style='font-size:12px;font-weight:600;color:#F1F5F9;"
            f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{html.escape(finfo['name'])}</div>"
            f"<div style='font-size:10px;color:#64748B;margin-top:1px'>{chunk_note}</div>"
            "</div>"
            "<span style='font-size:10px;background:#052E16;"
            "color:#4ADE80;border:1px solid #166534;"
            "border-radius:10px;padding:2px 8px'>✓</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    if not st.session_state.uploaded_files:
        st.markdown(
            "<div style='color:#64748B;font-size:12px;padding:6px 0 4px;line-height:1.5'>"
            "No documents yet — questions will use web search or general knowledge.</div>",
            unsafe_allow_html=True,
        )

    # Transparency
    st.markdown(
        "<hr style='border:none;border-top:1px solid #1E293B;margin:14px 0 4px'>",
        unsafe_allow_html=True,
    )
    _sidebar_section("TRANSPARENCY")
    with st.container(border=True):
        st.toggle("Grounding Check", value=True, key="enable_grounding",
                  help="Verify each sentence against your document.")
        st.toggle("Answer Evolution", value=True, key="enable_versioning",
                  help="Show how the answer improved across retrieval attempts.")
        st.toggle("Knowledge Gap Alerts", value=True, key="enable_gaps",
                  help="Flag when the document lacks enough info and suggest uploads.")

    # Session stats
    st.markdown(
        "<hr style='border:none;border-top:1px solid #E3E0D8;margin:14px 0 4px'>",
        unsafe_allow_html=True,
    )
    _sidebar_section("SESSION")

    if st.session_state.get("last_metadata"):
        meta = st.session_state.last_metadata
        route_val = meta.get("route_taken", "—")
        route_label = {"index": "📄 Doc", "search": "🌐 Web", "general": "⚡ AI"}.get(route_val, route_val)
        route_color = {"index": "#4ADE80", "search": "#C4B5FD", "general": "#FBBF24"}.get(route_val, "#94A3B8")

        grounding_m = meta.get("grounding") or {}
        trust_val, trust_color = "—", "#94A3B8"
        if not grounding_m.get("skipped") and grounding_m.get("summary"):
            t_lv = grounding_m["summary"].get("trust_level", "—")
            trust_val = t_lv
            trust_color = {"HIGH": "#4ADE80", "MODERATE": "#FBBF24", "LOW": "#FCA5A5"}.get(t_lv, "#94A3B8")

        def _stat(label, color, value, mono=True):
            cls = " class='num'" if mono else ""
            return (
                "<div style='background:#1E293B;border:1px solid #334155;"
                "border-radius:8px;padding:9px 11px'>"
                f"<div style='font-size:9px;font-weight:700;color:#64748B;"
                f"text-transform:uppercase;letter-spacing:0.07em'>{label}</div>"
                f"<div{cls} style='font-size:13px;font-weight:700;color:{color};margin-top:3px'>{value}</div></div>"
            )

        cards = (
            _stat("ROUTE", route_color, route_label, mono=False)
            + _stat("TIME", "#F1F5F9", format_time(meta.get("processing_ms", 0)))
            + _stat("LOOPS", "#F1F5F9", str(meta.get("loops_executed", 0)))
            + _stat("COST", "#F1F5F9", f"${meta.get('estimated_cost_usd', 0):.4f}")
            + _stat("TRUST", trust_color, trust_val, mono=False)
            + _stat("TOKENS", "#F1F5F9", f"{(meta.get('token_usage') or {}).get('completion', 0):,}")
        )
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:8px 0'>{cards}</div>",
            unsafe_allow_html=True,
        )

    _budget = 0.10  # $0.10 demo budget
    _cost_pct = min(int(st.session_state.total_cost / _budget * 100), 100) if _budget else 0
    _cost_color = "#4ADE80" if _cost_pct < 50 else "#FBBF24" if _cost_pct < 80 else "#FCA5A5"
    st.markdown(
        "<div style='margin:8px 0 12px'>"
        "<div style='display:flex;justify-content:space-between;margin-bottom:4px'>"
        "<span style='color:#64748B;font-size:10px;font-weight:600'>SESSION COST</span>"
        f"<span class='num' style='color:{_cost_color};font-size:10px;font-weight:700'>"
        f"${st.session_state.total_cost:.4f}</span></div>"
        "<div style='background:#1E293B;border-radius:999px;height:3px'>"
        f"<div style='width:{_cost_pct}%;height:100%;background:{_cost_color};"
        "border-radius:999px;transition:width 0.3s'></div></div>"
        f"<div style='color:#64748B;font-size:10px;margin-top:3px'>"
        f"<span class='num'>{st.session_state.total_tokens:,}</span> tokens · "
        f"<span class='num'>${_budget:.2f}</span> demo budget</div></div>",
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

    # ── Session timeline (Innovation #2) ──────────────────────────────────
    _tl_pairs = []
    _ti = 0
    while _ti < len(_qh_msgs) - 1:
        if _qh_msgs[_ti]["role"] == "user" and _qh_msgs[_ti + 1]["role"] == "assistant":
            _m = _qh_msgs[_ti + 1].get("meta") or {}
            _g = (_m.get("grounding") or {}).get("summary") or {}
            _tl_pairs.append((_qh_msgs[_ti]["content"], _m.get("route_taken", "general"),
                              _g.get("trust_level")))
            _ti += 2
        else:
            _ti += 1
    if _tl_pairs:
        _sidebar_section("SESSION TIMELINE")
        _tl_rows = []
        for _q, _rt, _trust in _tl_pairs[-5:]:
            _ico = _route_icon.get(_rt, "⚡")
            _short = (_q[:26] + "…") if len(_q) > 26 else _q
            _tnote = (f" · {_trust.lower()}" if _trust else "")
            _tl_rows.append(
                "<div style='display:flex;gap:8px;align-items:baseline;padding:3px 0;"
                "border-left:2px solid #E3E0D8;padding-left:10px;margin-left:2px'>"
                f"<span style='font-size:12px'>{_ico}</span>"
                f"<span style='font-size:11px;color:#6B6F76;line-height:1.4'>"
                f"{html.escape(_short)}<span style='color:#9A9EA5'>{_tnote}</span></span></div>"
            )
        st.markdown("".join(_tl_rows), unsafe_allow_html=True)

    # Backend status
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if check_backend():
        st.markdown(
            "<span style='display:inline-flex;align-items:center;gap:6px;"
            "font-size:11px;font-weight:600;"
            "background:#ECFDF5;color:#059669;"
            "border:1px solid #A7F3D0;"
            "border-radius:20px;padding:4px 12px'>"
            "<span style='width:6px;height:6px;border-radius:50%;"
            "background:#059669;display:inline-block'></span>"
            "Backend connected</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span style='display:inline-flex;align-items:center;gap:6px;"
            "font-size:11px;font-weight:600;"
            "background:#FBEDED;color:#A63D40;"
            "border:1px solid #E8C4C5;"
            "border-radius:20px;padding:4px 12px'>"
            "<span style='width:6px;height:6px;border-radius:50%;"
            "background:#A63D40;display:inline-block'></span>"
            "Backend offline</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:10px;color:#94A3B8;line-height:1.5;margin-top:6px'>"
            "On the free tier the server sleeps when idle — the first request can "
            "take ~30–50s to wake it. Send a question and give it a moment.</div>",
            unsafe_allow_html=True,
        )



# ══ MAIN AREA — wrapped as the Chat tab ══
def render_chat_tab():
    # ── MAIN AREA ─────────────────────────────────────────────────────────────────

    # Hero banner — dark gradient card sitting on the white page (FinGuard style)
    _HERO_BADGES = ["Groq LLaMA 3.3", "Adaptive RAG", "Qdrant Vector DB", "FastAPI", "Multi-Document"]
    _badge_pills = "".join(
        "<span style='background:#ECFDF5;border:1px solid #A7F3D0;border-radius:999px;"
        f"color:#059669;font-size:12px;font-weight:600;padding:5px 14px;white-space:nowrap'>{b}</span>"
        for b in _HERO_BADGES
    )

    # Live benchmark stats strip — only when real RAGAS results exist.
    _ragas = _load_results_json("ragas_results.json")
    _routing = _load_results_json("routing_results.json")
    if _ragas and _ragas.get("metrics"):
        _m = _ragas["metrics"]

        def _hstat(value, label, sub):
            return (
                "<div style='display:flex;flex-direction:column;gap:2px'>"
                f"<span class='num' style='color:#1C2128;font-size:21px;font-weight:600'>{value}</span>"
                f"<span style='color:#6B6F76;font-size:10px;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.06em'>{label}</span>"
                f"<span style='color:#9A9EA5;font-size:10px'>{sub}</span></div>"
            )

        _div = "<div style='width:1px;background:#E3E0D8'></div>"
        _route_acc = f"{_routing.get('accuracy_pct', 0):.0f}%" if _routing else "—"
        _route_sub = (f"{_routing.get('correct', 0)}/{_routing.get('total', 0)} test cases"
                      if _routing else "run routing eval")
        _hero_stats = (
            "<div style='border-top:1px solid #E3E0D8;margin-top:18px;"
            "padding-top:14px;display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start'>"
            + _hstat("3", "Routes", "doc · web · general") + _div
            + _hstat(f"{_m.get('faithfulness', 0):.3f}", "Faithfulness", "RAGAS · +7.5% vs naive") + _div
            + _hstat(_route_acc, "Routing Accuracy", _route_sub) + _div
            + _hstat(f"{_m.get('context_precision', 0):.3f}", "Context Precision", "RAGAS · perfect score")
            + "</div>"
        )
    else:
        _hero_stats = (
            "<div style='border-top:1px solid #E3E0D8;margin-top:18px;"
            "padding-top:14px;color:#9A9EA5;font-size:11px'>"
            "Run the benchmark scripts to surface live performance stats.</div>"
        )

    st.markdown(
        "<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-left:3px solid #059669;"
        "border-radius:0 8px 8px 0;padding:1.75rem 2rem;margin-bottom:1.5rem'>"
        # title row
        "<div style='display:flex;align-items:center;gap:14px;margin-bottom:8px'>"
        + _logo_html(48, 12) +
        "<div>"
        "<h1 style='margin:0;font-size:26px;line-height:1.15'>"
        "Adaptive <span style='color:#059669'>RAG AI</span></h1>"
        "<div style='font-size:12px;color:#6B6F76;margin-top:2px'>"
        "Powered by Groq · LLaMA 3.3 · Qdrant</div>"
        "</div></div>"
        # subtitle
        "<p style='color:#3F434A;font-size:14px;line-height:1.7;margin:0 0 16px;max-width:580px'>"
        "Ask anything about your documents — grounded answers, real sources, every claim "
        "checked against what you uploaded.</p>"
        # badges
        f"<div style='display:flex;flex-wrap:wrap;gap:8px'>{_badge_pills}</div>"
        # live benchmark stats strip
        f"{_hero_stats}"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Recruiter demo — capability test buttons (when a document is loaded) ───
    if st.session_state.uploaded_files:
        with st.expander("🎯 Recruiter demo — click a capability to test",
                         expanded=st.session_state.get("demo_mode", False)):
            for _lbl, _q, _hint in SHOWCASE_QUESTIONS:
                _cb, _ch = st.columns([1, 2])
                if _cb.button(_lbl, use_container_width=True, key=f"showcase_{_lbl}"):
                    st.session_state.pending_prompt = _q
                    st.rerun()
                _ch.markdown(
                    f"<div style='padding:6px 0;font-size:11px;color:#94A3B8'>"
                    f"<span style='color:#475569'>{html.escape(_q[:48])}"
                    f"{'…' if len(_q) > 48 else ''}</span>"
                    f"<div style='color:#059669;margin-top:1px'>{html.escape(_hint)}</div></div>",
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
            "<code style='background:#F1F5F9;border:1px solid #E3E0D8;border-radius:6px;"
            "padding:8px 14px;display:block;font-size:12px;color:#0F172A;margin:4px 0'>"
            "cd adaptive_rag &amp;&amp; python start.py</code>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Empty states
    if not st.session_state.messages:
        if not st.session_state.uploaded_files:
            # ── Instant demo banner + one-click launch ───────────────────────
            if st.session_state.get("demo_error"):
                st.warning(st.session_state.pop("demo_error"))
            st.markdown(
                "<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-left:3px solid #059669;"
                "border-radius:0 6px 6px 0;padding:18px 24px;margin-bottom:12px'>"
                "<div style='display:flex;align-items:center;justify-content:space-between'>"
                "<div><div style='color:#059669;font-size:10px;font-weight:700;"
                "letter-spacing:0.1em;margin-bottom:4px'>INSTANT DEMO</div>"
                "<div style='color:#1C2128;font-size:16px;font-weight:700;margin-bottom:4px'>"
                "Try it in one click</div>"
                "<div style='color:#6B6F76;font-size:13px'>"
                "Loads a real AI document and asks the first smart question automatically."
                "</div></div><div style='font-size:34px;color:#059669'>⚡</div></div></div>",
                unsafe_allow_html=True,
            )
            if st.button("⚡ Launch Demo — No Upload Needed",
                         use_container_width=True, type="primary", key="launch_demo"):
                st.session_state.pending_demo_upload = True
                st.rerun()

            # ── How it works — concrete worked example ───────────────────────
            st.markdown(_HOW_IT_WORKS_HTML, unsafe_allow_html=True)

            st.markdown(
                "<div style='text-align:center;color:#94A3B8;font-size:11px;"
                "font-weight:600;letter-spacing:0.1em;margin:6px 0 14px'>— OR —</div>",
                unsafe_allow_html=True,
            )

            # Section header
            st.markdown(
                "<div style='display:flex;align-items:center;gap:10px;margin-bottom:16px'>"
                "<div style='width:4px;height:22px;background:#059669;border-radius:2px'></div>"
                "<h2 style='margin:0;font-size:18px;font-weight:700;color:#0F172A'>"
                "Get started</h2></div>",
                unsafe_allow_html=True,
            )
            _, center, _ = st.columns([1, 2, 1])
            with center:
                st.markdown(
                    # Card wrapper
                    "<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-radius:14px;"
                    "padding:1.75rem 1.75rem 1.5rem;"
                    "box-shadow:none'>"
                    # Icon + heading block (centered)
                    "<div style='text-align:center;margin-bottom:14px'>"
                    "<div style='display:inline-flex;align-items:center;justify-content:center;"
                    "width:56px;height:56px;border-radius:14px;"
                    "background:linear-gradient(135deg,#ECFDF5,#DBEAFE);margin-bottom:10px'>"
                    "<svg width='28' height='28' viewBox='0 0 24 24' fill='none'"
                    " xmlns='http://www.w3.org/2000/svg'>"
                    "<path d='M12 16V4M12 4L8 8M12 4L16 8' stroke='#059669'"
                    " stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/>"
                    "<path d='M3 17v1a3 3 0 003 3h12a3 3 0 003-3v-1' stroke='#059669'"
                    " stroke-width='2' stroke-linecap='round'/>"
                    "</svg></div>"
                    "<div style='font-size:15px;font-weight:700;color:#059669;margin-bottom:5px'>"
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
                    "<div style='background:#ECFDF5;border-left:3px solid #A7F3D0;"
                    "border-radius:0 8px 8px 0;padding:11px 15px;margin-top:12px'>"
                    "<span style='font-size:13px;color:#3B82F6;font-weight:500'>"
                    "💬 Or ask a general question below — I'll search the web or use my own knowledge."
                    "</span></div>",
                    unsafe_allow_html=True,
                )

                # ── Starter questions — one-click, no upload needed ───────────
                st.markdown(
                    "<div style='font-size:10px;font-weight:700;color:#94A3B8;"
                    "text-transform:uppercase;letter-spacing:0.09em;margin:22px 0 8px'>"
                    "Try asking</div>",
                    unsafe_allow_html=True,
                )
                _starters = [
                    "What is Retrieval-Augmented Generation?",
                    "Explain vector embeddings in simple terms",
                    "What are the latest developments in AI agents?",
                    "How does semantic search differ from keyword search?",
                ]
                _scols = st.columns(2)
                for _si, _sq in enumerate(_starters):
                    with _scols[_si % 2]:
                        if st.button(_sq, key=f"starter_{_si}", use_container_width=True):
                            st.session_state.pending_prompt = _sq
                            st.rerun()
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
                "<div style='width:4px;height:22px;background:#059669;border-radius:2px'></div>"
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
                "<div style='background:#FFFFFF;border:1px solid #E3E0D8;"
                "border-radius:12px;padding:16px 20px;margin-bottom:14px;"
                "box-shadow:none'>"
                "<div style='display:flex;align-items:flex-start;gap:12px'>"
                "<div style='width:38px;height:38px;border-radius:8px;"
                "background:linear-gradient(135deg,#ECFDF5,#DBEAFE);"
                "display:flex;align-items:center;justify-content:center;flex-shrink:0'>"
                "<svg width='20' height='20' viewBox='0 0 24 24' fill='none'>"
                "<path d='M6 2.5h9l5 5v14H6V2.5z' stroke='#059669' stroke-width='1.9'"
                " fill='none' stroke-linejoin='round'/>"
                "<path d='M15 2.5v5h5' stroke='#059669' stroke-width='1.4' fill='none'/>"
                "<line x1='9' y1='12.5' x2='16' y2='12.5' stroke='#059669'"
                " stroke-width='1.5' stroke-linecap='round'/>"
                "<line x1='9' y1='16' x2='13' y2='16' stroke='#059669'"
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
                    "background:#FFFFFF;border:1px solid #E3E0D8;border-radius:10px;"
                    "padding:12px 16px;margin-bottom:14px;"
                    "box-shadow:none'>"
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
                if message.get("meta") and not message["meta"].get("error"):
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
            'background:#FFFFFF','border:1.5px solid #E3E0D8',
            'box-shadow:none',
            'cursor:pointer','font-size:16px',
            'display:flex','align-items:center','justify-content:center',
            'transition:all .15s ease','outline:none'
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
                btn.style.borderColor = '#E3E0D8';
                btn.style.background = '#FFFFFF';
                btn.style.boxShadow = 'none';
            };
            r.onerror = function() {
                listening = false;
                btn.textContent = '🎤';
                btn.style.borderColor = '#E3E0D8';
                btn.style.background = '#FFFFFF';
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
    # Innovation #1 — query complexity meter (router preview of the last query)
    _lpt = st.session_state.get("last_pipeline_trace")
    if _lpt and _lpt.get("question"):
        _cx = estimate_query_complexity(_lpt["question"], bool(st.session_state.uploaded_files))
        st.caption(
            f"Router preview · last query → {_cx['route_label']} route · "
            f"{_cx['complexity']} complexity (~{_cx['est_time']})"
        )
    pending = st.session_state.pop("pending_prompt", None)
    if pending and not user_input:
        user_input = pending

    # Guard: ignore identical message submitted within 2 seconds (double-click / double-enter)
    if user_input:
        import time as _t
        _now = _t.time()
        _last_t = st.session_state.get("_last_submit_t", 0.0)
        _last_c = st.session_state.get("_last_submit_c", "")
        if user_input == _last_c and (_now - _last_t) < 2.0:
            user_input = None
        else:
            st.session_state["_last_submit_t"] = _now
            st.session_state["_last_submit_c"] = user_input

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
                err_msg = stream_error or "No response received from the backend."
                render_error_state(err_msg)
                # Save error as an assistant message so the conversation history is
                # preserved and the user can see what went wrong without the question
                # silently disappearing.
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"[Error: {err_msg}]", "meta": {"error": True}}
                )
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
                # Tab 3: capture the full pipeline trace of this (the latest) query.
                st.session_state.last_pipeline_trace = {
                    "question": user_input,
                    "route_taken": data.get("route_taken"),
                    "loops_executed": data.get("loops_executed", 0),
                    "relevance_scores": data.get("relevance_scores", []),
                    "answer_versions": data.get("answer_versions", []),
                    "source_chunks": data.get("source_chunks", []),
                    "grounding": data.get("grounding", {}),
                    "processing_ms": data.get("processing_ms", 0),
                    "estimated_cost_usd": data.get("estimated_cost_usd", 0),
                    "token_usage": data.get("token_usage", {}),
                    "answer": data.get("answer", ""),
                    "knowledge_gaps": data.get("knowledge_gaps"),
                }
                st.session_state.messages.append(
                    {"role": "assistant", "content": data.get("answer", full_answer), "meta": data}
                )
                st.rerun()



# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD TABS  (Benchmark · Inspector · Grounding Lab)
# ══════════════════════════════════════════════════════════════════════════════

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate", "results")


def _load_results_json(name: str):
    """Load a results JSON file, returning None if missing or unreadable."""
    path = os.path.join(_RESULTS_DIR, name)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _section_header(title: str, subtitle: str = "") -> None:
    sub = (f"<div style='font-size:12px;color:#94A3B8;margin-top:2px'>{html.escape(subtitle)}</div>"
           if subtitle else "")
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin:18px 0 12px'>"
        "<div style='width:4px;height:22px;background:#059669;border-radius:2px'></div>"
        f"<div><h2 style='margin:0;font-size:18px;font-weight:700;color:#0F172A'>{html.escape(title)}</h2>"
        f"{sub}</div></div>",
        unsafe_allow_html=True,
    )


def _placeholder_card(message: str, commands: str = "") -> None:
    cmd_html = (
        "<pre style='background:#F1F5F9;border:1px solid #E3E0D8;border-radius:8px;"
        "padding:10px 14px;font-size:12px;color:#0F172A;margin:10px 0 0;white-space:pre-wrap'>"
        f"{html.escape(commands)}</pre>" if commands else ""
    )
    st.markdown(
        "<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-radius:12px;"
        "padding:20px 24px;box-shadow:none'>"
        f"<div style='font-size:13px;color:#64748B;line-height:1.6'>{html.escape(message)}</div>"
        f"{cmd_html}</div>",
        unsafe_allow_html=True,
    )


def _benchmark_feature_table() -> None:
    """Always-true feature matrix (independent of result files)."""
    _section_header("Feature Comparison")
    features = [
        ("Query Routing", "❌ Always retrieves", "✅ index / web / general"),
        ("Relevance Grading", "❌ None", "✅ 0.6 threshold"),
        ("Query Rewriting", "❌ None", "✅ Up to 2 retries"),
        ("Web Search Fallback", "❌ None", "✅ Tavily API"),
        ("Hallucination Check", "❌ None", "✅ Sentence-level"),
        ("Answer Versioning", "❌ None", "✅ Per-loop tracking"),
        ("Knowledge Gap Alerts", "❌ None", "✅ With caching"),
        ("Session Isolation", "❌ None", "✅ Per-user collection"),
    ]
    rows = "".join(
        "<tr>"
        f"<td style='padding:9px 14px;border-bottom:1px solid #F1F5F9;font-size:13px;"
        f"color:#0F172A;font-weight:600'>{html.escape(f)}</td>"
        f"<td style='padding:9px 14px;border-bottom:1px solid #F1F5F9;font-size:12px;"
        f"color:#94A3B8'>{n}</td>"
        f"<td style='padding:9px 14px;border-bottom:1px solid #F1F5F9;font-size:12px;"
        f"color:#059669;font-weight:500'>{a}</td>"
        "</tr>"
        for f, n, a in features
    )
    st.markdown(
        "<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-radius:12px;"
        "overflow:hidden;box-shadow:none'>"
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr style='background:#FAF9F6'>"
        "<th style='padding:10px 14px;text-align:left;font-size:11px;color:#64748B;"
        "text-transform:uppercase;letter-spacing:0.05em'>Feature</th>"
        "<th style='padding:10px 14px;text-align:left;font-size:11px;color:#64748B;"
        "text-transform:uppercase;letter-spacing:0.05em'>Naive RAG</th>"
        "<th style='padding:10px 14px;text-align:left;font-size:11px;color:#64748B;"
        "text-transform:uppercase;letter-spacing:0.05em'>Adaptive RAG</th></tr>"
        f"{rows}</table></div>",
        unsafe_allow_html=True,
    )


# ── TAB 2 — Benchmark Results ────────────────────────────────────────────────
def render_benchmark_results_tab() -> None:
    st.markdown(
        "<h1 style='font-size:22px;font-weight:800;color:#0F172A;margin:4px 0 2px'>"
        "📊 Benchmark Results</h1>"
        "<div style='font-size:13px;color:#64748B;margin-bottom:6px'>"
        "Adaptive RAG vs Naive RAG — measured with RAGAS (reference-free, "
        "judged by LLaMA-3.1-8B). Pre-computed; no live calls.</div>",
        unsafe_allow_html=True,
    )

    ragas = _load_results_json("ragas_results.json")
    naive = _load_results_json("naive_rag_results.json")
    routing = _load_results_json("routing_results.json")

    if not ragas or not naive:
        _placeholder_card(
            "Benchmarks not run yet. With the backend running (python main.py), run these "
            "in order:",
            "python evaluate/ragas_eval.py\npython evaluate/naive_rag.py\n"
            "python evaluate/compare.py\npython evaluate/test_routing.py\n"
            "python evaluate/benchmark_latency.py",
        )
        # Still show the always-true feature comparison below.
        _benchmark_feature_table()
        return

    a_m = ragas.get("metrics", {})
    n_m = naive.get("metrics", {})

    # ── Hero metrics row ──────────────────────────────────────────────────────
    h1, h2, h3, h4 = st.columns(4)
    _af, _nf = a_m.get("faithfulness"), n_m.get("faithfulness")
    _ac, _nc = a_m.get("context_precision"), n_m.get("context_precision")
    if isinstance(_af, (int, float)) and isinstance(_nf, (int, float)):
        h1.metric("Faithfulness", f"{_af:.3f}", delta=f"{_af - _nf:+.3f} vs naive")
    if isinstance(_ac, (int, float)) and isinstance(_nc, (int, float)):
        h2.metric("Context Precision", f"{_ac:.3f}", delta=f"{_ac - _nc:+.3f} vs naive")
    if routing:
        h3.metric("Routing Accuracy", f"{routing.get('accuracy_pct', 0):.0f}%",
                  delta=f"{routing.get('correct', 0)}/{routing.get('total', 0)} cases",
                  delta_color="off")
    else:
        h3.metric("Routing Accuracy", "—")
    _alat = ragas.get("summary", {}).get("avg_latency_ms")
    _nlat = naive.get("summary", {}).get("avg_latency_ms")
    if isinstance(_alat, (int, float)):
        h4.metric("Avg Latency", f"{_alat:,} ms",
                  delta=(f"{_alat - _nlat:+,} ms vs naive" if isinstance(_nlat, (int, float)) else None),
                  delta_color="inverse")

    # ── Side-by-side comparison cards ─────────────────────────────────────────
    _section_header("Naive RAG vs Adaptive RAG")

    def _cmp_lines(metrics: dict, other: dict) -> str:
        # Green ONLY on the genuinely-better number this row (higher is better).
        order = [("faithfulness", "Faithfulness"), ("answer_relevancy", "Answer Relevancy"),
                 ("context_precision", "Context Precision")]
        out = []
        for k, lbl in order:
            v, o = metrics.get(k), other.get(k)
            vs = f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"
            is_better = isinstance(v, (int, float)) and isinstance(o, (int, float)) and v > o
            color = "#059669" if is_better else "#1C2128"
            out.append(
                "<div style='display:flex;justify-content:space-between;padding:4px 0'>"
                f"<span style='font-size:12px;color:#6B6F76'>{lbl}</span>"
                f"<span class='num' style='font-size:13px;font-weight:600;color:{color}'>{vs}</span></div>"
            )
        return "".join(out)

    col_n, col_arrow, col_a = st.columns([2, 0.4, 2])
    col_n.markdown(
        "<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-left:3px solid #B5651D;"
        "border-radius:0 6px 6px 0;padding:18px 20px'>"
        "<div style='font-size:14px;font-weight:700;color:#B5651D'>Naive RAG</div>"
        "<div style='font-size:10px;color:#9A9EA5;font-weight:600;text-transform:uppercase;"
        "letter-spacing:0.05em;margin-bottom:10px'>Baseline</div>"
        f"{_cmp_lines(n_m, a_m)}"
        f"<div style='font-size:11px;color:#9A9EA5;margin-top:8px'>"
        f"avg latency <span class='num'>{naive.get('summary', {}).get('avg_latency_ms', '—')}</span> ms · "
        "always retrieves</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    col_arrow.markdown(
        "<div style='text-align:center;margin-top:54px;font-size:24px;color:#059669'>→</div>",
        unsafe_allow_html=True,
    )
    col_a.markdown(
        "<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-left:3px solid #059669;"
        "border-radius:0 6px 6px 0;padding:18px 20px'>"
        "<div style='font-size:14px;font-weight:700;color:#059669'>Adaptive RAG</div>"
        "<div style='font-size:10px;color:#9A9EA5;font-weight:600;text-transform:uppercase;"
        "letter-spacing:0.05em;margin-bottom:10px'>Ours · routed + graded</div>"
        f"{_cmp_lines(a_m, n_m)}"
        f"<div style='font-size:11px;color:#9A9EA5;margin-top:8px'>"
        f"avg latency <span class='num'>{ragas.get('summary', {}).get('avg_latency_ms', '—')}</span> ms · "
        "routed + graded</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Coverage note — the metric numbers live once in the hero row + the
    # side-by-side cards above; here we only explain the sample.
    _summ = ragas.get("summary", {})
    st.markdown(
        "<div style='font-size:11px;color:#94A3B8;margin:4px 0 2px'>"
        f"RAGAS judged by LLaMA-3.1-8B · adaptive scored over "
        f"{_summ.get('questions_with_context', _summ.get('total_questions', '?'))} of "
        f"{_summ.get('total_questions', '?')} questions that retrieved from the document "
        "(others routed to web search and are excluded from context metrics).</div>",
        unsafe_allow_html=True,
    )

    # ── Section B: Feature comparison ─────────────────────────────────────────
    _benchmark_feature_table()

    # ── Section C: Latency ────────────────────────────────────────────────────
    _section_header("Latency", "Average end-to-end response time")
    bench = _load_results_json("benchmark_results.json")
    c1, c2 = st.columns(2)
    a_lat = (ragas or {}).get("summary", {}).get("avg_latency_ms")
    n_lat = (naive or {}).get("summary", {}).get("avg_latency_ms")
    c1.metric("Naive RAG", f"{n_lat} ms" if n_lat else "n/a")
    if isinstance(a_lat, (int, float)) and isinstance(n_lat, (int, float)):
        c2.metric("Adaptive RAG", f"{a_lat} ms",
                  delta=f"{a_lat - n_lat:+d} ms (grading + rewrite cost)",
                  delta_color="inverse")
    else:
        c2.metric("Adaptive RAG", f"{a_lat} ms" if a_lat else "n/a")

    if bench and isinstance(bench.get("route_stats"), dict):
        rs = bench["route_stats"]
        chart_rows = {}
        for route in ("index", "general", "search", "overall"):
            stat = rs.get(route, {})
            if stat.get("count"):
                chart_rows[route] = stat.get("avg_ms", 0)
        if chart_rows:
            st.markdown("<div style='font-size:12px;color:#64748B;margin:10px 0 4px'>"
                        "Avg latency by route (ms)</div>", unsafe_allow_html=True)
            st.bar_chart(chart_rows, height=220)
    else:
        st.markdown(
            "<div style='font-size:11px;color:#94A3B8;margin-top:8px'>"
            "Per-route P50/P95/P99 breakdown: run "
            "<code>python evaluate/benchmark_latency.py</code> to populate.</div>",
            unsafe_allow_html=True,
        )

    # ── Section D: Routing accuracy ───────────────────────────────────────────
    _section_header("Routing Accuracy", "Tri-route classifier on a labelled set")
    routing = _load_results_json("routing_results.json")
    if routing:
        correct = routing.get("correct", 0)
        total = routing.get("total", 0)
        pct = routing.get("accuracy_pct", 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Routing Accuracy", f"{pct:.0f}%")
        c2.metric("Correct", f"{correct} / {total}")
        c3.metric("Index→Search fallbacks", routing.get("index_to_search_fallbacks", 0))
    else:
        _placeholder_card("Routing results not found. Run:",
                          "python evaluate/test_routing.py")


# ── TAB 3 — Retrieval Inspector ──────────────────────────────────────────────
def _inspector_card(step: str, title: str, body_html: str, accent: str = "#059669",
                    first: bool = False) -> None:
    # Vertical connector links each step to the previous one (a real sequence).
    connector = (
        "" if first else
        "<div style='width:2px;height:14px;background:#E3E0D8;margin:0 0 0 22px'></div>"
    )
    st.markdown(
        connector
        + f"<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-left:3px solid {accent};"
        "border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:0;"
        "box-shadow:none'>"
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'>"
        f"<span style='width:22px;height:22px;border-radius:50%;background:{accent};color:#FFFFFF;"
        "font-size:11px;font-weight:700;display:inline-flex;align-items:center;"
        f"justify-content:center;flex-shrink:0'>{step}</span>"
        f"<span style='font-size:13px;font-weight:700;color:#0F172A'>{html.escape(title)}</span></div>"
        f"{body_html}</div>",
        unsafe_allow_html=True,
    )


def render_inspector_tab() -> None:
    st.markdown(
        "<h1 style='font-size:22px;font-weight:800;color:#0F172A;margin:4px 0 2px'>"
        "🔍 Retrieval Inspector</h1>"
        "<div style='font-size:13px;color:#64748B;margin-bottom:6px'>"
        "Full pipeline trace of your most recent question.</div>",
        unsafe_allow_html=True,
    )
    trace = st.session_state.get("last_pipeline_trace")
    if not trace:
        _placeholder_card("Ask a question in the 💬 Chat tab to see its pipeline trace here.")
        return

    route = trace.get("route_taken", "unknown")
    route_meta = {
        "index": ("📄 INDEX", "#059669"),
        "search": ("🌐 SEARCH", "#6D28D9"),
        "general": ("⚡ GENERAL", "#92400E"),
    }.get(route, (route.upper(), "#64748B"))

    # Section A — overview
    _inspector_card(
        "①", "Query Overview",
        f"<div style='font-size:14px;color:#0F172A;font-weight:500;margin-bottom:8px'>"
        f"“{html.escape(trace.get('question',''))}”</div>"
        "<div style='display:flex;flex-wrap:wrap;gap:8px'>"
        f"<span style='background:{route_meta[1]}1a;color:{route_meta[1]};border:1px solid {route_meta[1]}40;"
        f"font-size:11px;font-weight:600;padding:3px 10px;border-radius:999px'>{route_meta[0]}</span>"
        f"<span style='background:#FAF9F6;border:1px solid #E3E0D8;color:#64748B;font-size:11px;"
        f"padding:3px 10px;border-radius:999px'>⏱ {trace.get('processing_ms',0)} ms</span>"
        f"<span style='background:#FAF9F6;border:1px solid #E3E0D8;color:#64748B;font-size:11px;"
        f"padding:3px 10px;border-radius:999px'>💰 ${trace.get('estimated_cost_usd',0):.5f}</span>"
        f"<span style='background:#FAF9F6;border:1px solid #E3E0D8;color:#64748B;font-size:11px;"
        f"padding:3px 10px;border-radius:999px'>🔁 {trace.get('loops_executed',0)} loop(s)</span>"
        "</div>",
        accent=route_meta[1],
        first=True,
    )

    # Section B — routing decision
    reason = {
        "index": "Document available and question appears document-specific.",
        "search": "Question needs real-time / current information.",
        "general": "No relevant document; answered from general knowledge.",
    }.get(route, "Classified by the routing LLM.")
    _inspector_card(
        "②", "Routing Decision",
        f"<div style='font-size:13px;color:#334155;line-height:1.6'>"
        f"Classified as <b style='color:{route_meta[1]}'>{html.escape(route_meta[0])}</b><br>"
        f"<span style='color:#64748B'>{html.escape(reason)}</span></div>",
        accent=route_meta[1],
    )

    # Section C — retrieval quality (index route)
    scores = trace.get("relevance_scores") or []
    if route == "index" and scores:
        bars = []
        for i, s in enumerate(scores):
            try:
                s = float(s)
            except (TypeError, ValueError):
                s = 0.0
            clr = "#059669" if s >= 0.70 else "#D97706" if s >= 0.45 else "#EF4444"
            status = "PASSED" if s >= 0.60 else "below threshold"
            icon = "✅" if s >= 0.60 else "❌"
            bars.append(
                "<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>"
                f"<span style='color:#94A3B8;font-size:11px;min-width:60px'>Chunk {i+1}</span>"
                "<div style='flex:1;background:#E3E0D8;height:6px;border-radius:999px;overflow:hidden'>"
                f"<div style='width:{int(s*100)}%;height:100%;background:{clr};border-radius:999px'></div></div>"
                f"<span style='color:{clr};font-size:11px;font-weight:600;min-width:130px;text-align:right'>"
                f"{s:.2f} {icon} {status}</span></div>"
            )
        _inspector_card(
            "③", "Retrieval Quality (threshold 0.60)",
            "".join(bars), accent="#059669",
        )

    # Section D — query rewrite history
    versions = trace.get("answer_versions") or []
    if len(versions) > 1:
        steps = []
        for idx, v in enumerate(versions):
            q = v.get("retrieval_quality", 0) or 0
            try:
                q = float(q)
            except (TypeError, ValueError):
                q = 0.0
            clr = "#059669" if q >= 0.60 else "#EF4444"
            is_last = idx == len(versions) - 1
            action = "GENERATE" if is_last else "REWRITE QUERY"
            steps.append(
                f"<div style='border-left:2px solid {'#059669' if is_last else '#E3E0D8'};"
                "padding-left:12px;margin-bottom:8px'>"
                f"<div style='font-size:12px;color:#475569;font-weight:600'>Attempt {v.get('loop_number', idx+1)}"
                f" <span style='color:#94A3B8;font-weight:400;font-family:monospace'>"
                f"“{html.escape(str(v.get('query_used','')))}”</span></div>"
                f"<div style='font-size:11px;margin-top:2px'>"
                f"<span style='color:{clr};font-weight:600'>score {q:.2f}</span>"
                f"<span style='color:#94A3B8'> → {action}</span></div></div>"
            )
        _inspector_card("④", "Query Rewrite History", "".join(steps), accent="#D97706")

    # Section E — source chunks
    chunks = trace.get("source_chunks") or []
    if chunks:
        st.markdown("<div style='font-size:13px;font-weight:700;color:#0F172A;margin:6px 0 4px'>"
                    "⑤ Retrieved Source Chunks</div>", unsafe_allow_html=True)
        render_source_chunks(chunks)

    # Section F — answer with inline grounding colors (the headline visual)
    grounding = trace.get("grounding") or {}
    g_results = grounding.get("results") or []
    answer_text = trace.get("answer", "")
    if g_results and answer_text:
        import re as _re
        sentence_labels = {r.get("sentence", ""): r.get("label") for r in g_results}
        colors = {
            "GROUNDED": ("#ECFDF5", "#059669", "✅"),
            "INFERRED": ("#FFFBEB", "#D97706", "⚠️"),
            "UNGROUNDED": ("#FEF2F2", "#DC2626", "❌"),
        }
        parts = []
        for sent in _re.split(r'(?<=[.!?])\s+', answer_text):
            s = sent.strip()
            if not s:
                continue
            label = sentence_labels.get(s)
            if label in colors:
                bg, bd, icon = colors[label]
                parts.append(
                    f"<span style='background:{bg};border-bottom:2px solid {bd};"
                    f"padding:1px 4px;border-radius:3px'>{html.escape(s)} "
                    f"<sup style='color:{bd};font-size:9px'>{icon}</sup></span> "
                )
            else:
                parts.append(f"<span style='color:#334155'>{html.escape(s)} </span>")
        _inspector_card(
            "⑥", "Answer — colour-coded by grounding",
            "<div style='font-size:13px;line-height:2.0'>" + "".join(parts) + "</div>"
            "<div style='font-size:10px;color:#94A3B8;margin-top:10px'>"
            "✅ grounded &nbsp;·&nbsp; ⚠️ inferred &nbsp;·&nbsp; ❌ unsupported "
            "&nbsp;·&nbsp; plain = not individually checked</div>",
            accent="#059669",
        )
    elif answer_text:
        # Search / general routes have no per-sentence grounding — show the
        # generated answer so the trace ends with a result, not empty space.
        note = ("Web-search route — answer composed from live results."
                if route == "search" else
                "General route — answered directly from the model, no retrieval.")
        _inspector_card(
            "⑥", "Generated Answer",
            f"<div style='font-size:13px;color:#334155;line-height:1.7'>{html.escape(answer_text)}</div>"
            f"<div style='font-size:10px;color:#94A3B8;margin-top:8px'>{note}</div>",
            accent=route_meta[1],
        )

    # Section G — grounding summary
    if grounding and not grounding.get("skipped") and grounding.get("summary"):
        summ = grounding["summary"]
        score = summ.get("trust_score", 0)
        level = summ.get("trust_level", "—")
        clr = {"HIGH": "#059669", "MODERATE": "#D97706", "LOW": "#EF4444"}.get(level, "#64748B")
        _inspector_card(
            "⑦", "Grounding Summary",
            f"<div style='font-size:14px;font-weight:700;color:{clr};margin-bottom:6px'>"
            f"Trust Score: {int(score*100)}% ({level})</div>"
            "<div style='display:flex;gap:16px;font-size:12px'>"
            f"<span style='color:#059669'>✅ {summ.get('grounded_count',0)} grounded</span>"
            f"<span style='color:#D97706'>⚠️ {summ.get('inferred_count',0)} inferred</span>"
            f"<span style='color:#EF4444'>❌ {summ.get('ungrounded_count',0)} ungrounded</span></div>",
            accent=clr,
        )


# ── TAB 4 — Grounding Lab ────────────────────────────────────────────────────
_LAB_LABEL_STYLE = {
    "GROUNDED": ("✅", "#059669", "#ECFDF5"),
    "INFERRED": ("⚠️", "#D97706", "#FFFBEB"),
    "UNGROUNDED": ("❌", "#DC2626", "#FEF2F2"),
}


def _lab_split_sentences(text: str):
    import re
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 15]


def render_grounding_lab_tab() -> None:
    st.markdown(
        "<h1 style='font-size:22px;font-weight:800;color:#0F172A;margin:4px 0 2px'>"
        "🧪 Grounding Lab</h1>"
        "<div style='font-size:13px;color:#64748B;margin-bottom:10px'>"
        "Ablation study: compare the <b>LLM judge</b> (accurate, slow, costs tokens) against "
        "<b>embedding cosine similarity</b> (instant, free) for hallucination grounding.</div>",
        unsafe_allow_html=True,
    )

    # Quick-start: prefer the last chat answer (primary) with an example fallback.
    trace = st.session_state.get("last_pipeline_trace")
    _qs1, _qs2 = st.columns([2, 1])
    if trace:
        if _qs1.button("↩ Use last chat answer + sources", key="lab_autofill",
                       use_container_width=True, type="primary"):
            st.session_state["lab_answer"] = trace.get("answer", "")
            st.session_state["lab_chunks"] = "\n---\n".join(
                c.get("text", "") for c in (trace.get("source_chunks") or [])
            )
            st.rerun()
        _example_col = _qs2
    else:
        st.caption("No chat yet — load an example to see the comparison instantly.")
        _example_col = _qs1
    if _example_col.button("🧪 Try an example", key="lab_example", use_container_width=True):
        # A deliberate mix: one grounded, one inferred, one ungrounded sentence.
        st.session_state["lab_answer"] = (
            "The system uses Groq LLaMA 3.3 70B at temperature zero. "
            "This makes its responses fast and deterministic. "
            "It also supports real-time voice translation in fifty languages."
        )
        st.session_state["lab_chunks"] = (
            "The system uses Groq LLaMA 3.3 70B at temperature zero for generation.\n---\n"
            "Vectors are stored in Qdrant using cosine similarity over 384-dimensional "
            "Cohere embeddings."
        )
        st.rerun()

    col_a, col_b = st.columns(2)
    with col_a:
        answer_text = st.text_area(
            "Answer text", key="lab_answer", height=200,
            placeholder="Paste the answer to analyze, one or more sentences...",
        )
    with col_b:
        chunks_text = st.text_area(
            "Source chunks (separate with --- or new lines)", key="lab_chunks", height=200,
            placeholder="Paste the retrieved source chunks here...",
        )

    run = st.button("⚡ Run Grounding Comparison", key="lab_run", type="primary")

    if run:
        if not answer_text.strip():
            st.warning("Paste some answer text above to analyze.")
            return
        chunks = [c.strip() for c in chunks_text.split("---") if c.strip()] or \
                 [c.strip() for c in chunks_text.split("\n") if c.strip()]
        if not chunks:
            st.warning("Without source chunks, every sentence will be UNGROUNDED.")
        sentences = _lab_split_sentences(answer_text)
        if not sentences:
            st.warning("No sentences long enough to analyze (min 15 chars).")
            return

        # ── Embedding-based grounding (fast) ──────────────────────────────────
        embed_result, embed_err = None, None
        try:
            import time as _t
            from src.services.grounding_checker_embedding import check_grounding_embedding
            _e0 = _t.time()
            embed_result = check_grounding_embedding(answer_text, chunks)
            embed_result["_wall_ms"] = int((_t.time() - _e0) * 1000)
        except Exception as exc:  # noqa: BLE001
            embed_err = str(exc)

        # ── LLM-based grounding (accurate) ────────────────────────────────────
        llm_result, llm_err = None, None
        try:
            import time as _t
            from langchain_groq import ChatGroq
            from src.core.config import settings
            from src.services.grounding_checker import check_answer_grounding
            _llm = ChatGroq(model=settings.GROQ_MODEL, temperature=0,
                            api_key=settings.GROQ_API_KEY, max_tokens=512)
            _l0 = _t.time()
            llm_result = check_answer_grounding(answer_text, chunks, _llm, max_sentences=20)
            if llm_result is not None:
                llm_result["_wall_ms"] = int((_t.time() - _l0) * 1000)
        except Exception as exc:  # noqa: BLE001
            llm_err = _friendly_error(str(exc))

        st.session_state.grounding_lab_result = {
            "sentences": sentences,
            "embed": embed_result, "embed_err": embed_err,
            "llm": llm_result, "llm_err": llm_err,
        }

    res = st.session_state.get("grounding_lab_result")
    if not res:
        st.markdown(
            "<div style='font-size:12px;color:#94A3B8;margin-top:12px'>"
            "Paste text (or auto-fill from your last chat) and run the comparison.</div>",
            unsafe_allow_html=True,
        )
        return

    embed, llm = res.get("embed"), res.get("llm")
    embed_map = {r["sentence"]: r for r in (embed.get("results") if embed else [])}
    llm_map = {r["sentence"]: r for r in (llm.get("results") if llm and not llm.get("skipped") else [])}

    # ── Side-by-side per-sentence ─────────────────────────────────────────────
    _section_header("Sentence-by-Sentence", "Disagreements are highlighted")
    h_l, h_r = st.columns(2)
    h_l.markdown("<div style='font-size:13px;font-weight:700;color:#059669'>"
                 "🤖 LLM Judge (Groq LLaMA 70B)</div>", unsafe_allow_html=True)
    h_r.markdown("<div style='font-size:13px;font-weight:700;color:#7C3AED'>"
                 "⚡ Embedding Similarity (Cohere)</div>", unsafe_allow_html=True)

    if res.get("llm_err"):
        h_l.markdown(f"<div style='font-size:12px;color:#DC2626'>{html.escape(res['llm_err'])}</div>",
                     unsafe_allow_html=True)
    if res.get("embed_err"):
        h_r.markdown(f"<div style='font-size:12px;color:#DC2626'>{html.escape(res['embed_err'])}</div>",
                     unsafe_allow_html=True)

    def _chip(label: str, extra: str = "") -> str:
        icon, clr, bg = _LAB_LABEL_STYLE.get(label, ("•", "#64748B", "#FAF9F6"))
        return (f"<span style='background:{bg};color:{clr};border:1px solid {clr}40;font-size:10px;"
                f"font-weight:700;padding:1px 7px;border-radius:6px'>{icon} {label}{extra}</span>")

    agree_count = 0
    comparable = 0
    for sent in res["sentences"]:
        lr = llm_map.get(sent)
        er = embed_map.get(sent)
        ll = lr["label"] if lr else None
        el = er["label"] if er else None
        disagree = ll and el and ll != el
        if ll and el:
            comparable += 1
            if ll == el:
                agree_count += 1
        row_bg = "#FEFCE8" if disagree else "#FFFFFF"
        sent_html = html.escape(sent[:240])
        c_l, c_r = st.columns(2)
        c_l.markdown(
            f"<div style='background:{row_bg};border:1px solid #E3E0D8;border-radius:8px;"
            f"padding:8px 12px;margin-bottom:6px;font-size:12px;color:#334155'>{sent_html}<br>"
            f"{_chip(ll) if ll else '<span style=color:#CBD5E1;font-size:10px>—</span>'}</div>",
            unsafe_allow_html=True,
        )
        sim_extra = f" · sim {er['similarity']:.2f}" if er else ""
        c_r.markdown(
            f"<div style='background:{row_bg};border:1px solid #E3E0D8;border-radius:8px;"
            f"padding:8px 12px;margin-bottom:6px;font-size:12px;color:#334155'>{sent_html}<br>"
            f"{_chip(el, sim_extra) if el else '<span style=color:#CBD5E1;font-size:10px>—</span>'}</div>",
            unsafe_allow_html=True,
        )

    # ── Comparison summary ────────────────────────────────────────────────────
    _section_header("Comparison Summary")
    agreement = (agree_count / comparable * 100) if comparable else 0.0
    llm_sum = (llm.get("summary") if llm and not llm.get("skipped") else None) or {}
    embed_sum = (embed.get("summary") if embed else None) or {}
    llm_lat = (llm or {}).get("_wall_ms", 0)
    embed_lat = (embed or {}).get("_wall_ms", embed_sum.get("total_latency_ms", 0))
    llm_calls = len(res["sentences"]) if llm_sum else 0

    def _row(metric, lval, eval_):
        return ("<tr>"
                f"<td style='padding:8px 14px;border-bottom:1px solid #F1F5F9;font-size:12px;"
                f"color:#0F172A;font-weight:600'>{metric}</td>"
                f"<td style='padding:8px 14px;border-bottom:1px solid #F1F5F9;font-size:12px;"
                f"color:#334155'>{lval}</td>"
                f"<td style='padding:8px 14px;border-bottom:1px solid #F1F5F9;font-size:12px;"
                f"color:#334155'>{eval_}</td></tr>")

    body = (
        _row("Trust Score",
             f"{int(llm_sum.get('trust_score',0)*100)}%" if llm_sum else "—",
             f"{int(embed_sum.get('trust_score',0)*100)}%" if embed_sum else "—")
        + _row("Trust Level", llm_sum.get("trust_level", "—"), embed_sum.get("trust_level", "—"))
        + _row("Latency", f"{llm_lat:,} ms", f"{embed_lat:,} ms")
        + _row("LLM Calls", f"{llm_calls}", "0")
        + _row("Cost", "≈ tokens used", "$0.00 (embeddings cached)")
    )
    st.markdown(
        "<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-radius:12px;overflow:hidden;"
        "box-shadow:none'>"
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr style='background:#FAF9F6'>"
        "<th style='padding:9px 14px;text-align:left;font-size:11px;color:#64748B;"
        "text-transform:uppercase'>Metric</th>"
        "<th style='padding:9px 14px;text-align:left;font-size:11px;color:#64748B;"
        "text-transform:uppercase'>🤖 LLM Judge</th>"
        "<th style='padding:9px 14px;text-align:left;font-size:11px;color:#64748B;"
        "text-transform:uppercase'>⚡ Embedding</th></tr>"
        f"{body}</table></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='margin-top:10px;font-size:13px;color:#0F172A'>"
        f"<b>Agreement: {agreement:.0f}%</b> "
        f"<span style='color:#64748B'>({agree_count}/{comparable} sentences classified the same)</span></div>",
        unsafe_allow_html=True,
    )

    # ── Disagreements ─────────────────────────────────────────────────────────
    disagreements = []
    for sent in res["sentences"]:
        lr, er = llm_map.get(sent), embed_map.get(sent)
        if lr and er and lr["label"] != er["label"]:
            disagreements.append((sent, lr["label"], er["label"], er.get("similarity", 0)))
    if disagreements:
        with st.expander(f"Disagreements ({len(disagreements)} sentence(s)) — the key interview talking point",
                         expanded=True):
            for sent, ll, el, sim in disagreements:
                st.markdown(
                    f"<div style='font-size:12px;color:#334155;margin-bottom:8px;line-height:1.6'>"
                    f"“{html.escape(sent[:200])}”<br>"
                    f"<span style='color:#059669'>LLM: {ll}</span> &nbsp;|&nbsp; "
                    f"<span style='color:#7C3AED'>Embedding: {el} (sim {sim:.2f})</span></div>",
                    unsafe_allow_html=True,
                )


# ── TAB 5 — System ────────────────────────────────────────────────────────────
def render_system_tab() -> None:
    st.markdown(
        "<h1 style='font-size:22px;font-weight:800;color:#0F172A;margin:4px 0 2px'>"
        "⚡ System</h1>"
        "<div style='font-size:13px;color:#64748B;margin-bottom:10px'>"
        "Live health, architecture, and runtime configuration.</div>",
        unsafe_allow_html=True,
    )

    # ── Section A: health dashboard ───────────────────────────────────────────
    _section_header("System Status")
    if st.button("🔄 Refresh status", key="sys_refresh"):
        st.session_state.last_health_check = 0.0
        st.rerun()
    backend_ok = check_backend()
    services = [
        ("FastAPI Backend", "🟢 ONLINE" if backend_ok else "🔴 OFFLINE",
         f":{API_BASE.rsplit(':', 1)[-1]}" if ":" in API_BASE else API_BASE,
         "#059669" if backend_ok else "#DC2626"),
        ("Qdrant Vector DB", "🟢 CONNECTED" if backend_ok else "⚪ UNKNOWN", "cosine · 384-dim",
         "#059669" if backend_ok else "#94A3B8"),
        ("Groq LLM", "🟢 ACTIVE" if backend_ok else "⚪ UNKNOWN", "llama-3.3-70b-versatile",
         "#059669" if backend_ok else "#94A3B8"),
        ("Cohere Embeddings", "🟢 ACTIVE" if backend_ok else "⚪ UNKNOWN", "embed-english-light-v3.0",
         "#059669" if backend_ok else "#94A3B8"),
        ("Tavily Web Search", "🟢 CONFIGURED" if backend_ok else "⚪ UNKNOWN", "fallback route",
         "#059669" if backend_ok else "#94A3B8"),
        ("MongoDB", "⚪ OPTIONAL", "chat history (graceful degradation)", "#94A3B8"),
    ]
    rows = "".join(
        "<tr>"
        f"<td style='padding:9px 14px;border-bottom:1px solid #F1F5F9;font-size:13px;"
        f"color:#0F172A;font-weight:600'>{html.escape(name)}</td>"
        f"<td style='padding:9px 14px;border-bottom:1px solid #F1F5F9;font-size:12px;"
        f"color:{clr};font-weight:600'>{status}</td>"
        f"<td style='padding:9px 14px;border-bottom:1px solid #F1F5F9;font-size:11px;"
        f"color:#94A3B8'>{html.escape(detail)}</td></tr>"
        for name, status, detail, clr in services
    )
    st.markdown(
        "<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-radius:12px;"
        "overflow:hidden;box-shadow:none'>"
        "<table style='width:100%;border-collapse:collapse'>" + rows + "</table></div>",
        unsafe_allow_html=True,
    )

    # ── Section B: architecture ───────────────────────────────────────────────
    _section_header("Architecture — Request Pipeline")

    def _abox(label, sub, c):
        return (
            f"<div style='background:#FFFFFF;border:1.5px solid {c}59;border-radius:10px;"
            "padding:9px 14px;text-align:center;min-width:118px'>"
            f"<div style='font-size:12px;font-weight:700;color:{c}'>{label}</div>"
            f"<div style='font-size:10px;color:#64748B;margin-top:2px'>{sub}</div></div>"
        )

    _ah = "<div style='color:#CBD5E1;font-size:18px;display:flex;align-items:center'>→</div>"
    _av = "<div style='text-align:center;color:#CBD5E1;font-size:16px;margin:4px 0'>↓</div>"
    _row = ("display:flex;gap:10px;justify-content:center;align-items:stretch;flex-wrap:wrap")
    st.markdown(
        "<div style='background:#FAF9F6;border:1px solid #E3E0D8;border-radius:12px;"
        "padding:20px 22px;box-shadow:none'>"
        f"<div style='{_row}'>"
        + _abox("Streamlit UI", "frontend", "#059669") + _ah
        + _abox("FastAPI :8080", "SSE stream", "#059669") + _ah
        + _abox("LangGraph", "StateGraph", "#7C3AED")
        + "</div>" + _av
        + "<div style='text-align:center;font-size:10px;color:#94A3B8;font-weight:600;"
          "text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px'>"
          "Route Question · Groq LLaMA 70B</div>"
        f"<div style='{_row}'>"
        + _abox("📄 Index", "Qdrant · retrieve + grade", "#059669")
        + _abox("⚡ General", "LLM direct", "#D97706")
        + _abox("🌐 Search", "Tavily web", "#7C3AED")
        + "</div>" + _av
        + f"<div style='{_row}'>" + _abox("Generate", "Groq LLaMA 70B", "#059669") + "</div>" + _av
        + f"<div style='{_row}'>"
        + _abox("Grounding · Gaps · Versioning", "post-processing", "#64748B")
        + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Section C: config + session stats ─────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        _section_header("Configuration")
        cfg = [
            ("Model", "llama-3.3-70b-versatile"),
            ("Embeddings", "cohere embed-english-light-v3.0 (384d)"),
            ("Relevance threshold", "0.60"),
            ("Low-match threshold", "0.30"),
            ("Max retry loops", "2"),
            ("Parent chunk", "1,500 chars"),
            ("Child chunk", "400 chars"),
        ]
        body = "".join(
            "<div style='display:flex;justify-content:space-between;padding:5px 0;"
            "border-bottom:1px solid #F1F5F9'>"
            f"<span style='font-size:12px;color:#64748B'>{html.escape(k)}</span>"
            f"<span style='font-size:12px;color:#0F172A;font-weight:600;font-family:monospace'>"
            f"{html.escape(v)}</span></div>"
            for k, v in cfg
        )
        st.markdown(
            "<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-radius:12px;"
            f"padding:14px 18px;box-shadow:none'>{body}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        _section_header("Session Stats")
        msgs = st.session_state.get("messages", [])
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        total_q = sum(1 for m in msgs if m["role"] == "user")
        by_route = {"index": 0, "search": 0, "general": 0}
        for m in assistant_msgs:
            rt = (m.get("meta") or {}).get("route_taken")
            if rt in by_route:
                by_route[rt] += 1
        stats = [
            ("Total queries", str(total_q)),
            ("Document (index)", str(by_route["index"])),
            ("Web (search)", str(by_route["search"])),
            ("General", str(by_route["general"])),
            ("Total cost", f"${st.session_state.get('total_cost', 0):.4f}"),
            ("Total tokens", f"{st.session_state.get('total_tokens', 0):,}"),
        ]
        body = "".join(
            "<div style='display:flex;justify-content:space-between;padding:5px 0;"
            "border-bottom:1px solid #F1F5F9'>"
            f"<span style='font-size:12px;color:#64748B'>{html.escape(k)}</span>"
            f"<span style='font-size:12px;color:#0F172A;font-weight:600'>{html.escape(v)}</span></div>"
            for k, v in stats
        )
        st.markdown(
            "<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-radius:12px;"
            f"padding:14px 18px;box-shadow:none'>{body}</div>",
            unsafe_allow_html=True,
        )

    # ── Section D: about ──────────────────────────────────────────────────────
    _section_header("About")
    st.markdown(
        "<div style='background:#FFFFFF;border:1px solid #E3E0D8;border-left:3px solid #059669;"
        "border-radius:0 6px 6px 0;padding:20px 24px;color:#3F434A'>"
        "<div style='font-size:15px;font-weight:700;color:#1C2128;margin-bottom:4px'>"
        "Adaptive RAG — Intelligent Document Intelligence</div>"
        "<div style='font-size:12px;color:#059669;margin-bottom:12px'>"
        "B.Tech CS (AI &amp; ML), Manipal University Jaipur</div>"
        "<div style='font-size:13px;line-height:1.7'>"
        "Production-grade AI engineering: adaptive routing eliminates unnecessary "
        "retrieval, sentence-level hallucination detection, quantitative benchmarking "
        "vs a naive baseline, and full-stack deployment "
        "(FastAPI + Streamlit + Qdrant + Docker).</div>"
        "<div style='font-size:12px;color:#9A9EA5;margin-top:12px'>"
        "github.com/Aditya0105singh/Adaptive-RAG---knowledge-retrieval</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB DISPATCH
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_inspector, tab_benchmark, tab_lab, tab_about = st.tabs([
    "💬  Chat",
    "🔍  Pipeline Inspector",
    "📊  Benchmarks",
    "🧪  Grounding Lab",
    "⚡  System",
])

with tab_chat:
    render_chat_tab()
with tab_inspector:
    render_inspector_tab()
with tab_benchmark:
    render_benchmark_results_tab()
with tab_lab:
    render_grounding_lab_tab()
with tab_about:
    render_system_tab()
