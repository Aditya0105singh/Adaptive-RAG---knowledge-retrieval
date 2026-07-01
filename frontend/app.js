// --- Sample document for demo ---
const DEMO_DOC_TEXT = `Adaptive RAG System — Complete Technical Research Report
=========================================================

ABSTRACT
This report presents Adaptive RAG, a production-grade tri-route Retrieval-Augmented Generation
system that intelligently routes queries across three specialized pipelines: document index
retrieval (INDEX), real-time web search (SEARCH), and parametric LLM knowledge (GENERAL).
The system achieves 100% routing accuracy on 25 test cases and outperforms naive RAG baselines
across all RAGAS evaluation dimensions. The core innovation is a self-correcting query rewrite
loop that automatically improves retrieval quality when initial chunk scores fall below threshold.

PROJECT OVERVIEW & MOTIVATION
Traditional RAG systems apply a single retrieval strategy to every query, regardless of whether
the question needs document context, live web data, or general knowledge. This one-size-fits-all
approach wastes tokens on irrelevant retrieval and reduces answer quality. Adaptive RAG solves
this by classifying each query before retrieval and routing it to the most appropriate pipeline.
The system was built as a research project demonstrating that adaptive routing significantly
improves faithfulness, context precision, and answer relevancy over naive RAG approaches.

KEY FINDINGS & PERFORMANCE METRICS
Faithfulness Score: 0.816 — evaluated using the RAGAS framework, representing a 7.5% improvement
over the naive RAG baseline score of 0.759. Faithfulness measures how well the generated answer
is supported by the retrieved context, with higher scores indicating fewer hallucinations.

Context Precision: 1.000 — a perfect score indicating that every retrieved chunk was relevant
to the query. This improved from the naive baseline of 0.834, a gain of 19.9%. Perfect context
precision means the system never wastes tokens on irrelevant retrieved content.

Routing Accuracy: 100% — all 25 hand-labeled test cases were correctly classified into the
appropriate route (INDEX, SEARCH, or GENERAL). The router uses LLaMA 3.1 8B Instant with a
structured prompt that considers document availability, query intent, and temporal requirements.

Answer Relevancy: 0.923 — measures how directly the generated answer addresses the question.
Improved from the naive baseline of 0.891, a gain of 3.6%.

Latency Benchmarks (9 queries, 3 per route):
- INDEX route: P50=4,910ms, P95=7,782ms, avg=5,813ms
- SEARCH route: P50=5,493ms, P95=5,513ms, avg=5,278ms
- GENERAL route: P50=8,376ms, P95=9,052ms, avg=7,737ms
- Overall: P50=5,513ms, P95=9,052ms, avg=6,276ms

TECHNICAL ARCHITECTURE

Routing Layer:
The router is a separate LLM call using LLaMA 3.1 8B Instant (faster and cheaper than the
answer model). It receives the user question, a flag indicating whether a document is uploaded,
and the conversation history. It outputs exactly one of: "index", "search", or "general".
The routing prompt includes few-shot examples for each route to improve consistency.

Parent-Child Chunking Strategy:
Documents are chunked at two levels: parent chunks of 1,500 characters provide broad context,
while child chunks of 400 characters enable precise semantic search. Retrieval is performed
on child chunks (finding the most semantically similar small piece), but the full parent chunk
is returned to the LLM for generation. This gives precise retrieval with rich context.

Chunk Grading and Self-Correcting Rewrite Loop:
After retrieval, each chunk is scored by the LLM for relevance to the query on a binary
pass/fail basis. If fewer than the required number of chunks pass, or if the best cosine
similarity score is below the 0.6 threshold, the system triggers a query rewrite. The rewriter
(LLaMA 3.3 70B) generates a more specific, keyword-rich version of the original query. The
improved query is then used for a second retrieval attempt. A maximum of 2 rewrite iterations
are allowed before the system falls back to web search. This loop is the core algorithmic
innovation that separates Adaptive RAG from standard RAG implementations.

Grounding Verification:
Every generated answer undergoes sentence-level grounding verification. Each sentence in the
response is classified as GROUNDED (directly supported by retrieved chunks), INFERRED
(reasonably implied but not explicitly stated), or UNGROUNDED (not supported by any source).
The verification uses embedding cosine similarity and LLM judgment in parallel. A Trust Score
(0-100%) is computed as: (grounded * 1.0 + inferred * 0.5) / total_sentences * 100. Answers
with trust scores below 60% display a warning to the user.

TECHNOLOGY STACK

LLM Infrastructure:
- Primary: Groq Cloud API with LLaMA 3.3 70B Versatile for answer generation
- Router: Groq with LLaMA 3.1 8B Instant (separate quota bucket, faster)
- Fallback chain: Groq → Gemini 2.0 Flash → Cerebras (gemma-4-31b)
- Multi-key rotation: 3 Groq API keys with automatic failover on rate limits

Vector Database:
- Qdrant Cloud (EU-Central-1 cluster) with cosine similarity search
- Per-session collections: each user session gets an isolated Qdrant collection
- Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- Cohere Reranker used for final chunk ordering before generation

Agentic Workflow:
- LangGraph StateGraph with nodes: route_question, retrieve, grade_documents,
  transform_query, web_search_node, generate_answer
- Conditional edges implement the rewrite loop and web search fallback
- MemorySaver checkpointer enables multi-turn conversation with state persistence

API & Streaming:
- FastAPI backend with Server-Sent Events (SSE) for real-time token streaming
- CORS configured for all origins to support multiple frontend deployments
- Session-based cost tracking with per-query token usage and USD cost estimation

DEPLOYMENT INFRASTRUCTURE
- Backend: Render.com (Docker container, auto-deploy from GitHub main branch)
- Frontend: Streamlit Cloud + Vercel (HTML/JS dashboard served as static files)
- Vector DB: Qdrant Cloud (managed, serverless pricing, EU data residency)
- Repository: github.com/Aditya0105singh/Adaptive-RAG---knowledge-retrieval

COMPARISON: ADAPTIVE RAG vs NAIVE RAG
Naive RAG always retrieves from the document index regardless of query type. For general
knowledge questions (e.g., "What is machine learning?"), naive RAG retrieves irrelevant
chunks and generates hallucinated answers citing those chunks. For current events questions
(e.g., "Who is the current OpenAI CEO?"), naive RAG either retrieves stale document content
or produces outdated answers. Adaptive RAG routes these queries correctly — to GENERAL
(using LLM knowledge) and SEARCH (using Tavily web search) respectively — producing accurate,
well-grounded answers without wasting document retrieval on inappropriate queries.

FUTURE WORK
Planned improvements include: multi-document cross-reference retrieval, GraphRAG integration
for entity-relationship-aware retrieval, streaming grounding verification (show trust score
as answer generates), and user feedback loop for routing accuracy improvement. The system
architecture supports adding new routes (e.g., SQL database query, API call) without
modifying the core graph structure.
`;

// --- API Base URL (auto-detects: localhost in dev, Render in production) ---
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? ''
    : 'https://adaptive-rag-knowledge-retrieval.onrender.com';

// --- State Management ---
function getOrCreateSessionId() {
    let sid = localStorage.getItem('adaptive_rag_session');
    if (!sid) {
        sid = crypto.randomUUID();
        localStorage.setItem('adaptive_rag_session', sid);
    }
    return sid;
}

const SESSION_ID = getOrCreateSessionId();
let uploadCount = 0;
let uploadedFileName = "";
let lastMetadata = null;
let benchmarksLoaded = false;

// --- Session Stats (Feature 3) ---
const _sessionStats = { total: 0, routes: { index: 0, search: 0, general: 0 }, trustScores: [], rewrites: 0, ungrounded: 0 };

function _updateSessionStats(metadata) {
    if (!metadata) return;
    _sessionStats.total++;
    const r = metadata.route_taken || 'general';
    if (_sessionStats.routes[r] !== undefined) _sessionStats.routes[r]++;
    if (metadata.loops_executed > 0) _sessionStats.rewrites += metadata.loops_executed;
    if (metadata.grounding?.trust_score !== undefined) _sessionStats.trustScores.push(metadata.grounding.trust_score);
    if (metadata.grounding?.summary?.ungrounded_count) _sessionStats.ungrounded += metadata.grounding.summary.ungrounded_count;
    _renderSessionStats();
}

function _renderSessionStats() {
    const bar = document.getElementById('session-stats-bar');
    if (!bar) return;
    if (_sessionStats.total < 2) return; // show after 2nd question
    const avgTrust = _sessionStats.trustScores.length
        ? Math.round(_sessionStats.trustScores.reduce((a,b)=>a+b,0) / _sessionStats.trustScores.length)
        : null;
    const routeParts = ['index','search','general']
        .filter(r => _sessionStats.routes[r] > 0)
        .map(r => {
            const colors = { index:'#0F766E', search:'#1D4ED8', general:'#475569' };
            return `<span style="color:${colors[r]};font-weight:700;">${_sessionStats.routes[r]} ${r.toUpperCase()}</span>`;
        }).join(' · ');
    bar.style.display = 'flex';
    bar.innerHTML = `
        <i class="ph-fill ph-chart-bar" style="color:var(--primary);font-size:13px;"></i>
        <span style="color:var(--text-muted);">Session:</span>
        <strong>${_sessionStats.total} queries</strong>
        <span style="color:var(--border);">|</span>
        ${routeParts}
        ${_sessionStats.rewrites ? `<span style="color:var(--border);">|</span><span style="color:#8B5CF6;font-weight:600;">${_sessionStats.rewrites} rewrite${_sessionStats.rewrites>1?'s':''}</span>` : ''}
        ${avgTrust !== null ? `<span style="color:var(--border);">|</span><span style="color:${avgTrust>=70?'#059669':'#D97706'};font-weight:600;">Avg trust ${avgTrust}%</span>` : ''}
        ${_sessionStats.ungrounded ? `<span style="color:var(--border);">|</span><span style="color:#DC2626;font-weight:600;">${_sessionStats.ungrounded} ungrounded</span>` : ''}
    `;
}

// --- Toast Notification System ---
function showToast(message, type = 'info', duration = 4000) {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed; top: 20px; right: 20px; z-index: 9999;
        padding: 12px 20px; border-radius: 8px; font-size: 13px;
        font-family: 'Inter', sans-serif; color: white; max-width: 400px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15); opacity: 0;
        transition: opacity 0.3s ease; pointer-events: none;
    `;
    if (type === 'success') toast.style.background = '#059669';
    else if (type === 'error') toast.style.background = '#DC2626';
    else toast.style.background = '#1E293B';
    toast.innerText = message;
    document.body.appendChild(toast);
    requestAnimationFrame(() => { toast.style.opacity = '1'; });
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// --- API Helper Functions ---
async function updateSessionCost() {
    try {
        const res = await fetch(`${API_BASE}/api/sessions/${SESSION_ID}/cost`);
        const data = await res.json();
        const costEl = document.querySelector('.session-cost-val');
        if (costEl && data.total_cost_usd !== undefined) {
            costEl.innerText = `$${data.total_cost_usd.toFixed(4)}`;
        }
    } catch (err) {
        console.error("Failed to fetch cost", err);
    }
}

async function updatePipelineInspector(metadata) {
    if (metadata) lastMetadata = metadata;

    const emptyEl = document.getElementById('pipeline-empty');
    const dataEl = document.getElementById('pipeline-data');

    if (!emptyEl || !dataEl) return;

    if (!lastMetadata) {
        try {
            const res = await fetch(`${API_BASE}/api/sessions/${SESSION_ID}/last-metadata`);
            const data = await res.json();
            if (data && !data.status) lastMetadata = data;
        } catch (err) {
            console.error("Failed to fetch pipeline metadata", err);
        }
    }

    if (!lastMetadata) {
        emptyEl.style.display = 'block';
        dataEl.style.display = 'none';
        return;
    }

    emptyEl.style.display = 'none';
    dataEl.style.display = 'block';

    const routeEl = document.getElementById('pi-route');
    const latencyEl = document.getElementById('pi-latency');
    const costEl = document.getElementById('pi-cost');
    const scoresEl = document.getElementById('pi-scores');
    const rawEl = document.getElementById('pi-raw-body');
    const toggleBtn = document.getElementById('pi-raw-toggle');

    if (routeEl) {
        const route = (lastMetadata.route_taken || 'UNKNOWN').toUpperCase();
        const colors = { INDEX: '#3B82F6', SEARCH: '#F59E0B', GENERAL: '#8B5CF6' };
        const c = colors[route] || '#64748B';
        routeEl.innerHTML = `<span style="background:${c}22; color:${c}; padding:4px 14px; border-radius:99px; font-size:14px; font-weight:700;">${route}</span>`;
    }
    if (latencyEl) latencyEl.innerText = lastMetadata.processing_ms ? `${lastMetadata.processing_ms}ms` : '—';
    if (costEl) costEl.innerText = lastMetadata.estimated_cost_usd !== undefined ? `$${lastMetadata.estimated_cost_usd.toFixed(5)}` : '—';

    if (scoresEl) {
        const scores = lastMetadata.relevance_scores;
        if (scores && scores.length > 0) {
            scoresEl.innerHTML = scores.map((score, i) => {
                const pct = (score * 100).toFixed(0);
                const color = score >= 0.8 ? '#059669' : score >= 0.6 ? '#F59E0B' : '#EF4444';
                return `<div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                    <span style="font-size:12px; color:var(--text-muted); width:56px; flex-shrink:0;">Chunk ${i + 1}</span>
                    <div style="flex:1; height:6px; background:#E2E8F0; border-radius:99px; overflow:hidden;">
                        <div style="height:100%; width:${pct}%; background:${color}; border-radius:99px;"></div>
                    </div>
                    <span style="font-family:monospace; font-size:12px; font-weight:700; color:${color}; width:36px; text-align:right;">${score.toFixed(2)}</span>
                </div>`;
            }).join('');
        } else {
            scoresEl.innerHTML = '<div style="font-size:13px; color:var(--text-muted);">No retrieval for this route (search or general path)</div>';
        }
    }

    if (rawEl) {
        rawEl.textContent = JSON.stringify(lastMetadata, null, 2);
    }
    if (toggleBtn) {
        toggleBtn.onclick = () => {
            const hidden = rawEl.style.display === 'none';
            rawEl.style.display = hidden ? 'block' : 'none';
            toggleBtn.textContent = hidden ? 'Hide' : 'Show';
        };
    }
}

async function checkSystemHealth() {
    const services = ['fastapi', 'qdrant', 'groq'];

    services.forEach(svc => {
        const card = document.getElementById(`svc-${svc}`);
        if (!card) return;
        card.querySelector('.svc-dot').style.background = '#F59E0B';
        card.querySelector('.svc-status').innerText = 'Checking...';
    });

    try {
        const res = await fetch(API_BASE + '/api/system-status');
        const data = await res.json();
        services.forEach(svc => {
            const card = document.getElementById(`svc-${svc}`);
            if (!card) return;
            const isOnline = data[svc] === 'online';
            card.querySelector('.svc-dot').style.background = isOnline ? '#10B981' : '#EF4444';
            card.querySelector('.svc-status').innerText = isOnline ? 'Online' : 'Offline';
        });
    } catch (err) {
        services.forEach(svc => {
            const card = document.getElementById(`svc-${svc}`);
            if (!card) return;
            card.querySelector('.svc-dot').style.background = '#EF4444';
            card.querySelector('.svc-status').innerText = 'Error — cannot reach backend';
        });
    }
}

async function loadBenchmarks() {
    if (benchmarksLoaded) return;

    const metricLabels = {
        faithfulness: 'Faithfulness',
        answer_relevancy: 'Answer Relevancy',
        context_precision: 'Context Precision'
    };

    function renderMetricBars(containerEl, metrics, compareMetrics, isAdaptive) {
        if (!containerEl || !metrics || Object.keys(metrics).length === 0) {
            if (containerEl) containerEl.innerHTML = '<div style="font-size:13px; color:#DC2626;">No data available.</div>';
            return;
        }
        let html = '';
        for (const [key, label] of Object.entries(metricLabels)) {
            const val = metrics[key] || 0;
            const cmp = compareMetrics ? (compareMetrics[key] || 0) : null;
            const delta = (isAdaptive && cmp !== null) ? val - cmp : null;
            const pct = Math.min(val * 100, 100).toFixed(1);
            const barColor = isAdaptive ? 'var(--primary)' : '#DC2626';
            const deltaHtml = delta !== null
                ? `<span style="font-size:11px; font-weight:600; margin-left:8px; color:${delta >= 0 ? '#059669' : '#DC2626'};">${delta >= 0 ? '+' : ''}${delta.toFixed(3)}</span>`
                : '';
            html += `<div style="margin-bottom:18px;">
                <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:7px;">
                    <span style="font-size:12px; font-weight:600; color:var(--text-main);">${label}</span>
                    <span style="font-family:monospace; font-weight:700; font-size:16px;">${val.toFixed(3)}${deltaHtml}</span>
                </div>
                <div style="height:7px; background:#E2E8F0; border-radius:99px; overflow:hidden;">
                    <div style="height:100%; width:${pct}%; background:${barColor}; border-radius:99px; transition:width 0.8s ease;"></div>
                </div>
            </div>`;
        }
        containerEl.innerHTML = html;
    }

    try {
        const res = await fetch(API_BASE + '/api/benchmarks');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        renderMetricBars(document.getElementById('naive-metrics'), data.naive, null, false);
        renderMetricBars(document.getElementById('adaptive-metrics'), data.adaptive, data.naive, true);
        benchmarksLoaded = true;
    } catch (err) {
        console.error('Failed to load benchmarks', err);
        ['naive-metrics', 'adaptive-metrics'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = `<div style="color:#DC2626; font-size:13px;">Failed to load: ${err.message}</div>`;
        });
    }
}

// --- Shared tab-switch helper (used by both top nav and rail) ---
function switchToTab(targetId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');

    // Show target
    const targetEl = document.getElementById(targetId);
    if (targetEl) targetEl.style.display = 'block';

    // Sync top nav active state
    document.querySelectorAll('.nav-tab').forEach(t => {
        t.classList.toggle('active', t.getAttribute('data-target') === targetId);
    });

    // Sync rail active state — first matching rail item wins
    document.querySelectorAll('.rail-item[data-target]').forEach(r => {
        r.classList.remove('active');
    });
    const matchingRail = document.querySelector(`.rail-item[data-target="${targetId}"]`);
    if (matchingRail) matchingRail.classList.add('active');

    // Side-effects per tab
    if (targetId === 'tab-pipeline') updatePipelineInspector();
    if (targetId === 'tab-system') checkSystemHealth();
    if (targetId === 'tab-benchmarks') loadBenchmarks();
}

// --- Main DOMContentLoaded ---
document.addEventListener('DOMContentLoaded', () => {
    console.log("Adaptive RAG Dashboard Loaded. Session:", SESSION_ID);
    updateSessionCost();
    updatePipelineInspector();

    // --- Top nav tab switching ---
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.getAttribute('data-target');
            if (targetId) switchToTab(targetId);
        });
    });

    // --- Rail icon switching ---
    document.querySelectorAll('.rail-item[data-target]').forEach(item => {
        item.addEventListener('click', () => {
            const targetId = item.getAttribute('data-target');
            if (targetId) switchToTab(targetId);
        });
    });

    // --- + (attach) button → file upload ---
    const attachBtn = document.querySelector('.btn-icon');
    if (attachBtn) {
        attachBtn.title = 'Attach file';
        attachBtn.addEventListener('click', () => fileInput.click());
    }

    // --- Hero "Get Started" upload box ---
    const heroUploadBox = document.getElementById('hero-upload-box');
    if (heroUploadBox) {
        heroUploadBox.addEventListener('click', () => fileInput.click());
    }

    // --- Suggested prompt chips ---
    document.querySelectorAll('.prompt-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const tab = chip.getAttribute('data-tab');
            if (tab) { switchToTab(tab); return; }
            const prompt = chip.getAttribute('data-prompt');
            if (chatInput && prompt) {
                chatInput.value = prompt;
                chatInput.focus();
            }
        });
    });

    // --- File Upload Logic ---
    const dropzone = document.querySelector('.dropzone');
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.pdf,.txt,.docx,.md,.csv';
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);

    const browseBtn = document.querySelector('.btn-outline');
    if (browseBtn) {
        browseBtn.addEventListener('click', (e) => {
            e.preventDefault();
            fileInput.click();
        });
    }

    if (dropzone) {
        dropzone.addEventListener('click', (e) => {
            if (e.target === browseBtn) return;
            fileInput.click();
        });

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--primary)';
            dropzone.style.background = 'white';
        });

        dropzone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--border)';
            dropzone.style.background = '#FAFAFA';
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--border)';
            dropzone.style.background = '#FAFAFA';
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });
    }

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        if (uploadCount >= 5) {
            showToast("Maximum 5 files allowed.", "error");
            return;
        }

        const dropzoneTitle = document.querySelector('.dropzone-title');
        const dropzoneDesc = document.querySelector('.dropzone-desc');
        const sidebarTitles = document.querySelectorAll('.section-title');
        let docCounter = null;
        sidebarTitles.forEach(el => {
            if (el.textContent.includes('YOUR DOCUMENTS')) docCounter = el;
        });

        if (dropzoneTitle) dropzoneTitle.innerText = "Uploading...";
        if (dropzoneDesc) dropzoneDesc.innerText = file.name;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('session_id', SESSION_ID);

        try {
            const res = await fetch(API_BASE + '/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (res.ok) {
                uploadCount++;
                uploadedFileName = file.name;
                if (docCounter) docCounter.innerText = `YOUR DOCUMENTS (${uploadCount}/5)`;
                if (dropzoneTitle) {
                    dropzoneTitle.innerText = "Drop another file or browse";
                    dropzoneTitle.style.color = "";
                }

                // Permanent file card in sidebar (replaces flying toast)
                const filesList = document.getElementById('uploaded-files-list');
                const footerText = document.getElementById('dropzone-footer-text');
                if (filesList) {
                    filesList.style.display = 'block';
                    const sizeKB = (file.size / 1024).toFixed(0);
                    const parentCount = data.parent_count || 0;
                    const childCount  = data.child_count  || 0;
                    const card = document.createElement('div');
                    card.style.cssText = `
                        background:#F0FDF4; border:1px solid #6EE7B7; border-radius:10px;
                        padding:10px 12px; margin-bottom:6px; font-size:12px;`;
                    card.innerHTML = `
                        <div style="display:flex; align-items:center; gap:7px; margin-bottom:8px;">
                          <i class="ph-fill ph-check-circle" style="color:#059669; font-size:16px; flex-shrink:0;"></i>
                          <span style="font-weight:700; color:#065F46; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1;">${escHtml(file.name)}</span>
                          <span style="color:#6B7280; font-size:10px; flex-shrink:0;">${sizeKB} KB</span>
                        </div>
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
                          <div style="background:white; border-radius:6px; padding:6px 8px; text-align:center; border:1px solid #D1FAE5;">
                            <div style="font-size:16px; font-weight:800; color:#059669; font-family:monospace;">${parentCount}</div>
                            <div style="font-size:10px; color:#6B7280; text-transform:uppercase; letter-spacing:0.04em;">Parent Chunks</div>
                          </div>
                          <div style="background:white; border-radius:6px; padding:6px 8px; text-align:center; border:1px solid #D1FAE5;">
                            <div style="font-size:16px; font-weight:800; color:#059669; font-family:monospace;">${childCount}</div>
                            <div style="font-size:10px; color:#6B7280; text-transform:uppercase; letter-spacing:0.04em;">Child Chunks</div>
                          </div>
                        </div>
                        <div style="margin-top:7px; font-size:10px; color:#6B7280; text-align:center;">
                          Ready — try a suggested question below
                        </div>`;

                    // Inject doc-specific question pills into the suggestions list
                    const promptsEl = document.getElementById('suggested-prompts');
                    if (promptsEl) {
                        promptsEl.innerHTML = `
                          <div style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:10px; text-align:center;">Questions about your document</div>
                          <div style="display:flex; flex-direction:column; gap:8px;">
                            <button class="prompt-chip" data-prompt="What are the key findings and performance metrics of this Adaptive RAG system?">
                              <span class="chip-icon"><i class="ph ph-atom"></i></span>
                              <span>What are the key findings & metrics?</span>
                            </button>
                            <button class="prompt-chip" data-prompt="Explain the parent-child chunking strategy and the self-correcting rewrite loop in detail.">
                              <span class="chip-icon"><i class="ph ph-files"></i></span>
                              <span>Explain chunking & rewrite loop</span>
                            </button>
                            <button class="prompt-chip" data-prompt="How does the grounding verification work and how is the Trust Score calculated?">
                              <span class="chip-icon"><i class="ph ph-shield-check"></i></span>
                              <span>How does grounding & Trust Score work?</span>
                            </button>
                            <button class="prompt-chip" data-prompt="Compare Adaptive RAG vs Naive RAG — what specific improvements were achieved in RAGAS scores?">
                              <span class="chip-icon"><i class="ph ph-arrows-left-right"></i></span>
                              <span>Adaptive RAG vs Naive RAG comparison</span>
                            </button>
                            <button class="prompt-chip" data-prompt="Who is the current CEO of OpenAI and what are their recent AI announcements?">
                              <span class="chip-icon" style="color:#2563EB; background:#EFF6FF;"><i class="ph ph-globe"></i></span>
                              <span>Latest OpenAI news <span style="font-size:11px;color:#94A3B8;">(web search)</span></span>
                            </button>
                            <button class="prompt-chip" data-prompt="What is cosine similarity and why is it used in vector search for RAG systems?">
                              <span class="chip-icon" style="color:#7C3AED; background:#F5F3FF;"><i class="ph ph-brain"></i></span>
                              <span>What is cosine similarity? <span style="font-size:11px;color:#94A3B8;">(general)</span></span>
                            </button>
                          </div>`;
                        // Re-attach click handlers for new cards
                        promptsEl.querySelectorAll('.prompt-chip[data-prompt]').forEach(btn => {
                            btn.addEventListener('click', () => {
                                if (chatInput) { chatInput.value = btn.dataset.prompt; chatInput.focus(); sendMessage(); }
                            });
                        });
                    }
                    filesList.appendChild(card);
                }
                if (footerText) footerText.style.display = 'none';

                const modeTitle = document.querySelector('.mode-title');
                const modeDesc = document.querySelector('.mode-desc');
                if (modeTitle) modeTitle.innerText = "Document Mode";
                if (modeDesc) modeDesc.innerText = "Querying uploaded documents";
            } else {
                const errMsg = data.detail?.message || data.detail || "Unknown error";
                showToast("Upload failed: " + errMsg, "error");
                if (dropzoneTitle) dropzoneTitle.innerText = "Drag and drop files here";
                if (dropzoneDesc) dropzoneDesc.innerText = "Limit 50MB per file - PDF, TXT, DOCX, MD, CSV";
            }
        } catch (err) {
            console.error("Upload error", err);
            showToast("Network error during upload.", "error");
            if (dropzoneTitle) dropzoneTitle.innerText = "Drag and drop files here";
            if (dropzoneDesc) dropzoneDesc.innerText = "Limit 50MB per file - PDF, TXT, DOCX, MD, CSV";
        }
    }

    // --- Try Sample Doc button ---
    const sampleDocBtn = document.getElementById('sample-doc-btn');
    if (sampleDocBtn) {
        sampleDocBtn.addEventListener('click', async () => {
            if (uploadCount >= 5) { showToast("Maximum 5 files reached.", "error"); return; }

            const label = document.getElementById('sample-doc-btn-label');
            sampleDocBtn.disabled = true;
            if (label) label.textContent = 'Uploading…';

            const blob = new Blob([DEMO_DOC_TEXT], { type: 'text/plain' });
            const demoFile = new File([blob], 'adaptive_rag_research.txt', { type: 'text/plain' });
            await handleFileUpload(demoFile);

            sampleDocBtn.style.display = 'none'; // hide after upload — doc is now in sidebar
        });
    }

    // --- New Chat button ---
    const newChatBtn = document.getElementById('new-chat-btn');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', () => {
            // Remove all answer wrappers (keep hero + grid)
            const responseCards = document.querySelectorAll('#tab-chat .content-max > .answer-wrapper');
            responseCards.forEach(c => c.remove());

            // Restore landing
            const defaultGrid = document.querySelector('#tab-chat .grid-2');
            const defaultHero = document.querySelector('#tab-chat .hero-card');
            const suggestedPrompts = document.getElementById('suggested-prompts');
            const demoGuide = document.getElementById('demo-guide-card');
            if (defaultGrid) defaultGrid.style.display = '';
            if (defaultHero) defaultHero.style.display = '';
            if (suggestedPrompts) suggestedPrompts.style.display = 'flex';
            if (demoGuide) demoGuide.style.display = '';

            // Hide the button again
            newChatBtn.style.display = 'none';

            if (chatInput) { chatInput.value = ''; chatInput.focus(); }
        });
    }

    // --- Chat Logic & SSE Streaming ---
    const chatInput = document.querySelector('.chat-input');
    const sendBtn = document.querySelector('.btn-send');
    const contentMax = document.querySelector('#tab-chat .content-max');

    // --- Utility ---
    function escHtml(s) {
        return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    // --- 3-level progressive disclosure answer card ---
    function createAnswerCard(query) {
        const wrapper = document.createElement('div');
        wrapper.style.marginBottom = '24px';

        const card = document.createElement('div');
        card.className = 'card response-card';
        card.style.position = 'relative';
        card.innerHTML = `
            <!-- Blinking NEW dot — shown on done_all, hidden on click or after 8s -->
            <div class="ai-new-dot" title="New AI response" style="
              display:none; position:absolute; top:-8px; right:-8px; z-index:50;
              align-items:center; gap:5px;">
              <span style="
                width:13px; height:13px; border-radius:50%; flex-shrink:0;
                background:#10B981; box-shadow:0 0 0 0 rgba(16,185,129,0.6);
                animation:ai-dot-pulse 1.2s ease-in-out infinite;
                display:inline-block;"></span>
              <span style="background:#10B981; color:white; font-size:10px; font-weight:800;
                padding:2px 7px; border-radius:99px; letter-spacing:0.04em;">NEW</span>
            </div>

            <div class="query-bubble">
                <i class="ph ph-user-circle" style="font-size:16px; flex-shrink:0; margin-top:1px;"></i>
                <span>${escHtml(query)}</span>
            </div>

            <!-- Level 1: answer text -->
            <div class="answer-body"><span class="live-answer" style="white-space:pre-wrap;"></span></div>

            <!-- Level 1: badge row (shown after metadata) -->
            <div class="badge-row" style="display:none;">
                <span class="badge-route">—</span>
                <span class="badge-mono badge-latency"></span>
                <span class="badge-mono badge-cost"></span>
                <span class="badge-trust">—</span>
                <span class="badge-loops" style="display:none; font-size:11px; color:var(--text-muted);"></span>
            </div>

            <!-- Level 1: sources -->
            <details class="sources-details" style="display:none; margin-bottom:8px;">
                <summary><i class="ph ph-files" style="font-size:13px;"></i> <span class="sources-count">Sources</span></summary>
                <div class="sources-list" style="margin-top:8px;"></div>
            </details>

            <!-- Level 2: View AI reasoning -->
            <details class="reasoning-details" open>
                <summary class="reasoning-summary">
                    <i class="ph ph-caret-right" style="font-size:12px; transition:transform 0.2s;" class="caret-icon"></i>
                    View AI reasoning
                </summary>
                <div class="reasoning-content" style="padding-top:8px;">

                    <!-- Pipeline trace (shown while streaming, hidden after) -->
                    <div class="pipeline-trace"></div>

                    <!-- Retrieval quality bars -->
                    <div class="retrieval-quality" style="display:none; margin-bottom:16px;">
                        <div style="font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-muted); margin-bottom:8px;">Retrieval Quality</div>
                        <div class="chunk-bars"></div>
                    </div>

                    <!-- Verification summary -->
                    <div class="verification-summary" style="display:none; margin-bottom:12px;"></div>

                    <!-- Level 3a: sentence breakdown -->
                    <details class="breakdown-details" style="display:none;">
                        <summary class="breakdown-summary">
                            <i class="ph ph-caret-right" style="font-size:12px;"></i>
                            See sentence breakdown
                        </summary>
                        <div style="padding-top:10px;">
                            <div style="display:flex; gap:8px; margin-bottom:12px;">
                                <button class="filter-btn active" data-filter="flagged">Show flagged only</button>
                                <button class="filter-btn" data-filter="all">Show all</button>
                            </div>
                            <div class="sentence-list"></div>
                            <!-- Grounding comparison (populated async) -->
                            <div class="comparison-section" style="display:none; margin-top:20px; padding-top:16px; border-top:1px solid var(--border);"></div>
                        </div>
                    </details>

                    <!-- Level 3b: raw metadata -->
                    <details class="metadata-details" style="display:none;">
                        <summary class="metadata-summary">
                            <i class="ph ph-caret-right" style="font-size:12px;"></i>
                            Raw metadata
                        </summary>
                        <pre class="metadata-json" style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#334155; overflow-x:auto; margin-top:8px; background:#F8FAFC; padding:12px; border-radius:6px; white-space:pre-wrap;"></pre>
                    </details>
                </div>
            </details>
        `;
        wrapper.appendChild(card);

        // Knowledge gap card (separate, below)
        const gapCard = document.createElement('div');
        gapCard.className = 'gap-card';
        gapCard.style.display = 'none';
        wrapper.appendChild(gapCard);

        // Filter toggle wiring
        card.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                card.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const mode = btn.getAttribute('data-filter');
                card.querySelectorAll('.sentence-row').forEach(row => {
                    const label = row.getAttribute('data-label');
                    if (mode === 'all' || label === 'INFERRED' || label === 'UNGROUNDED') {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });
            });
        });

        return {
            element: wrapper,
            card: card,
            liveAnswer: card.querySelector('.live-answer'),
            badgeRow: card.querySelector('.badge-row'),
            badgeRoute: card.querySelector('.badge-route'),
            badgeLatency: card.querySelector('.badge-latency'),
            badgeCost: card.querySelector('.badge-cost'),
            badgeTrust: card.querySelector('.badge-trust'),
            badgeLoops: card.querySelector('.badge-loops'),
            sourcesDetails: card.querySelector('.sources-details'),
            sourcesList: card.querySelector('.sources-list'),
            sourcesCount: card.querySelector('.sources-count'),
            reasoningDetails: card.querySelector('.reasoning-details'),
            pipelineTrace: card.querySelector('.pipeline-trace'),
            retrievalQuality: card.querySelector('.retrieval-quality'),
            chunkBars: card.querySelector('.chunk-bars'),
            verificationSummary: card.querySelector('.verification-summary'),
            breakdownDetails: card.querySelector('.breakdown-details'),
            sentenceList: card.querySelector('.sentence-list'),
            comparisonSection: card.querySelector('.comparison-section'),
            metadataDetails: card.querySelector('.metadata-details'),
            metadataJson: card.querySelector('.metadata-json'),
            gapCard: gapCard,
        };

        // Hide NEW badge when user clicks anywhere on the card
        card.addEventListener('click', () => {
            const dot = card.querySelector('.ai-new-dot');
            if (dot) dot.style.display = 'none';
        }, { once: false });
    }

    // Kept for Pipeline Inspector "How it works" static example
    function createStepHTML(num, iconHTML, title, desc, status = 'active') {
        const bgColor = status === 'done' ? 'background:var(--primary); color:white;' :
                         status === 'error' ? 'background:#DC2626; color:white;' : '';
        const checkIcon = status === 'done' ? '<i class="ph-bold ph-check"></i>' : num;
        return `
            <div class="step">
                <div class="step-num" style="${bgColor}">${checkIcon}</div>
                <div class="step-content">
                    <div class="step-title">${iconHTML} ${title}</div>
                    <div class="step-desc">${desc}</div>
                </div>
            </div>
        `;
    }

    function addPipelineStep(traceEl, iconHTML, title, desc, status = 'active') {
        const div = document.createElement('div');
        div.innerHTML = createStepHTML('•', iconHTML, title, desc, status);
        traceEl.appendChild(div.firstElementChild);
    }

    // --- Fill answer card from metadata event ---
    function fillAnswerCard(ac, evt, answer) {
        // Render markdown answer
        if (ac.liveAnswer) {
            if (typeof marked !== 'undefined') {
                ac.liveAnswer.innerHTML = marked.parse(answer);
                ac.liveAnswer.style.whiteSpace = '';
            } else {
                ac.liveAnswer.innerText = answer;
            }
        }

        // Badge row
        const route = (evt.route_taken || 'general').toLowerCase();
        const routeLabel = { index: 'Document', search: 'Web Search', general: 'General' }[route] || route;
        ac.badgeRoute.textContent = routeLabel;
        ac.badgeRoute.className = `badge-route route-${route}`;

        const latMs = evt.processing_ms || 0;
        ac.badgeLatency.textContent = latMs >= 1000 ? `${(latMs/1000).toFixed(1)}s` : `${latMs}ms`;
        ac.badgeCost.textContent = `$${(evt.estimated_cost_usd || 0).toFixed(4)}`;

        const grounding = evt.grounding;
        const summary = grounding && !grounding.skipped ? grounding.summary : null;
        if (summary) {
            const pct = Math.round(summary.trust_score * 100);
            const lvl = summary.trust_level || 'LOW';
            ac.badgeTrust.textContent = `${lvl} · ${pct}%`;
            const trustClass = { HIGH: 'trust-high', MODERATE: 'trust-moderate', LOW: 'trust-low' }[lvl] || 'trust-low';
            ac.badgeTrust.className = `badge-trust ${trustClass}`;
        } else {
            ac.badgeTrust.style.display = 'none';
        }

        const loops = evt.loops_executed || 0;
        if (loops > 1) {
            ac.badgeLoops.textContent = `${loops} loops`;
            ac.badgeLoops.style.display = '';
        }
        ac.badgeRow.style.display = 'flex';

        // Source citations
        const chunks = evt.source_chunks || [];
        if (chunks.length > 0) {
            ac.sourcesCount.textContent = `Sources (${chunks.length})`;
            chunks.forEach((c, i) => {
                const chip = document.createElement('div');
                chip.className = 'source-chip';
                const fname = c.filename || `Chunk ${i+1}`;
                const preview = (c.text || '').slice(0, 180).replace(/</g,'&lt;');
                chip.innerHTML = `
                    <i class="ph ph-file-text" style="font-size:14px; color:var(--text-muted); flex-shrink:0; margin-top:2px;"></i>
                    <div>
                        <div style="font-size:11px; font-weight:700; color:var(--text-main); margin-bottom:3px;">${escHtml(fname)}</div>
                        <div class="source-chip-text" style="font-size:11px; color:var(--text-muted);">${preview}${c.text && c.text.length > 180 ? '…' : ''}</div>
                    </div>`;
                ac.sourcesList.appendChild(chip);
            });
            ac.sourcesDetails.style.display = '';
        }

        // Retrieval quality bars
        const scores = evt.relevance_scores || [];
        if (scores.length > 0 && route === 'index') {
            const topScores = scores.slice(0, 4);
            topScores.forEach((score, i) => {
                const pct = Math.round(score * 100);
                const color = pct >= 70 ? '#059669' : pct >= 45 ? '#D97706' : '#DC2626';
                const qual = pct >= 70 ? 'Strong' : pct >= 45 ? 'Moderate' : 'Weak';
                const row = document.createElement('div');
                row.className = 'chunk-bar-row';
                row.innerHTML = `
                    <span class="chunk-bar-label">Chunk ${i+1}</span>
                    <div class="chunk-bar-track"><div class="chunk-bar-fill" style="width:${pct}%;background:${color};"></div></div>
                    <span class="chunk-bar-score" style="color:${color};">${(score).toFixed(2)}</span>
                    <span class="chunk-bar-qual" style="color:${color}; font-size:11px;">${qual}</span>`;
                ac.chunkBars.appendChild(row);
            });
            if (scores.length > 4) {
                const more = document.createElement('div');
                more.style.cssText = 'font-size:11px; color:var(--text-muted); padding:4px 0;';
                more.textContent = `and ${scores.length - 4} more chunks`;
                ac.chunkBars.appendChild(more);
            }
            ac.retrievalQuality.style.display = '';
        }

        // Verification summary + trust bar
        if (summary) {
            const pct = Math.round(summary.trust_score * 100);
            const barColor = pct >= 70 ? '#059669' : pct >= 40 ? '#D97706' : '#DC2626';
            ac.verificationSummary.innerHTML = `
                <span class="verification-line">
                    <span class="v-grounded">${summary.grounded_count} verified</span> ·
                    <span class="v-inferred">${summary.inferred_count} inferred</span> ·
                    <span class="v-ungrounded">${summary.ungrounded_count} unsupported</span>
                </span>
                <div class="trust-bar-track" style="margin-top:6px;">
                    <div class="trust-bar-fill" style="width:${pct}%; background:${barColor};"></div>
                </div>
                <div style="font-size:11px; color:var(--text-muted); margin-top:4px; font-family:'IBM Plex Mono',monospace;">
                    ${pct}% of sentences directly supported by document (${summary.trust_level})
                </div>`;
            ac.verificationSummary.style.display = '';

            // Sentence breakdown
            const results = grounding.results || [];
            if (results.length > 0) {
                results.forEach(r => {
                    const label = r.label || 'INFERRED';
                    const dotClass = { GROUNDED: 'dot-grounded', INFERRED: 'dot-inferred', UNGROUNDED: 'dot-ungrounded' }[label] || 'dot-inferred';
                    const labelClass = { GROUNDED: 'label-grounded', INFERRED: 'label-inferred', UNGROUNDED: 'label-ungrounded' }[label] || 'label-inferred';
                    const row = document.createElement('div');
                    row.className = 'sentence-row';
                    row.setAttribute('data-label', label);
                    // Default: hide GROUNDED rows (show flagged only)
                    if (label === 'GROUNDED') row.style.display = 'none';
                    row.innerHTML = `
                        <span class="sentence-dot ${dotClass}"></span>
                        <span class="sentence-text">${escHtml(r.sentence || '')}</span>
                        <span class="sentence-label ${labelClass}">${label}</span>`;
                    ac.sentenceList.appendChild(row);
                });
                ac.breakdownDetails.style.display = '';
            }
        }

        // Knowledge gap card
        const gaps = evt.knowledge_gaps;
        if (gaps && gaps.length > 0) {
            const gapItems = gaps.map(g => `
                <div class="gap-item">
                    <i class="ph ph-arrow-right" style="color:#3A6EA5; flex-shrink:0; font-size:13px; margin-top:2px;"></i>
                    <span>${escHtml(g)}</span>
                </div>`).join('');
            ac.gapCard.innerHTML = `
                <details>
                    <summary>
                        <i class="ph ph-lightbulb" style="font-size:16px;"></i>
                        ${gaps.length} knowledge gap${gaps.length > 1 ? 's' : ''} found
                        <span style="font-weight:400; margin-left:4px;">— click to see what's missing</span>
                        <i class="ph ph-caret-down" style="margin-left:auto; font-size:13px;"></i>
                    </summary>
                    <div class="gap-card-body">
                        ${gapItems}
                        <div style="margin-top:10px; font-size:12px; color:var(--text-muted);">Try uploading relevant documents to fill these gaps.</div>
                    </div>
                </details>`;
            ac.gapCard.style.display = '';
        }

        // Answer refined badge (inside reasoning, below pipeline trace)
        if (evt.answer_improvement) {
            const ref = document.createElement('div');
            ref.style.cssText = 'margin-top:8px; padding:8px 12px; background:#F0FDF4; border:1px solid #6EE7B7; border-radius:8px; font-size:12px; color:#065F46; display:flex; gap:8px; align-items:flex-start;';
            ref.innerHTML = `<i class="ph-fill ph-arrows-clockwise" style="color:#059669; flex-shrink:0; margin-top:1px;"></i><span><strong>Answer refined:</strong> ${escHtml(evt.answer_improvement)}</span>`;
            ac.pipelineTrace.after(ref);
        }

        // Raw metadata
        ac.metadataJson.textContent = JSON.stringify(evt, null, 2);
        ac.metadataDetails.style.display = '';

        // Close reasoning panel now that answer is ready
        ac.reasoningDetails.open = false;
    }

    // --- Grounding comparison (async, after done_all) ---
    async function fetchGroundingComparison(ac, evt) {
        const grounding = evt.grounding;
        if (!grounding || grounding.skipped || !grounding.results || grounding.results.length === 0) return;
        const chunks = (evt.source_chunks || []).map(c => c.text).filter(Boolean);
        if (chunks.length === 0) return;

        ac.comparisonSection.style.display = '';
        ac.comparisonSection.innerHTML = `<div style="font-size:12px; color:var(--text-muted); padding:8px 0;"><i class="ph ph-spinner"></i> Running embedding comparison...</div>`;

        try {
            const res = await fetch(API_BASE + '/api/grounding/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    answer: evt.answer || '',
                    chunks: chunks,
                    llm_results: grounding.results,
                })
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            const rows = data.comparison.map(c => {
                const llmClass = { GROUNDED: 'label-grounded', INFERRED: 'label-inferred', UNGROUNDED: 'label-ungrounded' }[c.llm] || '';
                const embClass = { GROUNDED: 'label-grounded', INFERRED: 'label-inferred', UNGROUNDED: 'label-ungrounded' }[c.embedding] || '';
                const short = escHtml((c.sentence || '').slice(0, 60)) + (c.sentence.length > 60 ? '…' : '');
                return `<tr>
                    <td style="color:var(--text-muted);">${short}</td>
                    <td><span class="sentence-label ${llmClass}">${c.llm}</span></td>
                    <td><span class="sentence-label ${embClass}">${c.embedding}</span></td>
                    <td style="font-family:'IBM Plex Mono',monospace; font-size:11px;">${c.similarity.toFixed(2)}</td>
                    <td class="${c.match ? 'match-yes' : 'match-no'}">${c.match ? '✓' : '✗'}</td>
                </tr>`;
            }).join('');

            const agePct = Math.round((data.agreement_rate || 0) * 100);
            const llmMs = (grounding.summary && grounding.summary.total_latency_ms) || '?';
            const embMs = data.embedding_latency_ms || 0;

            ac.comparisonSection.innerHTML = `
                <div style="font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-muted); margin-bottom:8px;">Method Comparison — LLM Judge vs Embedding Cosine</div>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Sentence</th><th>LLM Judge</th><th>Embedding</th>
                            <th style="font-family:'IBM Plex Mono',monospace;">Sim</th><th>Match?</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
                <div style="margin-top:10px; font-size:11px; color:var(--text-muted); font-family:'IBM Plex Mono',monospace;">
                    Agreement: <strong style="color:var(--text-main);">${agePct}%</strong>
                    (${data.agreement_count}/${data.total_sentences}) ·
                    LLM latency: <strong>${llmMs}ms</strong> ·
                    Embedding: <strong>${embMs}ms</strong> ·
                    LLM cost: <strong>$${((grounding.summary?.grounded_count||0 + grounding.summary?.inferred_count||0 + grounding.summary?.ungrounded_count||0) * 0.00001).toFixed(5)}</strong> ·
                    Embedding cost: <strong>$0.00</strong>
                </div>`;
        } catch (err) {
            ac.comparisonSection.innerHTML = `<div style="font-size:12px; color:#DC2626; padding:8px 0;">Embedding comparison failed: ${escHtml(err.message)}</div>`;
        }
    }

    // --- Launch Demo Button ---
    const DEMO_QUESTIONS = [
        "What are the key findings and performance metrics of this system?",
        "What is the technical methodology and chunking strategy used?",
        "How does the grounding verification and trust score work?",
    ];

    const launchBtn = document.querySelector('.btn-primary');
    if (launchBtn && launchBtn.textContent.includes('Launch Demo')) {
        launchBtn.addEventListener('click', async () => {
            launchBtn.disabled = true;
            launchBtn.innerHTML = '<i class="ph ph-spinner" style="animation:spin 1s linear infinite; font-size:14px;"></i> Loading…';

            // Upload sample doc if not already done
            if (uploadCount === 0) {
                const blob = new Blob([DEMO_DOC_TEXT], { type: 'text/plain' });
                await handleFileUpload(new File([blob], 'adaptive_rag_research.txt', { type: 'text/plain' }));
                if (sampleDocBtn) sampleDocBtn.style.display = 'none';
            }

            // Wait for upload to settle, then ask first question
            await new Promise(r => setTimeout(r, 600));
            if (chatInput) {
                chatInput.value = DEMO_QUESTIONS[0];
                sendMessage();
            }

            launchBtn.disabled = false;
            launchBtn.innerHTML = '<i class="ph-fill ph-lightning" style="color:#F59E0B"></i> Launch Demo';
        });
    }

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;
        chatInput.value = '';
        chatInput.disabled = true;
        if (sendBtn) sendBtn.disabled = true;

        const ac = createAnswerCard(text);

        // Hide landing page elements and suggested prompts on first message
        const defaultGrid = document.querySelector('#tab-chat .grid-2');
        const defaultHero = document.querySelector('#tab-chat .hero-card');
        const suggestedPrompts = document.getElementById('suggested-prompts');
        const demoGuide = document.getElementById('demo-guide-card');
        if (defaultGrid) defaultGrid.style.display = 'none';
        if (defaultHero) defaultHero.style.display = 'none';
        if (suggestedPrompts) suggestedPrompts.style.display = 'none';
        if (demoGuide) demoGuide.style.display = 'none';

        // Show New Chat button
        if (newChatBtn) newChatBtn.style.display = 'flex';

        if (contentMax) contentMax.appendChild(ac.element);

        // Auto-scroll to new message
        const contentDiv = document.querySelector('.content');
        if (contentDiv) setTimeout(() => { contentDiv.scrollTop = contentDiv.scrollHeight; }, 50);

        // Initial pipeline step
        addPipelineStep(ac.pipelineTrace,
            '<i class="ph-fill ph-spinner" style="color:#F59E0B; font-size:14px;"></i>',
            "Routing…", "Analyzing query intent");

        let answer = "";
        let sseBuffer = "";

        try {
            const response = await fetch(API_BASE + '/api/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: text,
                    session_id: SESSION_ID,
                    doc_available: uploadCount > 0,
                    doc_filename: uploadedFileName,
                    conversation_history: []
                })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail?.message || errData.message || `HTTP ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                sseBuffer += chunk;

                const lines = sseBuffer.split('\n');
                sseBuffer = lines.pop() || "";

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || trimmed.startsWith(':')) continue;

                    if (trimmed.startsWith('data: ')) {
                        const dataStr = trimmed.substring(6);
                        if (!dataStr) continue;

                        try {
                            const evt = JSON.parse(dataStr);

                            if (evt.type === 'stage') {
                                // --- Feature 1: Route Decision Card ---
                                if (evt.stage === 'routed' && evt.route) {
                                    const routeColors = { index: '#0F766E', search: '#1D4ED8', general: '#475569' };
                                    const routeBg    = { index: '#CCFBF1', search: '#DBEAFE', general: '#F1F5F9' };
                                    const routeIcons = { index: 'ph-file-text', search: 'ph-globe', general: 'ph-brain' };
                                    const r = evt.route;
                                    const card = document.createElement('div');
                                    card.style.cssText = `margin:10px 0; padding:10px 14px; border-radius:10px; background:${routeBg[r]||'#F1F5F9'}; border-left:3px solid ${routeColors[r]||'#64748B'};`;
                                    card.innerHTML = `
                                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                                          <i class="ph-fill ${routeIcons[r]||'ph-map-pin'}" style="color:${routeColors[r]||'#64748B'}; font-size:15px;"></i>
                                          <span style="font-weight:700; font-size:13px; color:${routeColors[r]||'#64748B'};">Routed → ${r.toUpperCase()}</span>
                                        </div>
                                        <div style="font-size:11px; color:#475569; line-height:1.4;">${escHtml(evt.reason||evt.message||'')}</div>`;
                                    ac.pipelineTrace.appendChild(card);
                                }

                                // --- Feature 2: Query Rewrite Visualizer ---
                                else if (evt.stage === 'rewriting') {
                                    const scores = evt.scores_before_rewrite || [];
                                    const best = scores.length ? Math.max(...scores) : null;
                                    const card = document.createElement('div');
                                    card.style.cssText = 'margin:10px 0; padding:12px 14px; border-radius:10px; background:#F5F3FF; border-left:3px solid #8B5CF6;';
                                    card.innerHTML = `
                                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                                          <i class="ph ph-arrows-clockwise" style="color:#8B5CF6; font-size:15px;"></i>
                                          <span style="font-weight:700; font-size:12px; color:#6D28D9;">Query Rewrite Loop ${evt.loop||1}</span>
                                          ${best !== null ? `<span style="margin-left:auto; font-size:11px; font-family:monospace; color:#DC2626; background:#FEE2E2; padding:2px 7px; border-radius:4px;">Best score: ${(best*100).toFixed(0)}% — below threshold</span>` : ''}
                                        </div>
                                        ${evt.original_question ? `
                                        <div style="font-size:11px; color:#64748B; margin-bottom:5px;">
                                          <span style="font-weight:600; color:#6D28D9;">Original:</span> <span style="font-style:italic;">"${escHtml(evt.original_question)}"</span>
                                        </div>
                                        <div style="font-size:11px; color:#64748B; display:flex; align-items:flex-start; gap:6px;">
                                          <i class="ph ph-arrow-bend-down-right" style="color:#8B5CF6; flex-shrink:0; margin-top:1px;"></i>
                                          <span><span style="font-weight:600; color:#059669;">Rewritten:</span> <span style="font-style:italic;">"${escHtml(evt.rewritten_question||'')}"</span></span>
                                        </div>` : `<div style="font-size:11px; color:#6D28D9;">${escHtml(evt.message||'')}</div>`}`;
                                    ac.pipelineTrace.appendChild(card);
                                }

                                else {
                                    const stageMap = {
                                        routing:      ['<i class="ph ph-map-pin" style="color:#F59E0B;font-size:14px;"></i>', 'Routing'],
                                        retrieving:   ['<i class="ph ph-magnifying-glass" style="color:#3B82F6;font-size:14px;"></i>', 'Retrieving'],
                                        retrieved:    ['<i class="ph-fill ph-check-circle" style="color:#10B981;font-size:14px;"></i>', 'Chunks graded'],
                                        searching_web:['<i class="ph ph-globe" style="color:#3B82F6;font-size:14px;"></i>', 'Web search'],
                                        generating:   ['<i class="ph ph-magic-wand" style="color:#F59E0B;font-size:14px;"></i>', 'Generating'],
                                        done:         ['<i class="ph-fill ph-check" style="color:#10B981;font-size:14px;"></i>', 'Done'],
                                    };
                                    const [icon, title] = stageMap[evt.stage] || ['•', evt.stage];
                                    addPipelineStep(ac.pipelineTrace, icon, title, evt.message || '');
                                }
                            }
                            else if (evt.type === 'token') {
                                answer += evt.content || '';
                                if (ac.liveAnswer) {
                                    ac.liveAnswer.textContent = answer;
                                    if (contentDiv) contentDiv.scrollTop = contentDiv.scrollHeight;
                                }
                            }
                            else if (evt.type === 'metadata') {
                                lastMetadata = {
                                    route_taken: evt.route_taken,
                                    relevance_scores: evt.relevance_scores,
                                    loops_executed: evt.loops_executed,
                                    processing_ms: evt.processing_ms,
                                    estimated_cost_usd: evt.estimated_cost_usd,
                                    token_usage: evt.token_usage,
                                    grounding: evt.grounding,
                                    answer_improvement: evt.answer_improvement,
                                    knowledge_gaps: evt.knowledge_gaps,
                                    source_chunks: evt.source_chunks,
                                    answer: answer,
                                };
                                fillAnswerCard(ac, { ...evt, answer }, answer);
                            }
                            else if (evt.type === 'done_all') {
                                updateSessionCost();
                                updatePipelineInspector(lastMetadata);
                                fetchGroundingComparison(ac, lastMetadata);
                                _updateSessionStats(lastMetadata);
                                // Show blinking NEW badge — answer is ready, draw attention
                                const dot = ac.card && ac.card.querySelector('.ai-new-dot');
                                if (dot) {
                                    dot.style.display = 'flex';
                                    setTimeout(() => { dot.style.display = 'none'; }, 8000);
                                }
                            }
                            else if (evt.type === 'error') {
                                const errDiv = document.createElement('div');
                                errDiv.style.cssText = 'color:#DC2626; margin-top:10px; font-size:13px; padding:10px; background:#FEF2F2; border-radius:8px;';
                                errDiv.innerHTML = `<i class="ph-fill ph-warning"></i> ${escHtml(evt.message || 'An unknown error occurred')}`;
                                ac.pipelineTrace.appendChild(errDiv);
                                showToast("Chat error: " + (evt.message || "Unknown"), "error");
                            }
                        } catch (e) {
                            console.warn("SSE parse skip:", e.message, "data:", dataStr.substring(0, 100));
                        }
                    }
                }
            }
        } catch (err) {
            console.error("Chat error:", err);
            const errDiv = document.createElement('div');
            errDiv.style.cssText = 'color:#DC2626; margin-top:10px; font-size:13px; padding:10px; background:#FEF2F2; border-radius:8px;';
            errDiv.innerHTML = `<i class="ph-fill ph-warning"></i> Error: ${escHtml(err.message)}`;
            ac.pipelineTrace.appendChild(errDiv);
            showToast("Chat request failed: " + err.message, "error");
        } finally {
            chatInput.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
            chatInput.focus();
        }
    }

    // --- Pipeline Inspector: "Try a live example" grounding demo ---
    const demoBtn = document.getElementById('demo-grounding-btn');
    if (demoBtn) {
        demoBtn.addEventListener('click', async () => {
            const demoContent = document.getElementById('demo-grounding-content');
            const demoAnswerEl = document.getElementById('demo-answer-text');
            const demoSentenceList = document.getElementById('demo-sentence-list');
            const demoComparison = document.getElementById('demo-comparison');
            const demoComparisonBody = document.getElementById('demo-comparison-body');
            const demoStats = document.getElementById('demo-comparison-stats');
            const demoLoading = document.getElementById('demo-loading');
            if (!demoContent) return;

            demoContent.style.display = 'block';
            demoBtn.disabled = true;
            demoBtn.textContent = 'Loading…';

            const DEMO_ANSWER = "The system uses LangGraph for agentic workflow orchestration. This approach likely reduces hallucination compared to naive RAG. The system processes over 10,000 queries per day.";
            const DEMO_CHUNK = "LangGraph is used to define the agentic graph with nodes for routing, retrieval, grading, and generation. The adaptive routing mechanism classifies each query into index, search, or general routes before retrieval.";
            const LLM_RESULTS = [
                { sentence: "The system uses LangGraph for agentic workflow orchestration.", label: "GROUNDED" },
                { sentence: "This approach likely reduces hallucination compared to naive RAG.", label: "INFERRED" },
                { sentence: "The system processes over 10,000 queries per day.", label: "UNGROUNDED" },
            ];

            if (demoAnswerEl) demoAnswerEl.textContent = DEMO_ANSWER;

            // Render sentence breakdown
            demoSentenceList.innerHTML = '';
            LLM_RESULTS.forEach(r => {
                const dotClass = { GROUNDED: 'dot-grounded', INFERRED: 'dot-inferred', UNGROUNDED: 'dot-ungrounded' }[r.label];
                const lblClass = { GROUNDED: 'label-grounded', INFERRED: 'label-inferred', UNGROUNDED: 'label-ungrounded' }[r.label];
                const row = document.createElement('div');
                row.className = 'sentence-row';
                row.innerHTML = `<span class="sentence-dot ${dotClass}"></span><span class="sentence-text">${escHtml(r.sentence)}</span><span class="sentence-label ${lblClass}">${r.label}</span>`;
                demoSentenceList.appendChild(row);
            });

            // Fetch embedding comparison
            demoLoading.style.display = 'block';
            try {
                const res = await fetch(API_BASE + '/api/grounding/compare', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ answer: DEMO_ANSWER, chunks: [DEMO_CHUNK], llm_results: LLM_RESULTS })
                });
                const data = await res.json();
                demoLoading.style.display = 'none';

                demoComparisonBody.innerHTML = data.comparison.map(c => {
                    const llmC = { GROUNDED: 'label-grounded', INFERRED: 'label-inferred', UNGROUNDED: 'label-ungrounded' }[c.llm] || '';
                    const embC = { GROUNDED: 'label-grounded', INFERRED: 'label-inferred', UNGROUNDED: 'label-ungrounded' }[c.embedding] || '';
                    return `<tr>
                        <td style="color:var(--text-muted); font-size:11px;">${escHtml(c.sentence.slice(0,55))}…</td>
                        <td><span class="sentence-label ${llmC}">${c.llm}</span></td>
                        <td><span class="sentence-label ${embC}">${c.embedding}</span></td>
                        <td style="font-family:'IBM Plex Mono',monospace; font-size:11px;">${c.similarity.toFixed(2)}</td>
                        <td class="${c.match ? 'match-yes' : 'match-no'}">${c.match ? '✓' : '✗'}</td>
                    </tr>`;
                }).join('');

                const agePct = Math.round((data.agreement_rate || 0) * 100);
                demoStats.textContent = `Agreement: ${agePct}% (${data.agreement_count}/${data.total_sentences}) · Embedding latency: ${data.embedding_latency_ms}ms · Embedding cost: $0.00`;
                demoComparison.style.display = 'block';
            } catch (e) {
                demoLoading.style.display = 'none';
                demoLoading.textContent = `Failed: ${e.message}`;
                demoLoading.style.display = 'block';
            }
            demoBtn.textContent = '✓ Loaded';
        });
    }

    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    }

});
