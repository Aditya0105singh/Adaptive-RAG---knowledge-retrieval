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
        const res = await fetch(`/api/sessions/${SESSION_ID}/cost`);
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
            const res = await fetch(`/api/sessions/${SESSION_ID}/last-metadata`);
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
        const res = await fetch('/api/system-status');
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
        const res = await fetch('/api/benchmarks');
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
            const res = await fetch('/api/upload', {
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
                showToast(`${file.name} uploaded! (${data.parent_count} parent, ${data.child_count} child chunks)`, "success");

                // Add file chip to sidebar list
                const filesList = document.getElementById('uploaded-files-list');
                const footerText = document.getElementById('dropzone-footer-text');
                if (filesList) {
                    filesList.style.display = 'block';
                    const chip = document.createElement('div');
                    chip.className = 'file-chip';
                    chip.innerHTML = `<i class="ph ph-file-text" style="font-size:14px; flex-shrink:0;"></i><span class="file-chip-name">${file.name}</span><span style="flex-shrink:0; color:var(--text-muted); font-size:10px;">${(file.size/1024).toFixed(0)}KB</span>`;
                    filesList.appendChild(chip);
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

    // --- New Chat button ---
    const newChatBtn = document.getElementById('new-chat-btn');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', () => {
            // Remove all response cards (keep hero + grid)
            const responseCards = document.querySelectorAll('#tab-chat .content-max > .card');
            responseCards.forEach(c => c.remove());

            // Restore landing
            const defaultGrid = document.querySelector('#tab-chat .grid-2');
            const defaultHero = document.querySelector('#tab-chat .hero-card');
            const suggestedPrompts = document.getElementById('suggested-prompts');
            if (defaultGrid) defaultGrid.style.display = '';
            if (defaultHero) defaultHero.style.display = '';
            if (suggestedPrompts) suggestedPrompts.style.display = 'flex';

            // Hide the button again
            newChatBtn.style.display = 'none';

            if (chatInput) { chatInput.value = ''; chatInput.focus(); }
        });
    }

    // --- Chat Logic & SSE Streaming ---
    const chatInput = document.querySelector('.chat-input');
    const sendBtn = document.querySelector('.btn-send');
    const contentMax = document.querySelector('#tab-chat .content-max');

    function createStepContainer(query) {
        const container = document.createElement('div');
        container.className = 'card';
        container.style.marginBottom = '24px';
        container.innerHTML = `
            <div class="section-title">RESPONSE</div>
            <div class="step-container">
                <div class="step-query">
                    <i class="ph ph-chat-teardrop-text" style="color:var(--text-muted); font-size:18px;"></i>
                    "${query}"
                </div>
                <div class="steps-list"></div>
            </div>
        `;
        return {
            element: container,
            stepsList: container.querySelector('.steps-list')
        };
    }

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

    // --- Launch Demo Button ---
    const launchBtn = document.querySelector('.btn-primary');
    if (launchBtn && launchBtn.textContent.includes('Launch Demo')) {
        launchBtn.addEventListener('click', () => {
            if (chatInput) {
                chatInput.value = "What is Adaptive RAG and how does it work?";
                chatInput.focus();
                sendMessage();
            }
        });
    }

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;
        chatInput.value = '';
        chatInput.disabled = true;
        if (sendBtn) sendBtn.disabled = true;

        const chatBlock = createStepContainer(text);

        // Hide landing page elements and suggested prompts on first message
        const defaultGrid = document.querySelector('#tab-chat .grid-2');
        const defaultHero = document.querySelector('#tab-chat .hero-card');
        const suggestedPrompts = document.getElementById('suggested-prompts');
        if (defaultGrid) defaultGrid.style.display = 'none';
        if (defaultHero) defaultHero.style.display = 'none';
        if (suggestedPrompts) suggestedPrompts.style.display = 'none';

        // Show New Chat button
        if (newChatBtn) newChatBtn.style.display = 'flex';

        if (contentMax) contentMax.appendChild(chatBlock.element);

        // Auto-scroll to new message
        const contentDiv = document.querySelector('.content');
        if (contentDiv) setTimeout(() => { contentDiv.scrollTop = contentDiv.scrollHeight; }, 50);

        chatBlock.stepsList.innerHTML = createStepHTML(
            "1",
            '<i class="ph-fill ph-spinner" style="color:#F59E0B; font-size:16px;"></i>',
            "Routing...",
            "Analyzing query intent"
        );

        let answer = "";
        let stepNum = 1;
        let sseBuffer = "";

        try {
            const response = await fetch('/api/chat/stream', {
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
                                if (evt.stage === 'routing') {
                                    chatBlock.stepsList.innerHTML = createStepHTML(
                                        stepNum,
                                        '<i class="ph-fill ph-map-pin" style="color:#F59E0B; font-size:16px;"></i>',
                                        "Routing",
                                        evt.message || "Classifying your question..."
                                    );
                                }
                                else if (evt.stage === 'routed') {
                                    chatBlock.stepsList.innerHTML = createStepHTML(
                                        stepNum++,
                                        '<i class="ph-fill ph-map-pin" style="color:#F59E0B; font-size:16px;"></i>',
                                        "Route Classification",
                                        `LLM decides: <strong>${(evt.route || 'GENERAL').toUpperCase()}</strong>`,
                                        'done'
                                    );
                                }
                                else if (evt.stage === 'retrieving') {
                                    chatBlock.stepsList.innerHTML += createStepHTML(
                                        stepNum,
                                        '<i class="ph ph-magnifying-glass" style="color:#3B82F6; font-size:16px; font-weight:bold;"></i>',
                                        "Semantic Retrieval",
                                        evt.message || "Searching your document..."
                                    );
                                }
                                else if (evt.stage === 'retrieved') {
                                    chatBlock.stepsList.innerHTML += createStepHTML(
                                        stepNum++,
                                        '<i class="ph-fill ph-check-circle" style="color:#10B981; font-size:16px;"></i>',
                                        "Chunks Graded",
                                        evt.message || "Relevance scoring complete",
                                        'done'
                                    );
                                }
                                else if (evt.stage === 'rewriting') {
                                    chatBlock.stepsList.innerHTML += createStepHTML(
                                        stepNum++,
                                        '<i class="ph ph-pencil-simple" style="color:#8B5CF6; font-size:16px;"></i>',
                                        "Query Rewrite",
                                        evt.message || "Rewriting query for better retrieval...",
                                        'done'
                                    );
                                }
                                else if (evt.stage === 'searching_web') {
                                    chatBlock.stepsList.innerHTML += createStepHTML(
                                        stepNum,
                                        '<i class="ph ph-globe" style="color:#3B82F6; font-size:16px;"></i>',
                                        "Web Search",
                                        evt.message || "Searching the web..."
                                    );
                                }
                                else if (evt.stage === 'generating') {
                                    chatBlock.stepsList.innerHTML += createStepHTML(
                                        stepNum++,
                                        '<i class="ph ph-magic-wand" style="color:#F59E0B; font-size:16px;"></i>',
                                        "Generating Answer",
                                        `<span class="live-answer"></span>`
                                    );
                                }
                            }
                            else if (evt.type === 'token') {
                                answer += evt.content || '';
                                const answerSpan = chatBlock.stepsList.querySelector('.live-answer');
                                if (answerSpan) {
                                    answerSpan.innerText = answer;
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
                                };

                                const answerSpan = chatBlock.stepsList.querySelector('.live-answer');
                                if (answerSpan && answer) {
                                    if (typeof marked !== 'undefined') {
                                        answerSpan.innerHTML = marked.parse(answer);
                                    } else {
                                        answerSpan.innerText = answer;
                                    }
                                }

                                if (evt.grounding && evt.grounding.overall_score !== undefined) {
                                    const score = (evt.grounding.overall_score * 100).toFixed(0);
                                    chatBlock.stepsList.innerHTML += createStepHTML(
                                        stepNum++,
                                        '<i class="ph-fill ph-shield-check" style="color:#3B82F6; font-size:16px;"></i>',
                                        "Grounding Verification",
                                        `Overall grounding score: <strong>${score}%</strong>`,
                                        'done'
                                    );
                                    updateGroundingLab(evt.grounding);
                                }

                                if (evt.knowledge_gaps && evt.knowledge_gaps.length > 0) {
                                    const gapItems = evt.knowledge_gaps.map(g =>
                                        `<div style="margin-bottom:4px; display:flex; gap:6px;"><i class="ph ph-warning" style="color:#F59E0B; flex-shrink:0; margin-top:2px;"></i><span>${g}</span></div>`
                                    ).join('');
                                    chatBlock.stepsList.innerHTML += `<div style="margin-top:10px; padding:10px 14px; background:#FFFBEB; border:1px solid #FDE68A; border-radius:8px; font-size:12px; color:#92400E;">
                                        <div style="font-weight:700; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.05em; font-size:10px;">Knowledge Gaps Detected</div>
                                        ${gapItems}
                                    </div>`;
                                }

                                if (evt.answer_improvement) {
                                    chatBlock.stepsList.innerHTML += `<div style="margin-top:8px; padding:8px 12px; background:#F0FDF4; border:1px solid #6EE7B7; border-radius:8px; font-size:12px; color:#065F46; display:flex; gap:8px; align-items:flex-start;">
                                        <i class="ph-fill ph-arrows-clockwise" style="color:#059669; flex-shrink:0; margin-top:1px;"></i>
                                        <span><strong>Answer refined:</strong> ${evt.answer_improvement}</span>
                                    </div>`;
                                }

                                chatBlock.stepsList.innerHTML += createStepHTML(
                                    stepNum,
                                    '<i class="ph-fill ph-check-circle" style="color:#10B981; font-size:16px;"></i>',
                                    "Complete",
                                    `Processed in ${evt.processing_ms}ms · Cost: $${(evt.estimated_cost_usd || 0).toFixed(4)} · Route: ${evt.route_taken}`,
                                    'done'
                                );
                            }
                            else if (evt.type === 'done_all') {
                                updateSessionCost();
                                updatePipelineInspector(lastMetadata);
                            }
                            else if (evt.type === 'error') {
                                chatBlock.stepsList.innerHTML += createStepHTML(
                                    "!",
                                    '<i class="ph-fill ph-warning" style="color:#DC2626; font-size:16px;"></i>',
                                    "Error",
                                    evt.message || "An unknown error occurred",
                                    'error'
                                );
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
            chatBlock.stepsList.innerHTML += `<div style="color:#DC2626; margin-top:10px; font-size:13px;"><i class="ph-fill ph-warning"></i> Error: ${err.message}</div>`;
            showToast("Chat request failed: " + err.message, "error");
        } finally {
            chatInput.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
            chatInput.focus();
        }
    }

    function updateGroundingLab(groundingData) {
        const groundingContainer = document.querySelector('#tab-grounding .card');
        if (!groundingContainer || !groundingData) return;

        let html = `<div class="section-title">GROUNDING LAB</div>`;
        html += `<div style="font-size:13px; color:var(--text-muted); margin-bottom:16px;">Sentence-level grounding analysis from the last query.</div>`;

        const score = (groundingData.overall_score * 100).toFixed(0);
        const scoreColor = score >= 80 ? '#059669' : score >= 50 ? '#F59E0B' : '#DC2626';
        html += `<div style="margin-bottom:16px; padding:12px 16px; background:#F0FDF4; border:1px solid #6EE7B7; border-radius:8px; display:flex; align-items:center; gap:16px;">
            <div style="font-size:13px; font-weight:500;">Overall Grounding Score</div>
            <span style="color:${scoreColor}; font-weight:800; font-size:22px; font-family:monospace; margin-left:auto;">${score}%</span>
        </div>`;

        if (groundingData.sentence_scores && groundingData.sentence_scores.length > 0) {
            html += `<div style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-muted); margin-bottom:10px;">SENTENCE BREAKDOWN</div>`;
            groundingData.sentence_scores.forEach((item, idx) => {
                const s = ((item.score || 0) * 100).toFixed(0);
                const c = s >= 80 ? '#059669' : s >= 50 ? '#F59E0B' : '#EF4444';
                html += `<div style="padding:10px 14px; border:1px solid var(--border); border-radius:8px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size:11px; color:var(--text-muted); font-weight:600;">Sentence ${idx + 1}</span>
                        <span style="color:${c}; font-weight:700; font-family:monospace;">${s}%</span>
                    </div>
                    <div style="font-size:13px; color:var(--text-main); line-height:1.5;">${item.sentence || ''}</div>
                </div>`;
            });
        }

        groundingContainer.innerHTML = html;
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
