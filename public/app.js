/**
 * SupportMind — AI Intelligence Platform
 * Frontend Application
 *
 * Handles:
 * - Single ticket classification with confidence visualization (REAL — /predict endpoint)
 * - Batch CSV upload with sortable/filterable results table (REAL — /predict/batch)
 * - Dashboard with charts and model metrics (REAL data from /model-info)
 * - Model Training page (DEMO controls — no in-browser retraining)
 * - Analytics page (DEMO — simulated volume/sentiment charts)
 * - Team page (DEMO — simulated agent data)
 * - Admin Panel (DEMO — simulated activity & health data)
 * - Settings page (DEMO — UI-only, no persistence)
 * - API Docs (REAL — documents actual /predict endpoint)
 *
 * API base URL is configurable via window.API_BASE_URL or defaults to localhost:8000.
 */

// ===========================================================================
// Configuration
// ===========================================================================
const RENDER_BACKEND = 'https://support-ticket-classifier-api-k2eq.onrender.com';
const isLocal = ['localhost', '127.0.0.1', ''].includes(window.location.hostname) || window.location.protocol === 'file:';
const API_BASE = window.API_BASE_URL || (isLocal ? 'http://localhost:8000' : RENDER_BACKEND);

// Chart.js default styling for dark theme
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";

// Color palettes
const CATEGORY_COLORS = [
    '#6366f1', '#8b5cf6', '#a78bfa', '#c084fc',
    '#e879f9', '#f472b6', '#fb7185', '#f97316',
];
const PRIORITY_COLORS = {
    'High': '#ef4444',
    'Medium': '#f59e0b',
    'Low': '#22c55e',
};

// Sample tickets for the Ticket Analyzer
const SAMPLE_TICKETS = [
    "My laptop screen is flickering intermittently and sometimes goes completely black. I've tried restarting multiple times but the issue persists. The laptop is a Dell Latitude 5520, about 2 years old. I need this fixed urgently as I have a client presentation tomorrow and cannot work without my primary machine.",
    "I'm unable to log into the company VPN since this morning. I keep getting an 'authentication failed' error even though I'm sure my password is correct. I've tried resetting my credentials through the self-service portal but it says my account is locked. I need remote access urgently as I'm working from home this week.",
    "We have a new employee starting next Monday in the Marketing department. Could you please set up the standard onboarding package? They'll need a company email, access to the marketing shared drive, Slack workspace invitation, and a laptop with the standard software suite installed.",
    "I need to purchase 5 additional Adobe Creative Cloud licenses for our design team. We've recently hired new designers and they need immediate access to Photoshop, Illustrator, and InDesign. Please process this as a priority purchase order and let me know the timeline for activation."
];

// ===========================================================================
// State
// ===========================================================================
let batchData = [];
let batchSortKey = 'index';
let batchSortAsc = true;
let demoMode = false;

// Real model data — populated from /model-info or hardcoded from actual training results
const REAL_CV_DATA = {
    category: {
        LogisticRegression: { mean_f1: 0.8470, std_f1: 0.0041, mean_accuracy: 0.8475, std_accuracy: 0.0038 },
        LinearSVC: { mean_f1: 0.8512, std_f1: 0.0050, mean_accuracy: 0.8514, std_accuracy: 0.0034 },
        MultinomialNB: { mean_f1: 0.7649, std_f1: 0.0079, mean_accuracy: 0.7845, std_accuracy: 0.0047 },
        RandomForest: { mean_f1: 0.8415, std_f1: 0.0042, mean_accuracy: 0.8366, std_accuracy: 0.0018 },
    },
    priority: {
        LogisticRegression: { mean_f1: 0.8156, std_f1: 0.0031, mean_accuracy: 0.8217, std_accuracy: 0.0030 },
        LinearSVC: { mean_f1: 0.8139, std_f1: 0.0025, mean_accuracy: 0.8224, std_accuracy: 0.0027 },
        MultinomialNB: { mean_f1: 0.7601, std_f1: 0.0045, mean_accuracy: 0.7706, std_accuracy: 0.0044 },
        RandomForest: { mean_f1: 0.8145, std_f1: 0.0028, mean_accuracy: 0.8221, std_accuracy: 0.0027 },
    }
};

// Best models (from actual training)
const BEST_MODELS = {
    category: { name: 'LinearSVC', algorithm: 'LinearSVC (CalibratedClassifierCV)' },
    priority: { name: 'LogisticRegression', algorithm: 'Logistic Regression' }
};

// Real category distribution from training data (per_class support counts)
const REAL_CATEGORY_DISTRIBUTION = {
    'Hardware': 2724, 'HR Support': 2183, 'Access': 1425,
    'Miscellaneous': 1412, 'Storage': 555, 'Purchase': 493,
    'Internal Project': 424, 'Administrative rights': 352
};

// ===========================================================================
// Demo Mode — client-side keyword classifier (fallback when no backend)
// ===========================================================================
const DEMO_RULES = [
    { category: 'Hardware', keywords: ['laptop', 'screen', 'monitor', 'keyboard', 'mouse', 'printer', 'cable', 'pc', 'computer', 'broken', 'replacement', 'hard drive', 'battery', 'charger', 'display', 'dock', 'headset', 'webcam', 'device'] },
    { category: 'Access', keywords: ['password', 'login', 'access', 'locked', 'reset', 'vpn', 'account', 'credential', 'authenticate', 'sso', 'mfa', 'token', 'expire', 'permission denied', 'cannot access', 'sign in'] },
    { category: 'Administrative rights', keywords: ['admin', 'administrator', 'privilege', 'elevated', 'rights', 'sudo', 'root', 'permission', 'install software', 'group policy'] },
    { category: 'HR Support', keywords: ['onboarding', 'offboarding', 'new employee', 'leave', 'payroll', 'benefits', 'hr', 'human resources', 'hire', 'termination', 'contract', 'salary'] },
    { category: 'Storage', keywords: ['storage', 'disk space', 'file share', 'drive', 'backup', 'quota', 'shared folder', 'onedrive', 'sharepoint', 'cloud storage', 'nas'] },
    { category: 'Purchase', keywords: ['purchase', 'order', 'buy', 'procurement', 'license', 'subscription', 'invoice', 'budget', 'vendor', 'quote', 'cost'] },
    { category: 'Internal Project', keywords: ['project', 'timeline', 'milestone', 'resource', 'planning', 'sprint', 'deliverable', 'stakeholder', 'scope'] },
    { category: 'Miscellaneous', keywords: ['help', 'question', 'information', 'general', 'other', 'inquiry', 'support', 'issue'] },
];
const URGENCY_WORDS = ['urgent', 'asap', 'critical', 'emergency', 'immediately', 'blocked', 'down', 'crashed', 'broken', 'not working', 'deadline', 'production'];

function demoPrediction(text) {
    const lower = text.toLowerCase();
    let scores = DEMO_RULES.map(rule => ({
        category: rule.category,
        score: rule.keywords.filter(kw => lower.includes(kw)).length
    }));
    scores.sort((a, b) => b.score - a.score);
    const bestCat = scores[0].score > 0 ? scores[0].category : 'Miscellaneous';
    const totalScore = Math.max(scores.reduce((s, x) => s + x.score, 0), 1);
    const catProbs = {};
    scores.forEach(s => { catProbs[s.category] = Math.max(0.01, s.score / totalScore); });
    const probSum = Object.values(catProbs).reduce((a, b) => a + b, 0);
    Object.keys(catProbs).forEach(k => { catProbs[k] = Math.round((catProbs[k] / probSum) * 10000) / 10000; });
    const catConf = catProbs[bestCat];

    const urgencyHits = URGENCY_WORDS.filter(w => lower.includes(w)).length;
    const highCats = ['Access', 'Administrative rights'];
    const medCats = ['Hardware', 'Storage'];
    let priScore = highCats.includes(bestCat) ? 2 : medCats.includes(bestCat) ? 1 : 0;
    priScore = Math.min(2, priScore + Math.min(urgencyHits, 2));
    const priority = ['Low', 'Medium', 'High'][priScore];
    const priProbs = { 'Low': 0.1, 'Medium': 0.1, 'High': 0.1 };
    priProbs[priority] = 0.8;

    return {
        text: text.substring(0, 500) + (text.length > 500 ? '...' : ''),
        category: bestCat,
        category_confidence: Math.max(0.55, Math.min(0.98, catConf)),
        priority: priority,
        priority_confidence: 0.8,
        category_probabilities: catProbs,
        priority_probabilities: priProbs,
    };
}

// ===========================================================================
// Initialization
// ===========================================================================
document.addEventListener('DOMContentLoaded', () => {
    initRouter();
    initTextarea();
    initFileUpload();
    checkHealth();
    loadModelInfo();
    populateModelComparison();
    populateAgentList();
    populateActivityFeed();
    populateAuditLog();
    measureApiLatency();
    initTrainingCurves();
});

// ===========================================================================
// Sidebar toggle (mobile)
// ===========================================================================
function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
    document.getElementById('sidebarOverlay').classList.toggle('open');
}

// ===========================================================================
// Router
// ===========================================================================
const PAGE_TITLES = {
    'dashboard': { title: 'Dashboard', subtitle: 'Good evening, Admin' },
    'ticket-analyzer': { title: 'Ticket Analyzer', subtitle: 'AI-powered ticket classification and prioritization' },
    'model-training': { title: 'Model Training', subtitle: 'Configure and retrain the machine learning models' },
    'dataset-upload': { title: 'Dataset Upload', subtitle: 'Upload and preprocess raw ticket data' },
    'analytics': { title: 'Analytics', subtitle: 'Historical ticket volume and sentiment trends' },
    'team': { title: 'Team', subtitle: 'Agent workload and AI routing rules' },
    'admin': { title: 'Admin Panel', subtitle: 'System health and audit logs' },
    'settings': { title: 'Settings', subtitle: 'Organization preferences and SLAs' },
    'api-docs': { title: 'API Docs', subtitle: 'Integration endpoints and authentication' }
};

function initRouter() {
    function navigate() {
        let hash = window.location.hash.replace('#', '') || 'dashboard';
        if (!PAGE_TITLES[hash]) hash = 'dashboard';

        // Close mobile sidebar on navigation
        document.getElementById('sidebar').classList.remove('open');
        document.getElementById('sidebarOverlay').classList.remove('open');

        // Update active nav link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.toggle('active', link.dataset.target === hash);
        });

        // Update active page view
        document.querySelectorAll('.page-view').forEach(view => {
            view.classList.toggle('active', view.id === 'view-' + hash);
        });

        // Update topbar titles
        const titles = PAGE_TITLES[hash];
        document.getElementById('pageTitle').textContent = titles.title;

        if (hash === 'dashboard') {
            const hour = new Date().getHours();
            let greeting = 'Good evening';
            if (hour < 12) greeting = 'Good morning';
            else if (hour < 18) greeting = 'Good afternoon';
            document.getElementById('pageSubtitle').textContent = greeting + ', Admin';
        } else {
            document.getElementById('pageSubtitle').textContent = titles.subtitle;
        }

        // Initialize charts when navigating to relevant pages
        if (hash === 'dashboard') initDashboardCharts();
        if (hash === 'analytics') initAnalyticsCharts();
    }

    window.addEventListener('hashchange', navigate);
    navigate();
}

// ===========================================================================
// Textarea
// ===========================================================================
function initTextarea() {
    const textarea = document.getElementById('ticketText');
    const charCount = document.getElementById('charCount');
    if (!textarea || !charCount) return;

    textarea.addEventListener('input', () => {
        const text = textarea.value;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        charCount.textContent = `${text.length} chars · ${words} words`;
    });

    textarea.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            classifySingle();
        }
    });
}

function loadSample(index) {
    const textarea = document.getElementById('ticketText');
    textarea.value = SAMPLE_TICKETS[index] || '';
    textarea.dispatchEvent(new Event('input'));
    textarea.focus();
}

// ===========================================================================
// Health Check
// ===========================================================================
async function checkHealth() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
        clearTimeout(timeoutId);

        const data = await res.json();
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');
        if (data.model_loaded) {
            demoMode = false;
            dot.classList.remove('error');
            text.textContent = 'AI Active';
        } else {
            demoMode = true;
            dot.classList.add('error');
            text.textContent = 'Demo Mode';
        }
    } catch (e) {
        demoMode = true;
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');
        dot.classList.add('error');
        text.textContent = 'Offline (Demo)';
    }
}

// ===========================================================================
// Error Handling
// ===========================================================================
function showError(message) {
    const banner = document.getElementById('errorBanner');
    document.getElementById('errorText').textContent = message;
    banner.classList.add('visible');
}

function hideError() {
    document.getElementById('errorBanner').classList.remove('visible');
}

// ===========================================================================
// Single Ticket Classification (REAL — calls /predict)
// ===========================================================================
async function classifySingle() {
    const text = document.getElementById('ticketText').value.trim();
    if (!text) { showError('Please enter ticket text before classifying.'); return; }
    if (text.length < 3) { showError('Ticket text must be at least 3 characters long.'); return; }

    hideError();
    const btn = document.getElementById('classifyBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span> Analyzing...';

    try {
        if (demoMode) {
            await new Promise(r => setTimeout(r, 400));
            displaySingleResult(demoPrediction(text));
        } else {
            const res = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            displaySingleResult(await res.json());
        }
    } catch (e) {
        showError(`Classification failed: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Analyze with AI`;
    }
}

function displaySingleResult(data) {
    const container = document.getElementById('singleResults');
    const emptyState = document.getElementById('emptyResultsState');

    if (emptyState) emptyState.style.display = 'none';
    if (container) container.style.display = 'block';

    document.getElementById('resultCategory').textContent = data.category;
    document.getElementById('catConfText').textContent = `${(data.category_confidence * 100).toFixed(1)}%`;
    document.getElementById('catConfBar').style.width = `${(data.category_confidence * 100)}%`;

    const priEl = document.getElementById('resultPriority');
    priEl.innerHTML = `<span class="priority-badge ${data.priority.toLowerCase()}">${data.priority}</span>`;
    document.getElementById('priConfText').textContent = `${(data.priority_confidence * 100).toFixed(1)}%`;
    document.getElementById('priConfBar').style.width = `${(data.priority_confidence * 100)}%`;

    if (data.category_probabilities) renderProbBars('catProbBars', data.category_probabilities);
    if (data.priority_probabilities) renderProbBars('priProbBars', data.priority_probabilities);

    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderProbBars(containerId, probs) {
    const container = document.getElementById(containerId);
    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    const maxProb = sorted[0][1];

    container.innerHTML = sorted.map(([label, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const barWidth = maxProb > 0 ? (prob / maxProb * 100) : 0;
        return `<div class="prob-item">
            <span class="prob-label">${label}</span>
            <div class="prob-bar"><div class="prob-bar-fill" style="width: ${barWidth}%"></div></div>
            <span class="prob-value">${pct}%</span>
        </div>`;
    }).join('');
}

function clearSingle() {
    document.getElementById('ticketText').value = '';
    document.getElementById('charCount').textContent = '0 chars · 0 words';
    const container = document.getElementById('singleResults');
    const emptyState = document.getElementById('emptyResultsState');
    if (container) container.style.display = 'none';
    if (emptyState) emptyState.style.display = 'flex';
    hideError();
}

// ===========================================================================
// Batch Upload (REAL — calls /predict/batch)
// ===========================================================================
function initFileUpload() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('csvFileInput');
    if (!zone || !input) return;

    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => { zone.classList.remove('dragover'); });
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleCSVFile(e.dataTransfer.files[0]);
    });

    input.addEventListener('change', (e) => {
        if (e.target.files.length) handleCSVFile(e.target.files[0]);
    });
}

async function handleCSVFile(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showError('Please upload a CSV file.');
        return;
    }

    hideError();
    document.getElementById('uploadedFileName').textContent = `📄 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    document.getElementById('uploadedFileInfo').style.display = 'flex';

    const text = await file.text();
    Papa.parse(text, {
        header: true,
        skipEmptyLines: true,
        complete: async (results) => {
            const columns = results.meta.fields;
            const textCol = ['text', 'Document', 'ticket_text', 'description', 'Ticket Description']
                .find(col => columns.includes(col));

            if (!textCol) {
                showError(`CSV must have a text column. Found: ${columns.join(', ')}`);
                return;
            }

            const tickets = results.data
                .map(row => row[textCol])
                .filter(t => t && t.trim().length > 0)
                .slice(0, 1000);

            if (tickets.length === 0) {
                showError('No valid ticket texts found in the CSV.');
                return;
            }

            await classifyBatch(tickets);
        },
        error: (err) => showError(`Failed to parse CSV: ${err.message}`)
    });
}

async function classifyBatch(texts) {
    const uploadZone = document.getElementById('uploadZone');
    uploadZone.innerHTML = `<span class="loading-spinner" style="width:32px;height:32px;border-width:3px;"></span><p style="margin-top:16px;">Classifying ${texts.length} tickets...</p>`;

    try {
        let predictions;
        if (demoMode) {
            await new Promise(r => setTimeout(r, 300));
            predictions = texts.map(t => demoPrediction(t));
        } else {
            const body = { tickets: texts.map(t => ({ text: t })) };
            const res = await fetch(`${API_BASE}/predict/batch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            predictions = data.predictions;
        }
        batchData = predictions.map((p, i) => ({ index: i + 1, ...p }));

        const categories = [...new Set(batchData.map(d => d.category))].sort();
        const filterCat = document.getElementById('filterCategory');
        filterCat.innerHTML = '<option value="">All Categories</option>' +
            categories.map(c => `<option value="${c}">${c}</option>`).join('');

        renderBatchTable();
        document.getElementById('batchResults').style.display = 'block';
    } catch (e) {
        showError(`Batch classification failed: ${e.message}`);
    } finally {
        resetUploadZone();
    }
}

function resetUploadZone() {
    const uploadZone = document.getElementById('uploadZone');
    uploadZone.innerHTML = `
        <input type="file" accept=".csv,.xlsx,.json" id="csvFileInput" aria-label="Upload dataset file">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        <p>Drop your dataset file here or <strong>click to browse</strong></p>
        <span class="hint">CSV, Excel, or JSON · Max 15 MB · Max 1,000 rows for batch classification</span>
    `;
    initFileUpload();
}

function renderBatchTable() {
    const catFilter = document.getElementById('filterCategory').value;
    const priFilter = document.getElementById('filterPriority').value;

    let filtered = batchData;
    if (catFilter) filtered = filtered.filter(d => d.category === catFilter);
    if (priFilter) filtered = filtered.filter(d => d.priority === priFilter);

    filtered.sort((a, b) => {
        let va = a[batchSortKey], vb = b[batchSortKey];
        if (typeof va === 'string') { va = va.toLowerCase(); vb = vb.toLowerCase(); }
        if (va < vb) return batchSortAsc ? -1 : 1;
        if (va > vb) return batchSortAsc ? 1 : -1;
        return 0;
    });

    const tbody = document.getElementById('batchTableBody');
    tbody.innerHTML = filtered.map(row => {
        const truncatedText = row.text.length > 100 ? row.text.substring(0, 100) + '...' : row.text;
        return `<tr>
            <td>${row.index}</td>
            <td title="${escapeHtml(row.text)}">${escapeHtml(truncatedText)}</td>
            <td>${row.category}</td>
            <td>${(row.category_confidence * 100).toFixed(1)}%</td>
            <td><span class="priority-badge ${row.priority.toLowerCase()}">${row.priority}</span></td>
            <td>${(row.priority_confidence * 100).toFixed(1)}%</td>
        </tr>`;
    }).join('');

    document.getElementById('batchCount').textContent = `${filtered.length} of ${batchData.length} tickets`;
}

function sortBatchTable(key) {
    if (batchSortKey === key) { batchSortAsc = !batchSortAsc; }
    else { batchSortKey = key; batchSortAsc = true; }
    renderBatchTable();
}

function filterBatchTable() { renderBatchTable(); }

function exportBatchCSV() {
    if (batchData.length === 0) return;
    const headers = ['#', 'Text', 'Category', 'Category Confidence', 'Priority', 'Priority Confidence'];
    const rows = batchData.map(d => [
        d.index,
        `"${d.text.replace(/"/g, '""')}"`,
        d.category,
        (d.category_confidence * 100).toFixed(1) + '%',
        d.priority,
        (d.priority_confidence * 100).toFixed(1) + '%',
    ]);
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ticket_predictions_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

function clearBatch() {
    batchData = [];
    document.getElementById('batchResults').style.display = 'none';
    document.getElementById('uploadedFileInfo').style.display = 'none';
    const fileInput = document.getElementById('csvFileInput');
    if (fileInput) fileInput.value = '';
    hideError();
}

// ===========================================================================
// Dashboard (REAL data from /model-info + demo chart data)
// ===========================================================================
async function loadModelInfo() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        const res = await fetch(`${API_BASE}/model-info`, { signal: controller.signal });
        clearTimeout(timeoutId);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        renderModelInfo(await res.json());
    } catch (e) {
        // Fallback to hardcoded real training results
        renderModelInfo({
            version: '1.0.0',
            trained_at: '2026-09-22T14:13:29Z',
            category_model: 'LinearSVC (CalibratedClassifierCV)',
            priority_model: 'LogisticRegression',
            category_metrics: {
                accuracy: 0.8578, f1_macro: 0.8605,
                per_class: REAL_CATEGORY_DISTRIBUTION ? Object.fromEntries(
                    Object.entries(REAL_CATEGORY_DISTRIBUTION).map(([k, v]) => [k, { support: v }])
                ) : {}
            },
            priority_metrics: { accuracy: 0.8293, f1_macro: 0.8242 },
        });
    }
}

function renderModelInfo(data) {
    const catMetrics = data.category_metrics || {};

    // Dashboard AI Accuracy
    const valAccuracy = document.getElementById('valAccuracy');
    if (valAccuracy) valAccuracy.textContent = ((catMetrics.accuracy || 0.8578) * 100).toFixed(1) + '%';

    const valModelName = document.getElementById('valModelName');
    if (valModelName && data.category_model) {
        valModelName.textContent = 'Model: ' + data.category_model.split(' ')[0];
    }

    // Total tickets from per_class support
    const valTotalTickets = document.getElementById('valTotalTickets');
    if (valTotalTickets && catMetrics.per_class) {
        const total = Object.values(catMetrics.per_class).reduce((sum, cls) => sum + (cls.support || 0), 0);
        if (total > 0) valTotalTickets.textContent = total.toLocaleString();
    }

    // Update API docs base URL
    const apiBaseUrl = document.getElementById('apiBaseUrl');
    if (apiBaseUrl) apiBaseUrl.textContent = API_BASE;
}

function refreshDashboard() {
    loadModelInfo();
    checkHealth();
    if (window.volumeChartInstance) { window.volumeChartInstance.destroy(); window.volumeChartInstance = null; }
    if (window.categoryChartInstance) { window.categoryChartInstance.destroy(); window.categoryChartInstance = null; }
    initDashboardCharts();
}

// ===========================================================================
// Dashboard Charts
// ===========================================================================
function initDashboardCharts() {
    // Volume Trend Chart (DEMO data)
    const volCtx = document.getElementById('volumeTrendChart');
    if (!volCtx || window.volumeChartInstance) return;

    window.volumeChartInstance = new Chart(volCtx, {
        type: 'line',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Today'],
            datasets: [
                {
                    label: 'Total',
                    data: [120, 150, 180, 140, 210, 90, 85, 230],
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 3
                },
                {
                    label: 'Resolved',
                    data: [100, 130, 160, 120, 180, 80, 75, 200],
                    borderColor: '#22c55e',
                    borderDash: [5, 5],
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 3
                },
                {
                    label: 'Critical',
                    data: [8, 12, 6, 15, 10, 3, 5, 14],
                    borderColor: '#ef4444',
                    borderDash: [2, 3],
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });

    // Category Distribution (REAL data)
    const catCtx = document.getElementById('categoryChart');
    if (!catCtx || window.categoryChartInstance) return;

    const catLabels = Object.keys(REAL_CATEGORY_DISTRIBUTION);
    const catValues = Object.values(REAL_CATEGORY_DISTRIBUTION);

    window.categoryChartInstance = new Chart(catCtx, {
        type: 'doughnut',
        data: {
            labels: catLabels,
            datasets: [{
                data: catValues,
                backgroundColor: CATEGORY_COLORS,
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 11 }, padding: 12 } }
            },
            cutout: '70%'
        }
    });
}

// ===========================================================================
// Model Training Page
// ===========================================================================
function populateModelComparison() {
    // Category model comparison (REAL cross-validation data)
    const catBody = document.getElementById('modelComparisonBody');
    if (!catBody) return;

    const catData = REAL_CV_DATA.category;
    catBody.innerHTML = Object.entries(catData).map(([name, m]) => {
        const isActive = name === BEST_MODELS.category.name;
        return `<tr class="${isActive ? 'highlight' : ''}">
            <td><strong>${name}</strong></td>
            <td>${(m.mean_accuracy * 100).toFixed(2)}%</td>
            <td>${(m.mean_f1 * 100).toFixed(2)}%</td>
            <td>±${(m.std_f1 * 100).toFixed(2)}%</td>
            <td>${isActive
                ? '<span class="status-pill connected">Active</span>'
                : '<span class="status-pill">Trained</span>'
            }</td>
        </tr>`;
    }).join('');

    // Priority model comparison
    const priBody = document.getElementById('priorityComparisonBody');
    if (!priBody) return;

    const priData = REAL_CV_DATA.priority;
    priBody.innerHTML = Object.entries(priData).map(([name, m]) => {
        const isActive = name === BEST_MODELS.priority.name;
        return `<tr class="${isActive ? 'highlight' : ''}">
            <td><strong>${name}</strong></td>
            <td>${(m.mean_accuracy * 100).toFixed(2)}%</td>
            <td>${(m.mean_f1 * 100).toFixed(2)}%</td>
            <td>±${(m.std_f1 * 100).toFixed(2)}%</td>
            <td>${isActive
                ? '<span class="status-pill connected">Active</span>'
                : '<span class="status-pill">Trained</span>'
            }</td>
        </tr>`;
    }).join('');

    // Update best model badge
    const bestModelName = document.getElementById('bestModelName');
    const bestModelScore = document.getElementById('bestModelScore');
    if (bestModelName) bestModelName.textContent = BEST_MODELS.category.name;
    if (bestModelScore) bestModelScore.textContent = (catData[BEST_MODELS.category.name].mean_f1 * 100).toFixed(1) + '%';
}

// DEMO: Simulate training (no real retraining — UI-only animation)
function simulateTraining() {
    const btn = document.getElementById('startTrainingBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span> Training...';

    setTimeout(() => {
        btn.disabled = false;
        btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start Training`;
        // Could update curves here
    }, 3000);
}

// DEMO: Training curves (empty/simulated)
function initTrainingCurves() {
    const accCtx = document.getElementById('accuracyCurveChart');
    const lossCtx = document.getElementById('lossCurveChart');
    if (!accCtx || !lossCtx) return;

    const labels = Array.from({ length: 5 }, (_, i) => `Fold ${i + 1}`);

    new Chart(accCtx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Accuracy',
                data: [0.845, 0.851, 0.848, 0.855, 0.853],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.3,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: 0.8, max: 0.9, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });

    new Chart(lossCtx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Loss',
                data: [0.42, 0.38, 0.35, 0.33, 0.31],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                fill: true,
                tension: 0.3,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: 0.2, max: 0.5, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });
}

// ===========================================================================
// Analytics Page (DEMO charts + REAL category data)
// ===========================================================================
function initAnalyticsCharts() {
    // Daily Volume (DEMO)
    const volCtx = document.getElementById('analyticsVolumeChart');
    if (volCtx && !window.analyticsVolumeInstance) {
        const days = ['Sep 16', 'Sep 17', 'Sep 18', 'Sep 19', 'Sep 20', 'Sep 21', 'Sep 22'];
        window.analyticsVolumeInstance = new Chart(volCtx, {
            type: 'bar',
            data: {
                labels: days,
                datasets: [{
                    label: 'Tickets',
                    data: [145, 178, 192, 165, 210, 88, 76],
                    backgroundColor: 'rgba(99, 102, 241, 0.6)',
                    borderColor: '#6366f1',
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    // Sentiment (DEMO / Future Feature)
    const sentCtx = document.getElementById('sentimentChart');
    if (sentCtx && !window.sentimentInstance) {
        window.sentimentInstance = new Chart(sentCtx, {
            type: 'bar',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [
                    { label: 'Positive', data: [45, 52, 58, 48, 62, 35, 30], backgroundColor: 'rgba(34, 197, 94, 0.6)', borderRadius: 4 },
                    { label: 'Neutral', data: [60, 65, 70, 55, 80, 40, 38], backgroundColor: 'rgba(99, 102, 241, 0.4)', borderRadius: 4 },
                    { label: 'Negative', data: [15, 18, 12, 22, 18, 8, 10], backgroundColor: 'rgba(239, 68, 68, 0.5)', borderRadius: 4 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top', labels: { font: { size: 11 } } } },
                scales: {
                    x: { stacked: true, grid: { display: false } },
                    y: { stacked: true, beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    }

    // Category Workload (REAL data)
    const workCtx = document.getElementById('categoryWorkloadChart');
    if (workCtx && !window.workloadInstance) {
        const cats = Object.keys(REAL_CATEGORY_DISTRIBUTION);
        const totals = Object.values(REAL_CATEGORY_DISTRIBUTION);
        // Simulate open/active/done splits from real totals
        const done = totals.map(t => Math.round(t * 0.7));
        const active = totals.map(t => Math.round(t * 0.2));
        const open = totals.map(t => Math.round(t * 0.1));

        window.workloadInstance = new Chart(workCtx, {
            type: 'bar',
            data: {
                labels: cats,
                datasets: [
                    { label: 'Done', data: done, backgroundColor: 'rgba(34, 197, 94, 0.5)', borderRadius: 4 },
                    { label: 'Active', data: active, backgroundColor: 'rgba(99, 102, 241, 0.5)', borderRadius: 4 },
                    { label: 'Open', data: open, backgroundColor: 'rgba(245, 158, 11, 0.5)', borderRadius: 4 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: { legend: { position: 'top', labels: { font: { size: 11 } } } },
                scales: {
                    x: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { stacked: true, grid: { display: false } }
                }
            }
        });
    }

    // Top category
    const topCatEl = document.getElementById('analyticsTopCategory');
    if (topCatEl) {
        const sorted = Object.entries(REAL_CATEGORY_DISTRIBUTION).sort((a, b) => b[1] - a[1]);
        topCatEl.textContent = sorted[0][0];
    }
}

function setAnalyticsRange(btn, range) {
    btn.parentElement.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    // In a real app, this would re-fetch data for the selected time range
}

// ===========================================================================
// Team Page (DEMO data)
// ===========================================================================
function populateAgentList() {
    const container = document.getElementById('agentList');
    if (!container) return;

    const agents = [
        { name: 'Sarah Chen', role: 'Senior Agent · Hardware', tickets: 24, score: 96, workload: 78, color: '#6366f1' },
        { name: 'James Wilson', role: 'Agent · Access & Security', tickets: 18, score: 91, workload: 65, color: '#8b5cf6' },
        { name: 'Priya Sharma', role: 'Senior Agent · HR Support', tickets: 31, score: 94, workload: 88, color: '#ec4899' },
        { name: 'Alex Thompson', role: 'Agent · General Support', tickets: 15, score: 87, workload: 52, color: '#f97316' },
        { name: 'Maria Garcia', role: 'Lead Agent · Billing', tickets: 22, score: 98, workload: 71, color: '#22c55e' },
    ];

    container.innerHTML = agents.map(a => {
        const wColor = a.workload > 80 ? '#ef4444' : a.workload > 60 ? '#f59e0b' : '#22c55e';
        return `<div class="agent-item">
            <div class="agent-avatar" style="background:${a.color}">${a.name.split(' ').map(n => n[0]).join('')}</div>
            <div class="agent-info">
                <div class="agent-name">${a.name}</div>
                <div class="agent-role">${a.role}</div>
            </div>
            <div class="agent-stats">
                <div class="agent-stat"><span class="agent-stat-value">${a.tickets}</span><span class="agent-stat-label">Tickets</span></div>
                <div class="agent-stat"><span class="agent-stat-value">${a.score}%</span><span class="agent-stat-label">Score</span></div>
                <div class="agent-stat">
                    <div class="workload-bar"><div class="workload-bar-fill" style="width:${a.workload}%;background:${wColor}"></div></div>
                    <span class="agent-stat-label">${a.workload}%</span>
                </div>
            </div>
        </div>`;
    }).join('');
}

// ===========================================================================
// Admin Panel (DEMO data)
// ===========================================================================
function populateActivityFeed() {
    const feed = document.getElementById('activityFeed');
    if (!feed) return;

    const activities = [
        { user: 'Admin User', initial: 'A', color: '#6366f1', action: 'Updated AI model to <strong>LinearSVC</strong>', time: '2 minutes ago' },
        { user: 'Sarah Chen', initial: 'SC', color: '#8b5cf6', action: 'Exported ticket report (1,247 records)', time: '15 minutes ago' },
        { user: 'System', initial: 'S', color: '#22c55e', action: 'Model retraining completed — Accuracy: <strong>85.8%</strong>', time: '1 hour ago' },
        { user: 'James Wilson', initial: 'JW', color: '#f97316', action: 'Uploaded new dataset: <strong>all_tickets_v3.csv</strong>', time: '3 hours ago' },
        { user: 'System', initial: '!', color: '#ef4444', action: 'Failed login attempt from IP 192.168.1.105', time: '5 hours ago' },
        { user: 'Priya Sharma', initial: 'PS', color: '#ec4899', action: 'Modified SLA configuration for <strong>Critical</strong> priority', time: '1 day ago' },
    ];

    feed.innerHTML = activities.map(a => `
        <div class="activity-item">
            <div class="activity-avatar" style="background:${a.color}">${a.initial}</div>
            <div class="activity-content">
                <div class="activity-text"><strong>${a.user}</strong> — ${a.action}</div>
                <div class="activity-time">${a.time}</div>
            </div>
        </div>
    `).join('');
}

function populateAuditLog() {
    const log = document.getElementById('auditLog');
    if (!log) return;

    const entries = [
        { user: 'Admin User', initial: 'A', color: '#6366f1', action: 'Changed model from RandomForest to <strong>LinearSVC</strong>', time: '2026-09-23 19:30:00' },
        { user: 'Admin User', initial: 'A', color: '#6366f1', action: 'Created API key: <strong>sk-prod-***3f9a</strong>', time: '2026-09-20 10:15:00' },
        { user: 'System', initial: 'S', color: '#22c55e', action: 'Training pipeline completed — 4 models evaluated', time: '2026-09-22 14:13:29' },
        { user: 'James Wilson', initial: 'JW', color: '#f97316', action: 'Dataset upload: <strong>48,840 records</strong> processed', time: '2026-09-22 14:00:00' },
        { user: 'System', initial: '!', color: '#ef4444', action: 'Rate limit exceeded from IP 10.0.0.42 (300 req/min)', time: '2026-09-21 08:45:00' },
    ];

    log.innerHTML = entries.map(e => `
        <div class="activity-item">
            <div class="activity-avatar" style="background:${e.color}">${e.initial}</div>
            <div class="activity-content">
                <div class="activity-text"><strong>${e.user}</strong> — ${e.action}</div>
                <div class="activity-time">${e.time}</div>
            </div>
        </div>
    `).join('');
}

function switchAdminTab(btn, tabId) {
    btn.parentElement.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.admin-tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
}

// ===========================================================================
// Settings Page
// ===========================================================================
function switchSettingsTab(btn, tabId) {
    btn.parentElement.querySelectorAll('.settings-nav-item').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.settings-tab').forEach(c => c.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
}

// ===========================================================================
// API Docs
// ===========================================================================
function switchCodeTab(btn, tabId) {
    btn.parentElement.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.code-tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
}

function copyCodeBlock(tabId) {
    const content = document.querySelector(`#${tabId} code`);
    if (content) copyToClipboard(content.textContent);
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        // Brief visual feedback could be added here
    }).catch(() => {
        // Fallback for older browsers
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    });
}

async function measureApiLatency() {
    const el = document.getElementById('apiLatency');
    if (!el) return;

    try {
        const start = performance.now();
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        await fetch(`${API_BASE}/health`, { signal: controller.signal });
        clearTimeout(timeoutId);
        const latency = Math.round(performance.now() - start);
        el.textContent = `Avg latency: ${latency}ms`;
    } catch (e) {
        el.textContent = 'Avg latency: N/A (offline)';
    }
}

// ===========================================================================
// Utilities
// ===========================================================================
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
