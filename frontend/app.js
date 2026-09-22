/**
 * Support Ticket Classifier — Frontend Application
 *
 * Handles:
 * - Single ticket classification with confidence visualization
 * - Batch CSV upload with sortable/filterable results table
 * - Dashboard with charts and model metrics
 *
 * API base URL is configurable via window.API_BASE_URL or defaults to localhost:8000.
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
// Auto-detect: if running on Vercel or any remote host, use the deployed backend.
// Override via window.API_BASE_URL if needed.
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

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let batchData = [];           // Current batch prediction results
let batchSortKey = 'index';
let batchSortAsc = true;
let categoryChartInstance = null;
let priorityChartInstance = null;
let demoMode = false;         // Activates when backend API is unreachable

// ---------------------------------------------------------------------------
// Demo Mode — client-side keyword classifier (fallback when no backend)
// ---------------------------------------------------------------------------
const DEMO_RULES = [
    { category: 'Hardware',              keywords: ['laptop', 'screen', 'monitor', 'keyboard', 'mouse', 'printer', 'cable', 'pc', 'computer', 'broken', 'replacement', 'hard drive', 'battery', 'charger', 'display', 'dock', 'headset', 'webcam', 'device'] },
    { category: 'Access',                keywords: ['password', 'login', 'access', 'locked', 'reset', 'vpn', 'account', 'credential', 'authenticate', 'sso', 'mfa', 'token', 'expire', 'permission denied', 'cannot access', 'sign in'] },
    { category: 'Administrative rights',  keywords: ['admin', 'administrator', 'privilege', 'elevated', 'rights', 'sudo', 'root', 'permission', 'install software', 'group policy'] },
    { category: 'HR Support',            keywords: ['onboarding', 'offboarding', 'new employee', 'leave', 'payroll', 'benefits', 'hr', 'human resources', 'hire', 'termination', 'contract', 'salary'] },
    { category: 'Storage',               keywords: ['storage', 'disk space', 'file share', 'drive', 'backup', 'quota', 'shared folder', 'onedrive', 'sharepoint', 'cloud storage', 'nas'] },
    { category: 'Purchase',              keywords: ['purchase', 'order', 'buy', 'procurement', 'license', 'subscription', 'invoice', 'budget', 'vendor', 'quote', 'cost'] },
    { category: 'Internal Project',      keywords: ['project', 'timeline', 'milestone', 'resource', 'planning', 'sprint', 'deliverable', 'stakeholder', 'scope'] },
    { category: 'Miscellaneous',         keywords: ['help', 'question', 'information', 'general', 'other', 'inquiry', 'support', 'issue'] },
];

const URGENCY_WORDS = ['urgent', 'asap', 'critical', 'emergency', 'immediately', 'blocked', 'down', 'crashed', 'broken', 'not working', 'deadline', 'production'];

function demoPrediction(text) {
    const lower = text.toLowerCase();
    // Score each category by keyword matches
    let scores = DEMO_RULES.map(rule => {
        let score = rule.keywords.filter(kw => lower.includes(kw)).length;
        return { category: rule.category, score };
    });
    scores.sort((a, b) => b.score - a.score);
    // Default to Miscellaneous if no matches
    const bestCat = scores[0].score > 0 ? scores[0].category : 'Miscellaneous';
    // Generate realistic-looking probabilities
    const totalScore = Math.max(scores.reduce((s, x) => s + x.score, 0), 1);
    const catProbs = {};
    scores.forEach(s => { catProbs[s.category] = Math.max(0.01, s.score / totalScore); });
    // Normalize
    const probSum = Object.values(catProbs).reduce((a, b) => a + b, 0);
    Object.keys(catProbs).forEach(k => { catProbs[k] = Math.round((catProbs[k] / probSum) * 10000) / 10000; });
    const catConf = catProbs[bestCat];

    // Priority from urgency keywords + category
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

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initTextarea();
    initFileUpload();
    checkHealth();
    loadModelInfo();
});

// ---------------------------------------------------------------------------
// Tab Navigation
// ---------------------------------------------------------------------------
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            // Deactivate all
            document.querySelectorAll('.tab-btn').forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            // Activate clicked
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
            const tabId = btn.dataset.tab;
            document.getElementById(`content-${tabId}`).classList.add('active');
        });
    });
}

// ---------------------------------------------------------------------------
// Textarea
// ---------------------------------------------------------------------------
function initTextarea() {
    const textarea = document.getElementById('ticketText');
    const charCount = document.getElementById('charCount');
    textarea.addEventListener('input', () => {
        charCount.textContent = `${textarea.value.length} chars`;
    });
    // Enable Ctrl+Enter to classify
    textarea.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            classifySingle();
        }
    });
}

// ---------------------------------------------------------------------------
// Health Check
// ---------------------------------------------------------------------------
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');
        if (data.model_loaded) {
            demoMode = false;
            dot.classList.remove('error');
            text.textContent = 'API Connected • Models Loaded';
        } else {
            demoMode = true;
            dot.classList.add('error');
            text.textContent = 'API Connected • Models Not Loaded — Demo Mode';
        }
    } catch (e) {
        demoMode = true;
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');
        dot.style.background = '#f59e0b';
        text.textContent = 'Demo Mode — predictions generated locally';
    }
}

// ---------------------------------------------------------------------------
// Error Handling
// ---------------------------------------------------------------------------
function showError(message) {
    const banner = document.getElementById('errorBanner');
    document.getElementById('errorText').textContent = message;
    banner.classList.add('visible');
}

function hideError() {
    document.getElementById('errorBanner').classList.remove('visible');
}

// ---------------------------------------------------------------------------
// Single Ticket Classification
// ---------------------------------------------------------------------------
async function classifySingle() {
    const text = document.getElementById('ticketText').value.trim();
    if (!text) {
        showError('Please enter ticket text before classifying.');
        return;
    }
    if (text.length < 3) {
        showError('Ticket text must be at least 3 characters long.');
        return;
    }

    hideError();
    const btn = document.getElementById('classifyBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span> Classifying...';

    try {
        if (demoMode) {
            // Demo mode — classify locally with keyword matching
            await new Promise(r => setTimeout(r, 400)); // Simulate brief delay
            const data = demoPrediction(text);
            displaySingleResult(data);
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

            const data = await res.json();
            displaySingleResult(data);
        }
    } catch (e) {
        showError(`Classification failed: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Classify Ticket`;
    }
}

function displaySingleResult(data) {
    const container = document.getElementById('singleResults');
    container.classList.add('visible');

    // Category
    document.getElementById('resultCategory').textContent = data.category;
    document.getElementById('catConfText').textContent = `${(data.category_confidence * 100).toFixed(1)}%`;
    document.getElementById('catConfBar').style.width = `${(data.category_confidence * 100)}%`;

    // Priority with badge
    const priEl = document.getElementById('resultPriority');
    const priClass = data.priority.toLowerCase();
    priEl.innerHTML = `<span class="priority-badge ${priClass}">${data.priority}</span>`;
    document.getElementById('priConfText').textContent = `${(data.priority_confidence * 100).toFixed(1)}%`;
    document.getElementById('priConfBar').style.width = `${(data.priority_confidence * 100)}%`;

    // Category probability bars
    if (data.category_probabilities) {
        renderProbBars('catProbBars', data.category_probabilities);
    }

    // Priority probability bars
    if (data.priority_probabilities) {
        renderProbBars('priProbBars', data.priority_probabilities);
    }

    // Scroll to results
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderProbBars(containerId, probs) {
    const container = document.getElementById(containerId);
    // Sort by probability descending
    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    const maxProb = sorted[0][1];

    container.innerHTML = sorted.map(([label, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const barWidth = maxProb > 0 ? (prob / maxProb * 100) : 0;
        return `
            <div class="prob-item">
                <span class="prob-label">${label}</span>
                <div class="prob-bar">
                    <div class="prob-bar-fill" style="width: ${barWidth}%"></div>
                </div>
                <span class="prob-value">${pct}%</span>
            </div>
        `;
    }).join('');
}

function clearSingle() {
    document.getElementById('ticketText').value = '';
    document.getElementById('charCount').textContent = '0 chars';
    document.getElementById('singleResults').classList.remove('visible');
    hideError();
}

// ---------------------------------------------------------------------------
// Batch Upload
// ---------------------------------------------------------------------------
function initFileUpload() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('csvFileInput');

    // Drag and drop
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleCSVFile(e.dataTransfer.files[0]);
        }
    });

    // Click upload
    input.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleCSVFile(e.target.files[0]);
        }
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

    // Parse CSV client-side to extract text column
    const text = await file.text();
    Papa.parse(text, {
        header: true,
        skipEmptyLines: true,
        complete: async (results) => {
            const columns = results.meta.fields;
            // Auto-detect text column
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
        error: (err) => {
            showError(`Failed to parse CSV: ${err.message}`);
        }
    });
}

async function classifyBatch(texts) {
    const uploadZone = document.getElementById('uploadZone');
    uploadZone.innerHTML = `<span class="loading-spinner" style="width:32px;height:32px;border-width:3px;"></span><p style="margin-top:16px;">Classifying ${texts.length} tickets...</p>`;

    try {
        let predictions;
        if (demoMode) {
            // Demo mode — classify locally
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

        // Populate category filter
        const categories = [...new Set(batchData.map(d => d.category))].sort();
        const filterCat = document.getElementById('filterCategory');
        filterCat.innerHTML = '<option value="">All Categories</option>' +
            categories.map(c => `<option value="${c}">${c}</option>`).join('');

        renderBatchTable();
        document.getElementById('batchResults').style.display = 'block';

        // Update dashboard charts with batch data
        updateDashboardCharts();

    } catch (e) {
        showError(`Batch classification failed: ${e.message}`);
    } finally {
        // Reset upload zone
        uploadZone.innerHTML = `
            <input type="file" accept=".csv" id="csvFileInput" aria-label="Upload CSV file">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
            <p>Drop your CSV file here or <strong>click to browse</strong></p>
            <span class="hint">CSV with a "text", "Document", or "description" column • Max 1000 rows</span>
        `;
        initFileUpload(); // Re-attach listeners
    }
}

function renderBatchTable() {
    const catFilter = document.getElementById('filterCategory').value;
    const priFilter = document.getElementById('filterPriority').value;

    let filtered = batchData;
    if (catFilter) filtered = filtered.filter(d => d.category === catFilter);
    if (priFilter) filtered = filtered.filter(d => d.priority === priFilter);

    // Sort
    filtered.sort((a, b) => {
        let va = a[batchSortKey], vb = b[batchSortKey];
        if (typeof va === 'string') { va = va.toLowerCase(); vb = vb.toLowerCase(); }
        if (va < vb) return batchSortAsc ? -1 : 1;
        if (va > vb) return batchSortAsc ? 1 : -1;
        return 0;
    });

    const tbody = document.getElementById('batchTableBody');
    tbody.innerHTML = filtered.map(row => {
        const priClass = row.priority.toLowerCase();
        const truncatedText = row.text.length > 100 ? row.text.substring(0, 100) + '...' : row.text;
        return `
            <tr>
                <td>${row.index}</td>
                <td title="${escapeHtml(row.text)}">${escapeHtml(truncatedText)}</td>
                <td>${row.category}</td>
                <td>${(row.category_confidence * 100).toFixed(1)}%</td>
                <td><span class="priority-badge ${priClass}">${row.priority}</span></td>
                <td>${(row.priority_confidence * 100).toFixed(1)}%</td>
            </tr>
        `;
    }).join('');

    document.getElementById('batchCount').textContent = `${filtered.length} of ${batchData.length} tickets`;
}

function sortBatchTable(key) {
    if (batchSortKey === key) {
        batchSortAsc = !batchSortAsc;
    } else {
        batchSortKey = key;
        batchSortAsc = true;
    }
    renderBatchTable();
}

function filterBatchTable() {
    renderBatchTable();
}

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
    document.getElementById('csvFileInput').value = '';
    hideError();
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------
async function loadModelInfo() {
    try {
        const res = await fetch(`${API_BASE}/model-info`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderModelInfo(data);
    } catch (e) {
        // Demo mode — show actual training results from the completed pipeline
        const demoInfo = {
            version: '1.0.0',
            trained_at: '2026-09-22T14:13:29Z',
            category_model: 'LinearSVC (CalibratedClassifierCV)',
            priority_model: 'LogisticRegression',
            category_metrics: {
                accuracy: 0.8578, f1_macro: 0.8605, f1_weighted: 0.8579,
                precision_macro: 0.8801, recall_macro: 0.8436,
                per_class: {
                    'Access': { precision: 0.90, recall: 0.89, f1: 0.89, support: 1425 },
                    'Administrative rights': { precision: 0.85, recall: 0.71, f1: 0.77, support: 352 },
                    'HR Support': { precision: 0.86, recall: 0.86, f1: 0.86, support: 2183 },
                    'Hardware': { precision: 0.82, recall: 0.87, f1: 0.84, support: 2724 },
                    'Internal Project': { precision: 0.90, recall: 0.86, f1: 0.88, support: 424 },
                    'Miscellaneous': { precision: 0.83, recall: 0.83, f1: 0.83, support: 1412 },
                    'Purchase': { precision: 0.96, recall: 0.89, f1: 0.92, support: 493 },
                    'Storage': { precision: 0.92, recall: 0.85, f1: 0.88, support: 555 },
                }
            },
            priority_metrics: {
                accuracy: 0.8293, f1_macro: 0.8242, f1_weighted: 0.8297,
                precision_macro: 0.8248, recall_macro: 0.8238,
                per_class: {
                    'High': { precision: 0.75, recall: 0.77, f1: 0.76, support: 2888 },
                    'Medium': { precision: 0.88, recall: 0.88, f1: 0.88, support: 3882 },
                    'Low': { precision: 0.85, recall: 0.82, f1: 0.83, support: 2798 },
                }
            },
        };
        renderModelInfo(demoInfo);
    }
}

function renderModelInfo(data) {
    const container = document.getElementById('modelMetrics');
    const catMetrics = data.category_metrics || {};
    const priMetrics = data.priority_metrics || {};

    const trainedDate = data.trained_at ? new Date(data.trained_at).toLocaleDateString() : 'N/A';

    container.innerHTML = `
        <div class="metric-card">
            <div class="metric-value">${data.version || '—'}</div>
            <div class="metric-label">Model Version</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${trainedDate}</div>
            <div class="metric-label">Trained</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${((catMetrics.accuracy || 0) * 100).toFixed(1)}%</div>
            <div class="metric-label">Category Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${((catMetrics.f1_macro || 0) * 100).toFixed(1)}%</div>
            <div class="metric-label">Category F1 (macro)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${((priMetrics.accuracy || 0) * 100).toFixed(1)}%</div>
            <div class="metric-label">Priority Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${((priMetrics.f1_macro || 0) * 100).toFixed(1)}%</div>
            <div class="metric-label">Priority F1 (macro)</div>
        </div>
    `;

    // Per-class metrics table
    if (catMetrics.per_class) {
        renderClassMetrics(catMetrics, priMetrics);
    }

    // Initialize empty charts (will populate with batch data or model info per-class)
    if (catMetrics.per_class) {
        const catLabels = Object.keys(catMetrics.per_class);
        const catSupport = catLabels.map(l => catMetrics.per_class[l].support);
        createCategoryChart(catLabels, catSupport);
    }

    if (priMetrics.per_class) {
        const priLabels = Object.keys(priMetrics.per_class);
        const priSupport = priLabels.map(l => priMetrics.per_class[l].support);
        createPriorityChart(priLabels, priSupport);
    }
}

function renderClassMetrics(catMetrics, priMetrics) {
    const card = document.getElementById('classMetricsCard');
    card.style.display = 'block';

    let html = '<h4 style="color:var(--text-secondary);margin-bottom:var(--space-md);">Category Classifier</h4>';
    html += buildMetricsTable(catMetrics.per_class);

    if (priMetrics.per_class) {
        html += '<h4 style="color:var(--text-secondary);margin-top:var(--space-xl);margin-bottom:var(--space-md);">Priority Classifier</h4>';
        html += buildMetricsTable(priMetrics.per_class);
    }

    document.getElementById('classMetricsContent').innerHTML = html;
}

function buildMetricsTable(perClass) {
    let html = `<div class="table-container"><table class="data-table">
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead><tbody>`;
    for (const [cls, m] of Object.entries(perClass)) {
        html += `<tr>
            <td>${cls}</td>
            <td>${(m.precision * 100).toFixed(1)}%</td>
            <td>${(m.recall * 100).toFixed(1)}%</td>
            <td>${(m.f1 * 100).toFixed(1)}%</td>
            <td>${m.support}</td>
        </tr>`;
    }
    html += '</tbody></table></div>';
    return html;
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------
function createCategoryChart(labels, values) {
    const ctx = document.getElementById('categoryChart').getContext('2d');
    if (categoryChartInstance) categoryChartInstance.destroy();
    categoryChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: CATEGORY_COLORS.slice(0, labels.length),
                borderWidth: 0,
                hoverOffset: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 16, usePointStyle: true, pointStyleWidth: 10, font: { size: 11 } },
                },
            },
        },
    });
}

function createPriorityChart(labels, values) {
    const ctx = document.getElementById('priorityChart').getContext('2d');
    if (priorityChartInstance) priorityChartInstance.destroy();
    const colors = labels.map(l => PRIORITY_COLORS[l] || '#6366f1');
    priorityChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Count',
                data: values,
                backgroundColor: colors.map(c => c + '99'),
                borderColor: colors,
                borderWidth: 1,
                borderRadius: 6,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
            },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } },
                x: { grid: { display: false } },
            },
        },
    });
}

function updateDashboardCharts() {
    if (batchData.length === 0) return;

    // Category distribution from batch
    const catCounts = {};
    const priCounts = {};
    batchData.forEach(d => {
        catCounts[d.category] = (catCounts[d.category] || 0) + 1;
        priCounts[d.priority] = (priCounts[d.priority] || 0) + 1;
    });

    createCategoryChart(Object.keys(catCounts), Object.values(catCounts));
    createPriorityChart(Object.keys(priCounts), Object.values(priCounts));
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
