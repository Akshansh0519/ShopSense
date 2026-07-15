/* ═══════════════════════════════════════
   ShopSense Frontend — app.js
   ═══════════════════════════════════════ */

const API_BASE = 'https://shopsense-pkys.onrender.com';
const SAMPLE_USER_IDS = ['user_00001', 'user_00042', 'user_00100', 'user_00250', 'user_00500', 'user_00777', 'user_00999'];

// ── Nav scroll effect ─────────────────────────────────────────────────────────
const nav = document.getElementById('nav');
window.addEventListener('scroll', () => {
  nav.classList.toggle('scrolled', window.scrollY > 30);
});

// ── Stat counter animation ────────────────────────────────────────────────────
function animateCounters() {
  document.querySelectorAll('.stat-number').forEach(el => {
    const target   = parseInt(el.dataset.target, 10);
    const suffix   = el.dataset.suffix || '';
    const duration = 1200;
    const start    = performance.now();
    function step(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased    = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.floor(eased * target) + (progress === 1 ? suffix : '');
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });
}

const statObserver = new IntersectionObserver(entries => {
  entries.forEach(e => { if (e.isIntersecting) { animateCounters(); statObserver.disconnect(); } });
}, { threshold: 0.5 });
statObserver.observe(document.querySelector('.stat-bar'));

// ── Demo Controls ─────────────────────────────────────────────────────────────
const userIdInput    = document.getElementById('userIdInput');
const kSlider        = document.getElementById('kSlider');
const kValue         = document.getElementById('kValue');
const categorySelect = document.getElementById('categorySelect');
const apiKeyInput    = document.getElementById('apiKeyInput');
const fetchBtn       = document.getElementById('fetchBtn');
const fetchBtnText   = document.getElementById('fetchBtnText');
const btnSpinner     = document.getElementById('btnSpinner');
const resultsList    = document.getElementById('resultsList');
const responseMeta   = document.getElementById('responseMeta');
const requestPreview = document.getElementById('requestPreview');
const randomBtn      = document.getElementById('randomBtn');

// Set a random user ID on load
userIdInput.value = SAMPLE_USER_IDS[Math.floor(Math.random() * SAMPLE_USER_IDS.length)];

kSlider.addEventListener('input', () => {
  kValue.textContent = kSlider.value;
  updatePreview();
});

[userIdInput, categorySelect].forEach(el => el.addEventListener('input', updatePreview));

function updatePreview() {
  const uid = userIdInput.value.trim() || '—';
  const k   = kSlider.value;
  const cat = categorySelect.value ? `&category=${categorySelect.value}` : '';
  requestPreview.textContent = `GET /recommendations/${uid}?k=${k}${cat}`;
}
updatePreview();

randomBtn.addEventListener('click', () => {
  userIdInput.value = SAMPLE_USER_IDS[Math.floor(Math.random() * SAMPLE_USER_IDS.length)];
  updatePreview();
});

// ── Fetch Recommendations ─────────────────────────────────────────────────────
fetchBtn.addEventListener('click', async () => {
  const userId   = userIdInput.value.trim();
  const k        = kSlider.value;
  const category = categorySelect.value;
  const apiKey   = apiKeyInput.value.trim();

  if (!userId) {
    showError('Please enter a User ID.');
    return;
  }

  setLoading(true);
  responseMeta.textContent = '';

  let url = `${API_BASE}/recommendations/${encodeURIComponent(userId)}?k=${k}`;
  if (category) url += `&category=${encodeURIComponent(category)}`;

  const headers = { 'Content-Type': 'application/json' };
  if (apiKey) headers['X-API-Key'] = apiKey;

  const t0 = performance.now();

  try {
    const res  = await fetch(url, { headers });
    const data = await res.json();
    const ms   = (performance.now() - t0).toFixed(0);

    if (!res.ok) {
      showError(data.detail || `Error ${res.status}`);
      return;
    }

    const cached = data.cached ? '⚡ Cached' : '🔄 Fresh';
    const lat    = data.latency_ms ? `${data.latency_ms.toFixed(1)}ms API` : `${ms}ms`;
    responseMeta.innerHTML = `
      <span style="color:var(--success)">${cached}</span>
      &nbsp;·&nbsp;${lat}
      &nbsp;·&nbsp;<span style="font-family:var(--mono)">${data.model_version || 'hybrid'}</span>
    `;

    renderResults(data.recommendations || []);
  } catch (err) {
    showError('API is unreachable or waking up from sleep. Please wait ~50 seconds and try again.');
  } finally {
    setLoading(false);
  }
});

function setLoading(on) {
  fetchBtnText.classList.toggle('hidden', on);
  btnSpinner.classList.toggle('hidden', !on);
  fetchBtn.disabled = on;
}

function showError(msg) {
  resultsList.innerHTML = `<div class="error-state">⚠ ${msg}</div>`;
}

function renderResults(items) {
  if (!items.length) {
    resultsList.innerHTML = '<div class="results-placeholder"><div class="placeholder-icon">◈</div><p>No results returned.</p></div>';
    return;
  }

  const html = items.map((item, i) => {
    const score  = typeof item.score === 'number' ? item.score : 0;
    const width  = Math.min(100, Math.round(score * 100));
    const reason = item.reason || '';

    const signals = item.signals || {};
    const signalHtml = Object.entries(signals)
      .filter(([, v]) => v > 0.01)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([key, val]) => `<span class="signal-pill ${key}">${key}: ${val.toFixed(2)}</span>`)
      .join('');

    return `
      <div class="result-item" style="animation-delay: ${i * 0.05}s">
        <div class="result-rank">#${item.rank}</div>
        <div class="result-body">
          <div class="result-id">Item ${item.item_id}</div>
          ${reason ? `<div class="result-reason">${reason}</div>` : ''}
          ${signalHtml ? `<div class="result-signals">${signalHtml}</div>` : ''}
        </div>
        <div class="result-score-bar">
          <div class="result-bar-track">
            <div class="result-bar-fill" style="width: ${width}%"></div>
          </div>
          <div class="result-score-val">${score.toFixed(3)}</div>
        </div>
      </div>
    `;
  }).join('');

  resultsList.innerHTML = html;
}

// ── Metrics Section ───────────────────────────────────────────────────────────
const metricsStatus    = document.getElementById('metricsStatus');
const metricsTableWrap = document.getElementById('metricsTableWrap');
const modelTabs        = document.getElementById('modelTabs');
const metricsGrid      = document.getElementById('metricsGrid');

let allMetrics  = null;
let activeModel = null;

async function loadMetrics() {
  try {
    const apiKey  = apiKeyInput ? apiKeyInput.value.trim() : '';
    const headers = {};
    if (apiKey) headers['X-API-Key'] = apiKey;

    // Fetch real metrics from the API
    const [metricsRes, modelRes] = await Promise.all([
      fetch(`${API_BASE}/metrics`).catch(() => null),
      fetch(`${API_BASE}/models/current`, { headers }).catch(() => null),
    ]);

    const modelData = modelRes && modelRes.ok ? await modelRes.json() : null;

    if (metricsRes && metricsRes.ok) {
      allMetrics = await metricsRes.json();
    } else {
      allMetrics = {
        metrics: {
          'hybrid_mmr': { 'ndcg@10': 0.0, 'map@10': 0.0, 'recall@10': 0.0 },
          'als':        { 'ndcg@10': 0.0, 'map@10': 0.0, 'recall@10': 0.0 },
          'bpr':        { 'ndcg@10': 0.0, 'map@10': 0.0, 'recall@10': 0.0 },
          'popularity': { 'ndcg@10': 0.0, 'map@10': 0.0, 'recall@10': 0.0 },
        }
      };
    }

    let serverModel = modelData?.active_model;
    if (serverModel && !allMetrics.metrics[serverModel]) {
      // Find a key that the server model starts with (e.g., "hybrid_mmr_v1" -> "hybrid_mmr")
      const match = Object.keys(allMetrics.metrics).find(k => serverModel.startsWith(k));
      if (match) serverModel = match;
    }
    
    activeModel = serverModel || Object.keys(allMetrics.metrics)[0];

    renderTabs();
    renderMetricCards(activeModel);
    metricsStatus.classList.add('hidden');
    metricsTableWrap.classList.remove('hidden');

  } catch (err) {
    metricsStatus.innerHTML = `<div style="color:var(--text-muted);font-size:13px;text-align:center;padding:20px;">
      API unreachable or waking up from sleep. (If running locally, ensure <code style="font-family:var(--mono);color:var(--accent-2)">python scripts/evaluate_all.py</code> was run).
    </div>`;
  }
}

function renderTabs() {
  const models = Object.keys(allMetrics.metrics);
  modelTabs.innerHTML = models.map(m => `
    <button class="model-tab ${m === activeModel ? 'active' : ''}" data-model="${m}">
      ${m.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
    </button>
  `).join('');
  modelTabs.querySelectorAll('.model-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      activeModel = btn.dataset.model;
      modelTabs.querySelectorAll('.model-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderMetricCards(activeModel);
    });
  });
}

function renderMetricCards(model) {
  const data    = allMetrics.metrics[model] || {};
  const isEmpty = Object.values(data).every(v => v === 0);

  const metricNames = {
    'ndcg@10':        'NDCG@10',
    'map@10':         'MAP@10',
    'recall@10':      'Recall@10',
    'precision@10':   'Precision@10',
    'coverage@10':    'Coverage@10',
    'novelty@10':     'Novelty@10',
    'diversity@10':   'Diversity@10',
    'serendipity@10': 'Serendipity@10',
  };

  // Only render metrics that actually exist in the data
  const available = Object.entries(metricNames).filter(([key]) => key in data);

  metricsGrid.innerHTML = available.map(([key, label]) => {
    const val     = data[key] || 0;
    // Scale bar: ranking metrics are 0–1, novelty is 0–20ish, so cap at reasonable %
    const isLarge = val > 1;
    const pct     = isLarge ? Math.min(100, (val / 20) * 100).toFixed(0) : (val * 100).toFixed(0);
    const display = isEmpty ? '—' : val.toFixed(4);
    return `
      <div class="metric-card">
        <div class="metric-name">${label}</div>
        <div class="metric-value">${display}</div>
        <div class="metric-bar-wrap">
          <div class="metric-bar" style="width: ${isEmpty ? 0 : pct}%"></div>
        </div>
      </div>
    `;
  }).join('');

  if (isEmpty) {
    metricsGrid.innerHTML += `
      <div class="metric-card" style="grid-column: 1/-1; text-align:center; padding: 20px;">
        <div style="color:var(--text-muted); font-size:13px;">
          No metric data yet. Run <code style="font-family:var(--mono);color:var(--accent-2)">python scripts/evaluate_all.py</code>
        </div>
      </div>
    `;
  }
}

const metricsObserver = new IntersectionObserver(entries => {
  entries.forEach(e => { if (e.isIntersecting) { loadMetrics(); metricsObserver.disconnect(); } });
}, { threshold: 0.2 });
metricsObserver.observe(document.getElementById('metrics'));

// ── Initial preview update ────────────────────────────────────────────────────
updatePreview();

// ── Backend Status Banner ───────────────────────────────────────────────────
const statusBanner = document.getElementById('backendStatusBanner');
const statusBannerInner = document.querySelector('.status-banner-inner');
const statusTitle = document.getElementById('statusTitle');
const statusDesc = document.getElementById('statusDesc');
const statusBadge = document.getElementById('statusBadge');
const statusAttempt = document.getElementById('statusAttempt');
const statusDismissBtn = document.getElementById('statusDismissBtn');

if (statusBanner) {
  let attempt = 1;
  let checkInterval = null;
  let isDismissed = false;

  // Show banner by default on page load to test Render free tier wake-up
  statusBanner.classList.remove('hidden');

  statusDismissBtn.addEventListener('click', () => {
    isDismissed = true;
    statusBanner.classList.add('hidden');
    if (checkInterval) clearInterval(checkInterval);
  });

  const checkHealth = async () => {
    if (isDismissed) return;
    try {
      const res = await fetch(`${API_BASE}/ready`, { method: 'GET' });
      if (res.ok) {
        // API IS ONLINE!
        if (checkInterval) clearInterval(checkInterval);
        statusBannerInner.classList.add('is-online');
        
        statusTitle.textContent = 'API is Online!';
        statusAttempt.style.display = 'none';
        
        statusDesc.innerHTML = 'The ShopSense Recommendation API has returned <strong style="color:#34d399">200 OK</strong> and is fully online!<br/>You can now query personalized recommendations and test cold-start fallbacks instantly.';
        
        statusBadge.textContent = '🟢 ONLINE (200 OK)';
        statusBadge.classList.remove('checking');
        statusBadge.classList.add('online');

        // Transform dismiss button into a primary continue button
        statusDismissBtn.textContent = 'Got It, Continue to ShopSense ◈';
        statusDismissBtn.classList.add('status-continue-btn');
        document.getElementById('statusIntervalTxt').style.display = 'none';
      } else {
        throw new Error('Not ok');
      }
    } catch (e) {
      // Still waking up
      attempt++;
      statusAttempt.textContent = `Attempt #${attempt}`;
    }
  };

  // Immediate first check, then every 3s
  checkHealth();
  checkInterval = setInterval(checkHealth, 3000);
}