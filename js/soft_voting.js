// ── Ensemble Soft Voting ───────────────────────────────────────────
async function loadSoftVoting() {
  const res = await api('/api/dataset/info');
  if (res.success && res.label_cols) {
    const sel = document.getElementById('soft-target');
    if (sel) {
      sel.innerHTML = res.label_cols.map(c => `<option value="${c}">${c}</option>`).join('');
    }
  }
}

async function runSoftEval() {
  const btn = document.querySelector('button[onclick="runSoftEval()"]');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Evaluasi...';

  const targetLabel = document.getElementById('soft-target').value;
  const weights = {
    svm: parseFloat(document.getElementById('w-svm-page').value) || 0.4,
    nb: parseFloat(document.getElementById('w-nb-page').value) || 0.32,
    rf: parseFloat(document.getElementById('w-rf-page').value) || 0.28
  };

  const res = await api('/api/ensemble/evaluate_soft', 'POST', { 
      target_label: targetLabel,
      weights: weights
  });

  btn.disabled = false; btn.innerHTML = '▶ Evaluasi Soft Voting';

  const statusEl = document.getElementById('soft-status');
  if (!res.success) {
    statusEl.innerHTML = `<span style="color:var(--red)">✗ ${res.error}</span>`;
    return;
  }

  statusEl.innerHTML = `<span style="color:var(--green)">✓ Evaluasi selesai (${res.total_samples} sampel)</span>`;
  document.getElementById('soft-comparison').classList.remove('hidden');

  const comp = res.comparison;
  const colors = {'SVM':'var(--blue)','Naive Bayes':'var(--purple)','Random Forest':'var(--green)','Ensemble (Soft Voting)':'var(--amber)'};

  document.getElementById('soft-tbody').innerHTML = comp.map((m, i) => {
    const isEns = m.model.includes('Ensemble');
    const bg = isEns ? 'background:rgba(245,158,11,0.08);font-weight:600;' : '';
    return `<tr style="${bg}">
      <td>${i+1}</td>
      <td><span style="color:${colors[m.model]||'var(--text-1)'}">${isEns?'🏆 ':''} ${m.model}</span></td>
      <td class="text-mono">${(m.accuracy*100).toFixed(2)}%</td>
      <td class="text-mono">${(m.precision*100).toFixed(2)}%</td>
      <td class="text-mono">${(m.recall*100).toFixed(2)}%</td>
      <td class="text-mono">${(m.f1_score*100).toFixed(2)}%</td>
      <td class="text-mono">${m.confusion_matrix.TP}</td>
      <td class="text-mono">${m.confusion_matrix.TN}</td>
      <td class="text-mono">${m.confusion_matrix.FP}</td>
      <td class="text-mono">${m.confusion_matrix.FN}</td>
    </tr>`;
  }).join('');

  document.getElementById('soft-bars').innerHTML = comp.map(m => {
    const col = colors[m.model] || 'var(--text-1)';
    const isEns = m.model.includes('Ensemble');
    return `<div style="display:flex;align-items:center;gap:12px;padding:6px 0;">
      <span style="min-width:150px;font-size:13px;font-weight:${isEns?'700':'400'};color:${col}">${m.model}</span>
      <div style="flex:1;"><div class="idf-bar-wrap" style="height:10px;"><div class="idf-bar-fill" style="width:${(m.accuracy*100).toFixed(1)}%;background:${col}"></div></div></div>
      <span class="text-mono text-sm" style="min-width:60px;">${(m.accuracy*100).toFixed(2)}%</span>
    </div>`;
  }).join('');

  document.getElementById('soft-cm-grid').innerHTML = comp.map(m => {
    const cm = m.confusion_matrix;
    const col = colors[m.model] || 'var(--text-1)';
    return `<div style="padding:14px;border-radius:var(--r);background:var(--surface);">
      <div style="font-weight:600;font-size:13px;color:${col};margin-bottom:8px;">${m.model}</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:var(--mono);text-align:center;">
        <tr><td></td><td style="padding:4px;color:var(--text-2)">Pred 1</td><td style="padding:4px;color:var(--text-2)">Pred 0</td></tr>
        <tr><td style="color:var(--text-2);padding:4px;">Actual 1</td>
          <td style="padding:8px;background:rgba(34,197,94,0.15);border-radius:4px;"><strong>${cm.TP}</strong><br><span style="font-size:10px;color:var(--green)">TP</span></td>
          <td style="padding:8px;background:rgba(239,68,68,0.1);border-radius:4px;">${cm.FN}<br><span style="font-size:10px;color:var(--red)">FN</span></td></tr>
        <tr><td style="color:var(--text-2);padding:4px;">Actual 0</td>
          <td style="padding:8px;background:rgba(239,68,68,0.1);border-radius:4px;">${cm.FP}<br><span style="font-size:10px;color:var(--red)">FP</span></td>
          <td style="padding:8px;background:rgba(34,197,94,0.15);border-radius:4px;"><strong>${cm.TN}</strong><br><span style="font-size:10px;color:var(--green)">TN</span></td></tr>
      </table>
    </div>`;
  }).join('');
  const rawEl = document.getElementById('soft-raw-output');
  if (rawEl) rawEl.textContent = JSON.stringify(res, null, 2);
}

async function predictSoft() {
  const text = document.getElementById('soft-input').value.trim();
  if (!text) return;

  const weights = {
    svm: parseFloat(document.getElementById('w-svm-page').value) || 0.4,
    nb: parseFloat(document.getElementById('w-nb-page').value) || 0.32,
    rf: parseFloat(document.getElementById('w-rf-page').value) || 0.28
  };

  const res = await api('/api/ensemble/predict_soft', 'POST', { text, weights });
  if (!res.success) { alert(res.error); return; }

  document.getElementById('soft-result').classList.remove('hidden');
  const predEl = document.getElementById('soft-pred-text');
  predEl.textContent = res.prediction;
  predEl.style.color = res.code === 1 ? 'var(--red)' : 'var(--green)';
  document.getElementById('soft-prob-detail').textContent = (res.probability * 100).toFixed(2) + '%';
  document.getElementById('soft-pred-detail').textContent = 'Teks Ternormalisasi: ' + res.normalized;
}
