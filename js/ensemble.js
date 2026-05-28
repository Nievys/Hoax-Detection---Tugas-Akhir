// ── Ensemble Majority Voting ───────────────────────────────────────────
async function loadEnsemble() {
  const res = await api('/api/dataset/info');
  if (res.success && res.label_cols) {
    const sel = document.getElementById('ens-target');
    if (sel) {
      sel.innerHTML = res.label_cols.map(c => `<option value="${c}">${c}</option>`).join('');
    }
  }
}

async function runEnsembleEval() {
  const btn = document.querySelector('button[onclick="runEnsembleEval()"]');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Evaluasi...';

  const targetLabel = document.getElementById('ens-target').value;
  const res = await api('/api/ensemble/evaluate', 'POST', { target_label: targetLabel });

  btn.disabled = false; btn.innerHTML = '▶ Evaluasi Semua Model';

  const statusEl = document.getElementById('ens-status');
  if (!res.success) {
    statusEl.innerHTML = `<span style="color:var(--red)">✗ ${res.error}</span>`;
    return;
  }

  statusEl.innerHTML = `<span style="color:var(--green)">✓ Evaluasi selesai (${res.total_samples} sampel)</span>`;
  document.getElementById('ens-comparison').classList.remove('hidden');

  const comp = res.comparison;
  const colors = {'SVM':'var(--blue)','Naive Bayes':'var(--purple)','Random Forest':'var(--green)','Ensemble (Voting)':'var(--amber)'};

  document.getElementById('ens-tbody').innerHTML = comp.map((m, i) => {
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

  document.getElementById('ens-bars').innerHTML = comp.map(m => {
    const col = colors[m.model] || 'var(--text-1)';
    const isEns = m.model.includes('Ensemble');
    return `<div style="display:flex;align-items:center;gap:12px;padding:6px 0;">
      <span style="min-width:140px;font-size:13px;font-weight:${isEns?'700':'400'};color:${col}">${m.model}</span>
      <div style="flex:1;"><div class="idf-bar-wrap" style="height:10px;"><div class="idf-bar-fill" style="width:${(m.accuracy*100).toFixed(1)}%;background:${col}"></div></div></div>
      <span class="text-mono text-sm" style="min-width:60px;">${(m.accuracy*100).toFixed(2)}%</span>
    </div>`;
  }).join('');

  document.getElementById('ens-cm-grid').innerHTML = comp.map(m => {
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
}

async function predictEnsemble() {
  const text = document.getElementById('ens-input').value.trim();
  if (!text) return;

  const res = await api('/api/ensemble/predict', 'POST', { text });
  if (!res.success) { alert(res.error); return; }

  document.getElementById('ens-result').classList.remove('hidden');
  const predEl = document.getElementById('ens-pred-text');
  predEl.textContent = res.prediction;
  predEl.style.color = res.code === 1 ? 'var(--red)' : 'var(--green)';
  document.getElementById('ens-pred-detail').textContent = 'Teks Ternormalisasi: ' + res.normalized;

  const colors = {'SVM':'var(--blue)','Naive Bayes':'var(--purple)','Random Forest':'var(--green)'};
  document.getElementById('ens-model-votes').innerHTML = Object.entries(res.votes).map(([model, vote]) =>
    `<div style="padding:8px 14px;border-radius:var(--r);background:${vote===1?'rgba(239,68,68,0.12)':'rgba(34,197,94,0.12)'};text-align:center;min-width:100px;">
      <div style="font-size:11px;color:${colors[model]||'var(--text-2)'};font-weight:600;">${model}</div>
      <div style="font-size:18px;font-weight:700;color:${vote===1?'var(--red)':'var(--green)'};">${vote}</div>
    </div>`
  ).join('');
}
