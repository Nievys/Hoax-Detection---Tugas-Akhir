// ── Random Forest ───────────────────────────────────────────────────
async function loadRF() {
  const res = await api('/api/dataset/info');
  if (res.success && res.label_cols) {
    const sel = document.getElementById('rf-target');
    if (sel) {
      sel.innerHTML = res.label_cols.map(c => `<option value="${c}">${c}</option>`).join('');
    }
  }
}

async function trainRF() {
  const btn = document.querySelector('button[onclick="trainRF()"]');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Melatih...';

  const n_trees = parseInt(document.getElementById('rf-ntrees').value) || 10;
  const max_depth = parseInt(document.getElementById('rf-depth').value) || 10;
  const min_samples = parseInt(document.getElementById('rf-minsamp').value) || 2;
  const targetLabel = document.getElementById('rf-target').value;

  const res = await api('/api/rf/train', 'POST', {
    n_trees, max_depth, min_samples, target_label: targetLabel
  });

  btn.disabled = false; btn.innerHTML = 'Latih Model';

  const statusEl = document.getElementById('rf-status');
  if (res.success) {
    statusEl.innerHTML = `<span style="color:var(--green)">✓ ${res.message}</span>`;

    document.getElementById('rf-train-result').classList.remove('hidden');
    document.getElementById('rf-info-trees').textContent = res.training_info.n_trees;
    document.getElementById('rf-info-depth').textContent = res.training_info.max_depth;
    document.getElementById('rf-info-feats').textContent = res.training_info.max_features;
    document.getElementById('rf-info-samples').textContent = res.total_samples;

    const dist = res.class_distribution;
    const total = Object.values(dist).reduce((a,b)=>a+b,0);
    document.getElementById('rf-class-dist').innerHTML = Object.entries(dist).map(([c, n]) => `
      <div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--border);">
        <span class="chip ${c==='1'?'chip-red':'chip-green'}" style="min-width:80px;text-align:center;">Kelas ${c}</span>
        <div style="flex:1;"><div class="idf-bar-wrap" style="height:8px;"><div class="idf-bar-fill" style="width:${(n/total*100).toFixed(1)}%;background:${c==='1'?'var(--red)':'var(--green)'}"></div></div></div>
        <span class="text-mono text-sm">${(n/total*100).toFixed(1)}%</span>
        <span class="text-mute text-sm">(${n} dok)</span>
      </div>
    `).join('');
  } else {
    statusEl.innerHTML = `<span style="color:var(--red)">✗ ${res.error}</span>`;
  }
}

async function predictRF() {
  const text = document.getElementById('rf-input').value.trim();
  if (!text) return;

  const res = await api('/api/rf/predict', 'POST', { text });

  if (res.success) {
    document.getElementById('rf-result-area').classList.remove('hidden');
    const predEl = document.getElementById('rf-prediction-text');
    predEl.textContent = res.prediction;
    predEl.style.color = res.code === 1 ? 'var(--red)' : 'var(--green)';

    const probaHtml = Object.entries(res.probabilities).map(([c, p]) =>
      `<span class="chip ${c==='1'?'chip-red':'chip-green'}" style="margin-right:8px;">Kelas ${c}: ${(p*100).toFixed(1)}%</span>`
    ).join('');
    document.getElementById('rf-prediction-proba').innerHTML = probaHtml;
    document.getElementById('rf-prediction-detail').textContent = 'Teks Ternormalisasi: ' + res.normalized;

    document.getElementById('rf-vote-detail').innerHTML = res.vote_detail.map(v =>
      `<div style="padding:4px 8px;border-radius:6px;font-size:11px;font-family:var(--mono);background:${v.prediction===1?'rgba(239,68,68,0.15)':'rgba(34,197,94,0.15)'};color:${v.prediction===1?'var(--red)':'var(--green)'};">T${v.tree_id}:${v.prediction}</div>`
    ).join('');
  } else {
    alert(res.error);
  }
}
