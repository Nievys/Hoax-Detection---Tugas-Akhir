// ── Naive Bayes ───────────────────────────────────────────────────
async function loadNB() {
  const res = await api('/api/dataset/info');
  if (res.success && res.label_cols) {
    const sel = document.getElementById('nb-target');
    if (sel) {
      sel.innerHTML = res.label_cols.map(c => `<option value="${c}">${c}</option>`).join('');
    }
  }
}

async function trainNB() {
  const btn = document.querySelector('button[onclick="trainNB()"]');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Melatih...';

  const alpha = parseFloat(document.getElementById('nb-alpha').value) || 1.0;
  const targetLabel = document.getElementById('nb-target').value;

  const res = await api('/api/nb/train', 'POST', { alpha, target_label: targetLabel });

  btn.disabled = false; btn.innerHTML = 'Latih Model';

  const statusEl = document.getElementById('nb-status');
  if (res.success) {
    statusEl.innerHTML = `<span style="color:var(--green)">✓ ${res.message}</span>`;

    document.getElementById('nb-train-result').classList.remove('hidden');
    document.getElementById('nb-total').textContent = res.training_info.n_features ? 
      Object.values(res.training_info.class_distribution).reduce((a,b)=>a+b, 0) : '—';
    document.getElementById('nb-feats').textContent = res.training_info.n_features;
    document.getElementById('nb-alpha-val').textContent = res.training_info.alpha;

    const priorHtml = Object.entries(res.prior).map(([c, info]) => `
      <div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--border);">
        <span class="chip ${c==='1'?'chip-red':'chip-green'}" style="min-width:80px;text-align:center;">
          Kelas ${c}
        </span>
        <div style="flex:1;">
          <div class="idf-bar-wrap" style="height:8px;">
            <div class="idf-bar-fill" style="width:${(info.prior*100).toFixed(1)}%;background:${c==='1'?'var(--red)':'var(--green)'}"></div>
          </div>
        </div>
        <span class="text-mono text-sm" style="min-width:60px;">${(info.prior*100).toFixed(2)}%</span>
        <span class="text-mute text-sm">(${info.count} dok)</span>
        <span class="text-mono text-sm text-mute">log: ${info.log_prior}</span>
      </div>
    `).join('');
    document.getElementById('nb-prior').innerHTML = priorHtml;

    const topFeat = res.top_features;
    let featHtml = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;">';
    for (const [c, features] of Object.entries(topFeat)) {
      featHtml += `
        <div style="padding:12px;border-radius:var(--r);background:var(--surface);">
          <div class="text-sm" style="font-weight:600;color:${c==='1'?'var(--red)':'var(--green)'};">
            Kelas ${c} — Kata Paling Khas
          </div>
          <div class="mt-1">
            ${features.map(f => `
              <div class="idf-bar-item" style="padding:3px 0;">
                <span class="idf-term">${f.term}</span>
                <div class="idf-bar-wrap" style="height:5px;">
                  <div class="idf-bar-fill" style="width:${Math.min(Math.abs(f.log_prob)/10*100, 100).toFixed(1)}%;background:${c==='1'?'var(--red)':'var(--green)'}"></div>
                </div>
                <span class="idf-val">${f.prob.toExponential(3)}</span>
              </div>
            `).join('')}
          </div>
        </div>`;
    }
    featHtml += '</div>';
    document.getElementById('nb-top-features').innerHTML = featHtml;
  } else {
    statusEl.innerHTML = `<span style="color:var(--red)">✗ ${res.error}</span>`;
  }
}

async function predictNB() {
  const text = document.getElementById('nb-input').value.trim();
  if (!text) return;

  const res = await api('/api/nb/predict', 'POST', { text });

  if (res.success) {
    document.getElementById('nb-result-area').classList.remove('hidden');
    const predEl = document.getElementById('nb-prediction-text');
    predEl.textContent = res.prediction;
    predEl.style.color = res.code === 1 ? 'var(--red)' : 'var(--green)';

    const probaHtml = Object.entries(res.probabilities).map(([c, p]) =>
      `<span class="chip ${c==='1'?'chip-red':'chip-green'}" style="margin-right:8px;">
        Kelas ${c}: ${(p*100).toFixed(2)}%
      </span>`
    ).join('');
    document.getElementById('nb-prediction-proba').innerHTML = probaHtml;
    document.getElementById('nb-prediction-detail').textContent = 'Teks Ternormalisasi: ' + res.normalized;
  } else {
    alert(res.error);
  }
}
