// ── SVM ───────────────────────────────────────────────────────────
async function loadSVM() {
  const res = await api('/api/dataset/info');
  if (res.success && res.label_cols) {
    const sel = document.getElementById('svm-target');
    if (sel) {
      sel.innerHTML = res.label_cols.map(c => `<option value="${c}">${c}</option>`).join('');
    }
  }
}

async function trainSVM() {
  const btn = document.querySelector('button[onclick="trainSVM()"]');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Melatih...';
  
  const kernel = document.getElementById('svm-kernel').value;
  const cVal = document.getElementById('svm-c').value;
  const targetLabel = document.getElementById('svm-target').value;
  
  const res = await api('/api/svm/train', 'POST', { kernel: kernel, C: cVal, target_label: targetLabel });
  
  btn.disabled = false; btn.innerHTML = 'Latih Model';
  
  const statusEl = document.getElementById('svm-status');
  if (res.success) {
    statusEl.innerHTML = `<span style="color:var(--green)">✓ ${res.message}</span> (Support Vectors: ${res.support_vectors})`;
    const resContainer = document.getElementById('svm-train-result');
    const rawEl = document.getElementById('svm-raw-output');
    if (resContainer) resContainer.classList.remove('hidden');
    if (rawEl) rawEl.textContent = JSON.stringify(res, null, 2);
  } else {
    statusEl.innerHTML = `<span style="color:var(--red)">✗ ${res.error}</span>`;
  }
}

async function predictSVM() {
  const text = document.getElementById('input-deteksi').value.trim();
  if (!text) return;
  
  const res = await api('/api/svm/predict', 'POST', { text: text });
  
  if (res.success) {
    document.getElementById('result-area').classList.remove('hidden');
    const predEl = document.getElementById('prediction-text');
    predEl.textContent = res.prediction;
    predEl.style.color = res.code === 1 ? 'var(--red)' : 'var(--green)';
    document.getElementById('prediction-detail').textContent = 'Teks Ternormalisasi: ' + res.normalized;
  } else {
    alert(res.error);
  }
}
