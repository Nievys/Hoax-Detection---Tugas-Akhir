// ── Dashboard ─────────────────────────────────────────────────────
async function loadDashStats() {
  const d = await api('/api/lexicon/status');
  if (d.success) {
    document.getElementById('d-lex').textContent = d.merged_count;
    document.getElementById('cnt-lex').textContent = d.merged_count;
  }
  const ts = await api('/api/tfidf/status');
  if (ts.success) {
    if (ts.tfidf_ready) {
      document.getElementById('d-vocab').textContent = ts.stats.n_features;
      document.getElementById('d-sparse').textContent = ts.stats.sparsity_percent + '%';
      document.getElementById('cnt-vocab').textContent = ts.stats.n_features;
    }
    document.getElementById('d-ds').textContent = ts.corpus_size;
    document.getElementById('cnt-ds').textContent = ts.corpus_size;
  }
}

function qtLoad() {
  document.getElementById('qt-input').value =
    'Dasar brengsek lo! gak tau diri bgt, pergi aja dari sini! https://bit.ly/xyz #kebencian @user123 😡';
  quickTest();
}

async function quickTest() {
  const text = document.getElementById('qt-input').value.trim();
  if (!text) return;
  const d = await api('/api/preprocess/single','POST',{text,verbose:false});
  if (!d.success) return;
  const r = d.result;
  document.getElementById('qt-raw').textContent = r.raw;
  document.getElementById('qt-out').textContent = r.normalized || '(kosong)';
  document.getElementById('qt-result').classList.remove('hidden');
  const te = document.getElementById('qt-tags');
  if (r.replacements.length > 0) {
    te.innerHTML = '<div class="replace-list">' + r.replacements.map(x =>
      `<span class="replace-tag"><span class="orig">${x.original}</span><span class="arr">→</span><span class="repl">${x.replaced||'∅'}</span></span>`
    ).join('') + '</div>';
  } else {
    te.innerHTML = '<p class="text-sm text-mute mt-1">Tidak ada penggantian slang.</p>';
  }
}
