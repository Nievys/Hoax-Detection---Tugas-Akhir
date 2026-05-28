// ── TF-IDF Config ─────────────────────────────────────────────────
async function loadTfidfPrereq() {
  const ts = await api('/api/tfidf/status');
  let html = '';
  if (!ts.corpus_ready)
    html += mkAlert('warn','Belum ada corpus. Jalankan Batch Processing terlebih dahulu.');
  else
    html += mkAlert('success',`Corpus siap: ${ts.corpus_size} dokumen`);
  if (ts.tfidf_ready)
    html += mkAlert('info',`TF-IDF sudah dihitung: ${ts.stats.n_features} fitur, sparsity ${ts.stats.sparsity_percent}%`);
  document.getElementById('tfidf-prereq').innerHTML = html;
}

async function tfidfFit() {
  const btn = document.getElementById('tfidf-fit-btn');
  btn.disabled=true; btn.innerHTML='<span class="spinner"></span> Menghitung…';

  const mindf = parseInt(document.getElementById('tf-mindf').value)||1;
  const maxdf = parseFloat(document.getElementById('tf-maxdf').value)||1.0;
  const mf = document.getElementById('tf-maxfeat').value;
  const maxfeat = mf ? parseInt(mf) : null;
  const smooth  = document.getElementById('tf-smooth').value==='true';
  const norm    = document.getElementById('tf-norm').value==='true';

  const d = await api('/api/tfidf/fit','POST',{
    min_df:mindf, max_df_ratio:maxdf,
    max_features:maxfeat, smooth_idf:smooth, normalize:norm
  });

  btn.disabled=false; btn.innerHTML='▶ Fit & Transform';

  if (!d.success) { alert(d.error); return; }

  tfidfReady = true;
  document.getElementById('tf-n-doc').textContent    = d.stats.n_documents;
  document.getElementById('tf-n-feat').textContent   = d.stats.n_features;
  document.getElementById('tf-total-el').textContent = d.stats.total_elements.toLocaleString();
  document.getElementById('tf-sparsity').textContent = d.stats.sparsity_percent+'%';
  document.getElementById('tfidf-fit-result').classList.remove('hidden');

  // IDF bar chart
  const topIdf = d.top_idf_terms;
  maxIdf = topIdf.length > 0 ? topIdf[0].idf : 1;
  document.getElementById('top-idf-list').innerHTML = topIdf.map(x=>`
    <div class="idf-bar-item">
      <span class="idf-term">${x.term}</span>
      <div class="idf-bar-wrap"><div class="idf-bar-fill" style="width:${(x.idf/maxIdf*100).toFixed(1)}%"></div></div>
      <span class="idf-val">${x.idf.toFixed(4)}</span>
      <span class="idf-df">df=${x.df}</span>
    </div>`
  ).join('');

  // Update dashboard
  document.getElementById('d-vocab').textContent = d.stats.n_features;
  document.getElementById('d-sparse').textContent = d.stats.sparsity_percent+'%';
  document.getElementById('cnt-vocab').textContent = d.stats.n_features;
}

// ── Vocabulary ─────────────────────────────────────────────────────
let vocabSrchT;
function vocabSearch() {
  clearTimeout(vocabSrchT);
  vocabSrchT = setTimeout(()=>{ vocabPage=1; vocabLoad(); }, 300);
}

async function vocabLoad() {
  const srch  = document.getElementById('vocab-srch')?.value||'';
  const sort  = document.getElementById('vocab-sort')?.value||'alpha';
  const d = await api(`/api/tfidf/vocabulary?page=${vocabPage}&per_page=${VOCAB_PP}&search=${encodeURIComponent(srch)}&sort=${sort}`);

  const empty = document.getElementById('vocab-empty');
  const tbl   = document.getElementById('vocab-tbl');

  if (!d.success) {
    empty.classList.remove('hidden'); tbl.classList.add('hidden'); return;
  }
  empty.classList.add('hidden'); tbl.classList.remove('hidden');

  const items = d.items;
  const maxIdfLocal = items.length>0 ? Math.max(...items.map(x=>x.idf)) : 1;

  document.getElementById('vocab-tbody').innerHTML = items.map(x=>`
    <tr>
      <td class="text-mute text-mono">${x.index}</td>
      <td style="color:var(--blue); font-family:var(--mono)">${x.term}</td>
      <td class="text-mute">${x.df}</td>
      <td class="text-mono">${x.idf.toFixed(6)}</td>
      <td style="min-width:100px;">
        <div class="idf-bar-wrap" style="height:6px;">
          <div class="idf-bar-fill" style="width:${(x.idf/maxIdfLocal*100).toFixed(1)}%"></div>
        </div>
      </td>
    </tr>`
  ).join('');

  const pages = Math.ceil(d.total/VOCAB_PP);
  document.getElementById('vocab-pg-info').textContent = `Hal ${vocabPage}/${pages} — ${d.total} term`;
}

function vocabPg(dir) {
  const srch = document.getElementById('vocab-srch')?.value||'';
  const sort = document.getElementById('vocab-sort')?.value||'alpha';
  fetch(`/api/tfidf/vocabulary?page=1&per_page=${VOCAB_PP}&search=${encodeURIComponent(srch)}&sort=${sort}`)
    .then(r=>r.json()).then(d=>{
      const np = vocabPage+dir;
      if (np<1||np>Math.ceil(d.total/VOCAB_PP)) return;
      vocabPage=np; vocabLoad();
    });
}

// ── Matrix Preview ────────────────────────────────────────────────
async function loadMatrix() {
  const ts = await api('/api/tfidf/status');
  const empty   = document.getElementById('matrix-empty');
  const content = document.getElementById('matrix-content');

  if (!ts.tfidf_ready) {
    empty.classList.remove('hidden');
    content.classList.add('hidden');
    return;
  }

  empty.classList.add('hidden');
  content.classList.remove('hidden');

  const d = await api('/api/tfidf/fit','POST',{}); // re-use last result
  if (!d.success) return;

  const p = d.preview;
  document.getElementById('matrix-shape-chip').textContent =
    `${p.full_shape[0]} × ${p.full_shape[1]}${p.truncated?' (preview)':''}`;

  // Build table
  const cols = p.preview_cols;
  let html = '<table class="matrix-tbl"><thead><tr><th class="row-header">doc\\term</th>';
  cols.forEach(c => { html+=`<th title="${c}">${c.length>8?c.slice(0,8)+'…':c}</th>`; });
  html += '</tr></thead><tbody>';

  const allVals = p.preview_matrix.flat().filter(v=>v>0);
  const hiThr = allVals.length>0 ? allVals.sort((a,b)=>b-a)[Math.floor(allVals.length*.2)] : 0.5;

  p.preview_matrix.forEach((row, i) => {
    html += `<tr><th class="row-header">d${i}</th>`;
    row.forEach(v => {
      const cls = v===0 ? 'zero' : (v>=hiThr ? 'high' : '');
      html += `<td class="${cls}">${v===0?'·':v.toFixed(4)}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('matrix-wrap').innerHTML = html;
}

async function loadDocDetail() {
  const idx = parseInt(document.getElementById('doc-idx-inp').value)||0;
  const d = await api(`/api/tfidf/document/${idx}?top_n=10`);
  if (!d.success) { document.getElementById('doc-detail').innerHTML = mkAlert('error',d.error); return; }

  let html = `
    <div class="diff mt-2">
      <div class="diff-panel diff-before">
        <div class="diff-label">Teks Asli <span style="font-weight:400;">[dok ${idx}]</span></div>
        <div>${d.original_text||'(kosong)'}</div>
      </div>
      <div class="diff-panel diff-after">
        <div class="diff-label">Ternormalisasi</div>
        <div>${d.normalized_text||'(kosong)'}</div>
      </div>
    </div>
    <div class="section-sep">Top TF-IDF — Dokumen ${idx}  ·  label: <strong>${d.label}</strong>  ·  L2 norm: ${d.l2_norm}  ·  non-zero: ${d.non_zero_count}</div>
    <div style="margin-top:10px;">
  `;
  d.top_features.forEach(f => {
    html += `
      <div class="idf-bar-item">
        <span class="idf-term">${f.term}</span>
        <div class="idf-bar-wrap"><div class="idf-bar-fill" style="width:${(f.tfidf*100).toFixed(1)}%"></div></div>
        <span class="idf-val">${f.tfidf.toFixed(6)}</span>
      </div>`;
  });
  html += '</div>';
  document.getElementById('doc-detail').innerHTML = html;
}

// ── TF-IDF Single ─────────────────────────────────────────────────
function tfsEx() {
  document.getElementById('tfs-inp').value =
    'Dasar brengsek lo! gak tau diri bgt, emang ga ada otaknya ya?';
  tfsRun();
}

async function tfsRun() {
  const text = document.getElementById('tfs-inp').value.trim();
  if (!text) return;
  const d = await api('/api/tfidf/single','POST',{text});
  if (!d.success) { alert(d.error); return; }

  document.getElementById('tfs-raw').textContent  = d.raw_text;
  document.getElementById('tfs-norm').textContent = d.normalized_text;
  document.getElementById('tfs-vocab-chip').textContent =
    (d.used_existing_vocab ? `vocab ${d.vocab_size} term` : 'standalone');
  document.getElementById('tfs-result').classList.remove('hidden');

  const terms = d.top_terms||[];
  const maxTf = terms.length>0 ? Math.max(...terms.map(x=>x.tfidf)) : 1;
  document.getElementById('tfs-terms').innerHTML = terms.map(x=>`
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:6px; padding:7px 0; border-bottom:1px solid var(--border); font-size:12px;">
      <span class="text-mono" style="color:var(--blue)">${x.term}</span>
      <span class="text-mono text-mute">${x.tf.toFixed(6)}</span>
      <span class="text-mono text-mute">${x.idf.toFixed(6)}</span>
      <span>
        <div style="display:flex;align-items:center;gap:6px;">
          <div class="idf-bar-wrap" style="flex:1;height:5px;"><div class="idf-bar-fill" style="width:${(x.tfidf/maxTf*100).toFixed(1)}%"></div></div>
          <span class="text-mono" style="color:var(--purple);min-width:70px">${x.tfidf.toFixed(6)}</span>
        </div>
      </span>
    </div>`
  ).join('') || '<p class="text-sm text-mute">Tidak ada term yang cocok dengan vocabulary.</p>';
}
