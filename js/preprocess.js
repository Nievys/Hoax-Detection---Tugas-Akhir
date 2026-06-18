// ── Preprocessing Single ──────────────────────────────────────────
const PP_EX = [
  'Dasar brengsek lo! gak tau diri bgt, pergi aja dari sini! https://bit.ly/abc #kebencian @user123 456',
  'Wkwk lucu bgt deh, tp gw gak setuju sama pendapat lo yg itu, gimana menurut org lain?',
  'Lo emang ga punya otak! <b>Blm</b> bisa mikir waras ya? https://spam.com @bot 99',
];
function ppEx(n) { document.getElementById('pp-inp').value=PP_EX[n-1]; ppRun(); }

async function ppRun() {
  const text = document.getElementById('pp-inp').value.trim();
  if (!text) return;
  const use_stopword = document.getElementById('pp-stopword')?.checked || false;
  const use_stemming = document.getElementById('pp-stemming')?.checked || false;
  const d = await api('/api/preprocess/single','POST',{text,verbose:true, use_stopword, use_stemming});
  if (!d.success) return;
  const r = d.result;
  document.getElementById('pp-raw').textContent  = r.raw;
  document.getElementById('pp-norm').textContent = r.normalized||'(kosong)';
  document.getElementById('pp-results').classList.remove('hidden');
  document.getElementById('pp-steps').innerHTML = (r.cleansing_steps||[]).map(([name,val],i)=>
    `<div class="step-item">
      <div class="step-num">${i+1}</div>
      <div class="step-body">
        <div class="step-name">${name}</div>
        <div class="step-val">${val||'<em class="text-mute">(kosong)</em>'}</div>
      </div>
    </div>`
  ).join('');
  const rc = r.replacements||[];
  document.getElementById('pp-rep-chip').textContent = rc.length+' penggantian';
  document.getElementById('pp-reps').innerHTML = rc.length>0
    ? '<div class="replace-list">'+rc.map(x=>`<span class="replace-tag"><span class="orig">${x.original}</span><span class="arr">→</span><span class="repl">${x.replaced||'∅'}</span></span>`).join('')+'</div>'
    : '<p class="text-sm text-mute" style="padding:8px 0;">Tidak ada penggantian.</p>';
  document.getElementById('pp-s-raw').textContent = r.stats.raw_length;
  document.getElementById('pp-s-cln').textContent = r.stats.cleansed_length;
  document.getElementById('pp-s-tok').textContent = r.stats.tokens_raw;
  document.getElementById('pp-s-rep').textContent = r.stats.replacements_made;
}

// ── Batch ─────────────────────────────────────────────────────────
async function checkBatchPre() {
  const ls = await api('/api/lexicon/status');
  const ts = await api('/api/tfidf/status');
  let html = '';
  if (ts.corpus_size===0)
    html += mkAlert('warn','Belum ada dataset. Upload dataset terlebih dahulu.');
  else
    html += mkAlert('success',`Dataset siap: ${ts.corpus_size} teks`);
  html += mkAlert('info',`Kamus gabungan: ${ls.merged_count} entri`);
  document.getElementById('batch-prereq').innerHTML = html;
}

async function runBatch() {
  const btn = document.getElementById('batch-btn');
  btn.disabled=true; btn.innerHTML='<span class="spinner"></span> Memproses…';
  const use_stopword = document.getElementById('b-stopword')?.checked || false;
  const use_stemming = document.getElementById('b-stemming')?.checked || false;
  const d = await api('/api/preprocess/batch','POST', {use_stopword, use_stemming});
  btn.disabled=false; btn.innerHTML='▶ Jalankan Batch Processing';
  if (!d.success) { alert(d.error); return; }
  document.getElementById('batch-results').classList.remove('hidden');
  document.getElementById('b-total').textContent = d.aggregate_stats.total_texts;
  document.getElementById('b-rep').textContent   = d.aggregate_stats.total_replacements;
  document.getElementById('b-avg').textContent   = d.aggregate_stats.avg_replacements;
  document.getElementById('batch-tbody').innerHTML = d.results.map((r,i)=>
    `<tr>
      <td class="text-mute">${i+1}</td>
      <td class="${r.label=='1'||r.label?.toLowerCase()=='hate'?'label-hate':'label-ok'}">${r.label}</td>
      <td title="${r.raw}">${r.raw}</td>
      <td title="${r.normalized}">${r.normalized}</td>
      <td style="color:var(--purple)">${r.stats.replacements_made}</td>
    </tr>`
  ).join('');
  loadDashStats();
}
