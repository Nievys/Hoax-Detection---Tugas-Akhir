// ── Lexicon ───────────────────────────────────────────────────────
async function loadLexStatus() {
  const d = await api('/api/lexicon/status');
  if (!d.success) return;
  
  const setTxt = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  setTxt('ms-int', d.internal_count);
  setTxt('ms-ext', d.external_count);
  setTxt('ms-mrg', d.merged_count);
  setTxt('ms-con', d.conflict_count);
  setTxt('cnt-lex', d.merged_count);

  const conflicts = d.conflicts || {};
  const keys = Object.keys(conflicts);
  setTxt('conflict-chip', keys.length);
  const cc = document.getElementById('conflict-card');
  if (cc) {
    if (keys.length > 0) {
      cc.classList.remove('hidden');
      const cl = document.getElementById('conflict-list');
      if (cl) {
        cl.innerHTML = keys.slice(0,20).map(w => {
          const c = conflicts[w];
          return `<div class="conflict-row">
            <span class="word">${w}</span>
            <span class="int">${c.internal}</span>
            <span class="ext">${c.external}</span>
            <span class="win"><span class="chip chip-green" style="font-size:10px;">→ ${c.resolved}</span></span>
          </div>`;
        }).join('');
      }
    } else {
      cc.classList.add('hidden');
    }
  }
}

function uploadLex(input, type) {
  const file = input.files[0]; if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  fd.append('type', type);
  const stratEl = document.getElementById('merge-strat');
  fd.append('strategy', stratEl ? stratEl.value : 'internal_priority');
  fetch('/api/lexicon/upload', {method:'POST', body:fd})
    .then(r=>r.json()).then(d=>{
      if (d.success) {
        showAlert(`msg-${type==='internal'?'int':'ext'}`, 'success',
          `${d.count} entri dimuat. Gabungan: ${d.merged_count} entri.`);
        loadLexStatus(); loadLexEntries(); loadDashStats();
      } else {
        showAlert(`msg-${type==='internal'?'int':'ext'}`, 'error', d.error);
      }
    });
}

async function remerge() {
  const stratEl = document.getElementById('merge-strat');
  const s = stratEl ? stratEl.value : 'internal_priority';
  const d = await api('/api/merge','POST',{strategy:s});
  if (d.success) { loadLexStatus(); loadLexEntries(); }
}

let lexSrchT;
function lexSearch() {
  clearTimeout(lexSrchT);
  lexSrchT = setTimeout(()=>{ lexPage=1; loadLexEntries(); }, 300);
}

async function loadLexEntries() {
  const srch = document.getElementById('lex-srch')?.value || '';
  const d = await api(`/api/lexicon/entries?page=${lexPage}&per_page=${LEX_PP}&search=${encodeURIComponent(srch)}`);
  if (!d.success) return;
  const st = (lexPage-1)*LEX_PP+1;
  document.getElementById('lex-tbody').innerHTML = d.entries.map((e,i) =>
    `<tr><td class="text-mute">${st+i}</td>
     <td><span class="text-mono" style="color:var(--amber)">${e.slang}</span></td>
     <td style="color:var(--green)">${e.formal||'<em class="text-mute">dihapus</em>'}</td></tr>`
  ).join('');
  const pages = Math.ceil(d.total/LEX_PP);
  document.getElementById('lex-pg-info').textContent = `Hal ${lexPage}/${pages} — ${d.total} entri`;
}

function lexPg(dir) {
  const srch = document.getElementById('lex-srch')?.value||'';
  fetch(`/api/lexicon/entries?page=1&per_page=${LEX_PP}&search=${encodeURIComponent(srch)}`)
    .then(r=>r.json()).then(d=>{
      const np = lexPage+dir;
      if (np<1||np>Math.ceil(d.total/LEX_PP)) return;
      lexPage=np; loadLexEntries();
    });
}
