// ── Navigation ────────────────────────────────────────────────────
async function loadHTML(id) {
  const el = document.getElementById('page-' + id);
  if (el && !el.innerHTML.trim()) {
    try {
      const res = await fetch(`/Pages/${id}/${id}.html`);
      if (res.ok) {
        el.innerHTML = await res.text();
      }
    } catch (e) {
      console.error('Failed to load page:', id, e);
    }
  }
}

async function go(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  
  if (id !== 'classification') {
    await loadHTML(id);
  }
  
  const pageEl = document.getElementById('page-'+id);
  if (pageEl) pageEl.classList.add('active');

  document.querySelectorAll('.nav-item').forEach(n => {
    if (n.getAttribute('onclick')?.includes(`'${id}'`)) n.classList.add('active');
  });

  if (id==='lexicon')       { loadLexStatus(); loadLexEntries(); }
  if (id==='dashboard')     loadDashStats();
  if (id==='batch')         checkBatchPre();
  if (id==='tfidf-config')  loadTfidfPrereq();
  if (id==='tfidf-vocab')   vocabLoad();
  if (id==='tfidf-matrix')  loadMatrix();
  if (id==='svm')           loadSVM();
  if (id==='naive-bayes')   loadNB();
  if (id==='random-forest') loadRF();
  if (id==='ensemble')      loadEnsemble();
  if (id==='soft_voting')   loadSoftVoting();
  if (id==='cross_validation') loadCV();
}
