// ── Drag & Drop ───────────────────────────────────────────────────
function dov(e, id) { e.preventDefault(); document.getElementById(id).classList.add('drag'); }
function dlv(id)    { document.getElementById(id).classList.remove('drag'); }

function ddf(e, type) {
  e.preventDefault(); dlv(`uz-${type==='internal'?'int':'ext'}`);
  const file = e.dataTransfer.files[0]; if (!file) return;
  const id   = type==='internal'?'fi-int':'fi-ext';
  const inp  = document.getElementById(id);
  const dt   = new DataTransfer(); dt.items.add(file); inp.files = dt.files;
  uploadLex(inp, type);
}

function dds(e) {
  e.preventDefault(); dlv('uz-ds');
  const file = e.dataTransfer.files[0]; if (!file) return;
  const inp = document.getElementById('fi-ds');
  const dt  = new DataTransfer(); dt.items.add(file); inp.files = dt.files;
  uploadDs(inp);
}
