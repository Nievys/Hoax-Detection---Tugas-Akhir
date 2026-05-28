// ── Dataset ───────────────────────────────────────────────────────
function uploadDs(input) {
  const file = input.files[0]; if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  fd.append('text_col', document.getElementById('ds-tc').value||'text');
  fd.append('label_col', document.getElementById('ds-lc').value||'label');
  fetch('/api/dataset/upload',{method:'POST',body:fd})
    .then(r=>r.json()).then(d=>{
      if (d.success) {
        showAlert('msg-ds','success',`${d.total} baris dimuat.`);
        document.getElementById('cnt-ds').textContent = d.total;
        document.getElementById('d-ds').textContent = d.total;
        document.getElementById('ds-preview-card').classList.remove('hidden');
        document.getElementById('ds-total-chip').textContent = d.total+' baris';
        document.getElementById('ds-tbody').innerHTML = d.sample.map((r,i)=>
          `<tr><td class="text-mute">${i+1}</td>
           <td title="${r.text}">${r.text}</td>
           <td class="${r.label=='1'||r.label?.toLowerCase()=='hate'?'label-hate':'label-ok'}">${r.label}</td></tr>`
        ).join('');
      } else {
        showAlert('msg-ds','error',d.error);
      }
    });
}
