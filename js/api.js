// ── API ────────────────────────────────────────────────────────────
async function api(url, method='GET', body=null) {
  const o = {method, headers:{'Content-Type':'application/json'}};
  if (body) o.body = JSON.stringify(body);
  const r = await fetch(url, o);
  return r.json();
}

function mkAlert(type, msg) {
  const icons = {info:'ℹ',success:'✓',warn:'⚠',error:'✗'};
  return `<div class="alert alert-${type}"><span class="alert-icon">${icons[type]}</span>${msg}</div>`;
}

function showAlert(id, type, msg) {
  document.getElementById(id).innerHTML = mkAlert(type, msg);
}
