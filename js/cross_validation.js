async function loadCV() {
  const st = await api('/api/dataset/info');
  
  // Populate target label dropdown
  const sel = document.getElementById('cv-target');
  if (sel && st.success && st.label_cols && st.label_cols.length > 0) {
    const keys = st.label_cols;
    sel.innerHTML = '<option value="">(Pilih Label Target)</option>';
    keys.forEach(k => {
      const opt = document.createElement('option');
      opt.value = k;
      opt.textContent = k;
      sel.appendChild(opt);
    });
    // Auto-select if there's only one
    if (keys.length === 1) {
      sel.value = keys[0];
    }
  }

  // Hide results on initial load
  document.getElementById('cv-results-container').classList.add('hidden');
  document.getElementById('cv-status').innerHTML = '';
}

function toggleWeights(show) {
  const div = document.getElementById('soft-voting-weights');
  if (div) {
    div.style.display = show ? 'block' : 'none';
  }
}

async function runCV() {
  const k = parseInt(document.getElementById('cv-k').value, 10);
  const seed = parseInt(document.getElementById('cv-seed').value, 10);
  const targetLabel = document.getElementById('cv-target').value;
  const statusEl = document.getElementById('cv-status');
  const resultsContainer = document.getElementById('cv-results-container');

  if (isNaN(k) || k < 2) {
    alert("Nilai K harus bilangan bulat >= 2.");
    return;
  }
  if (!targetLabel) {
    alert("Silakan pilih label target terlebih dahulu.");
    return;
  }

  statusEl.innerHTML = `<span style="color:var(--primary);">Executing K-Fold...</span>`;
  resultsContainer.classList.add('hidden');
  document.getElementById('btn-run-cv').disabled = true;

  // Cek metode ensemble
  const methodEls = document.getElementsByName('ensemble_method');
  let ensembleMethod = 'hard';
  for (let m of methodEls) {
    if (m.checked) ensembleMethod = m.value;
  }
  
  const weights = {
    svm: parseFloat(document.getElementById('w-svm').value) || 0.4,
    nb: parseFloat(document.getElementById('w-nb').value) || 0.32,
    rf: parseFloat(document.getElementById('w-rf').value) || 0.28
  };

  try {
    const res = await api('/api/cv/run', 'POST', {
      k: k,
      seed: isNaN(seed) ? 42 : seed,
      target_label: targetLabel,
      ensemble_method: ensembleMethod,
      weights: weights
    });

    if (res.success) {
      statusEl.innerHTML = `<span style="color:var(--green);">Selesai! Evaluasi silang pada ${res.total_samples} sampel data.</span>`;
      renderCVSummary(res.summary_table);
      renderCVConfusionMatrix(res.aggregated_confusion_matrix);
      renderCVCMSamples(res.cm_samples);
      renderCVFoldDetails(res.fold_results);
      const reportEl = document.getElementById('cv-report-text');
      if (reportEl) reportEl.textContent = res.report_text;
      resultsContainer.classList.remove('hidden');
    } else {
      statusEl.innerHTML = `<span style="color:var(--red);">Gagal: ${res.error}</span>`;
      alert(res.error);
    }
  } catch (err) {
    statusEl.innerHTML = `<span style="color:var(--red);">Terjadi kesalahan jaringan.</span>`;
    console.error(err);
  } finally {
    document.getElementById('btn-run-cv').disabled = false;
  }
}

function renderCVSummary(summaryTable) {
  const tbody = document.getElementById('cv-summary-tbody');
  tbody.innerHTML = '';

  summaryTable.forEach((row, idx) => {
    const tr = document.createElement('tr');
    const timeMs = row.execution_time_ms ? row.execution_time_ms.toFixed(1) + ' ms' : '-';
    tr.innerHTML = `
      <td>#${idx + 1}</td>
      <td style="font-weight:500; color:var(--primary);">${row.model}</td>
      <td>${(row.accuracy_mean).toFixed(4)} <span class="text-mute text-sm">±${(row.accuracy_std).toFixed(4)}</span></td>
      <td>${(row.precision_mean).toFixed(4)} <span class="text-mute text-sm">±${(row.precision_std).toFixed(4)}</span></td>
      <td>${(row.recall_mean).toFixed(4)} <span class="text-mute text-sm">±${(row.recall_std).toFixed(4)}</span></td>
      <td style="font-weight:600;">${(row.f1_score_mean).toFixed(4)} <span class="text-mute text-sm" style="font-weight:normal">±${(row.f1_score_std).toFixed(4)}</span></td>
      <td style="font-family:monospace; font-size:12px;">${timeMs}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderCVConfusionMatrix(cmData) {
  const container = document.getElementById('cv-confusion-matrices');
  if (!container) return;
  container.innerHTML = '';

  if (!cmData) return;

  const models = ['SVM', 'Naive Bayes', 'Random Forest', 'Ensemble (Voting)'];
  models.forEach(model => {
    const cm = cmData[model];
    if (!cm) return;

    const total = cm.TP + cm.TN + cm.FP + cm.FN;

    const div = document.createElement('div');
    div.style.background = 'var(--surface)';
    div.style.padding = '16px';
    div.style.borderRadius = 'var(--r)';
    
    div.innerHTML = `
      <div style="font-weight:600; margin-bottom:12px; color:var(--primary);">${model}</div>
      <div style="display:grid; grid-template-columns: 80px 1fr 1fr; gap:4px; text-align:center; font-size:12px;">
        <div></div>
        <div style="font-weight:600;">Pred Positif</div>
        <div style="font-weight:600;">Pred Negatif</div>
        
        <div style="font-weight:600; display:flex; align-items:center; justify-content:flex-end; padding-right:8px;">Aktual Positif</div>
        <div style="background:rgba(239, 248, 239, 0.89); border:1px solid var(--green); padding:8px; border-radius:4px;">
          <div style="font-weight:bold; font-size:16px;">${cm.TP}</div>
          <div class="text-mute" style="font-size:10px;">True Positive</div>
        </div>
        <div style="background:rgba(248, 239, 239, 0.89); border:1px solid var(--red); padding:8px; border-radius:4px;">
          <div style="font-weight:bold; font-size:16px;">${cm.FN}</div>
          <div class="text-mute" style="font-size:10px;">False Negative</div>
        </div>
        
        <div style="font-weight:600; display:flex; align-items:center; justify-content:flex-end; padding-right:8px;">Aktual Negatif</div>
        <div style="background:rgba(248, 239, 239, 0.89); border:1px solid var(--red); padding:8px; border-radius:4px;">
          <div style="font-weight:bold; font-size:16px;">${cm.FP}</div>
          <div class="text-mute" style="font-size:10px;">False Positive</div>
        </div>
        <div style="background:rgba(239, 248, 239, 0.89); border:1px solid var(--green); padding:8px; border-radius:4px;">
          <div style="font-weight:bold; font-size:16px;">${cm.TN}</div>
          <div class="text-mute" style="font-size:10px;">True Negative</div>
        </div>
      </div>
      <div class="text-mute mt-2 text-center" style="font-size:11px;">Total Sampel: ${total}</div>
    `;
    container.appendChild(div);
  });
}

function renderCVFoldDetails(foldResults) {
  const container = document.getElementById('cv-fold-details');
  container.innerHTML = '';

  foldResults.forEach((fold) => {
    const foldDiv = document.createElement('div');
    foldDiv.style.background = 'var(--surface)';
    foldDiv.style.padding = '12px 16px';
    foldDiv.style.borderRadius = 'var(--r)';

    let html = `
      <div style="font-weight:600; margin-bottom:8px; display:flex; justify-content:space-between;">
        <span>Fold ${fold.fold}</span>
        <span class="text-sm text-mute" style="font-weight:normal;">Train: ${fold.train_size} | Val: ${fold.val_size} | Fitur: ${fold.n_features}</span>
      </div>
      <table class="data-tbl text-sm" style="width:100%; margin:0;">
        <thead>
          <tr>
            <th style="padding:4px 8px;">Model</th>
            <th style="padding:4px 8px;">Akurasi</th>
            <th style="padding:4px 8px;">Presisi</th>
            <th style="padding:4px 8px;">Recall</th>
            <th style="padding:4px 8px;">F1-Score</th>
          </tr>
        </thead>
        <tbody>
    `;

    const modelNames = ['SVM', 'Naive Bayes', 'Random Forest', 'Ensemble (Voting)'];
    modelNames.forEach(modelName => {
      const ev = fold.evaluations[modelName];
      if (ev) {
        html += `
          <tr>
            <td style="padding:4px 8px;">${modelName}</td>
            <td style="padding:4px 8px;">${ev.accuracy.toFixed(4)}</td>
            <td style="padding:4px 8px;">${ev.precision.toFixed(4)}</td>
            <td style="padding:4px 8px;">${ev.recall.toFixed(4)}</td>
            <td style="padding:4px 8px; font-weight:500;">${ev.f1_score.toFixed(4)}</td>
          </tr>
        `;
      }
    });

    html += `</tbody></table>`;
    foldDiv.innerHTML = html;
    container.appendChild(foldDiv);
  });
}

function renderCVCMSamples(cmSamples) {
  const container = document.getElementById('cv-cm-samples');
  if (!container || !cmSamples) return;

  const models = ['Ensemble (Voting)', 'SVM', 'Naive Bayes', 'Random Forest'];

  let html = `
    <div style="display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap;">
      ${models.map((m, idx) => `
        <button class="btn ${idx === 0 ? 'btn-primary' : 'btn-secondary'} btn-sm cm-tab-btn" onclick="switchCMSampleTab('${m}', this)">${m}</button>
      `).join('')}
    </div>
    <div id="cm-sample-content"></div>
  `;
  container.innerHTML = html;

  window._cmSamplesData = cmSamples;
  switchCMSampleTab('Ensemble (Voting)', container.querySelector('.cm-tab-btn'));
}

function switchCMSampleTab(modelName, btnEl) {
  if (btnEl) {
    document.querySelectorAll('#cv-cm-samples .cm-tab-btn').forEach(b => {
      b.classList.remove('btn-primary');
      b.classList.add('btn-secondary');
    });
    btnEl.classList.remove('btn-secondary');
    btnEl.classList.add('btn-primary');
  }

  const cmSamples = window._cmSamplesData || {};
  const data = cmSamples[modelName] || {TP:[], TN:[], FP:[], FN:[]};

  const renderList = (items, borderColor, emptyMsg) => {
    if (!items || items.length === 0) {
      return `<div class="text-mute text-sm" style="font-style:italic; padding:8px 0;">${emptyMsg}</div>`;
    }
    return items.map((it, i) => `
      <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-weight:600; font-size:11px; color:var(--mute);">Sampel #${i+1}</span>
          <span class="chip" style="font-size:10px; background:rgba(255,255,255,0.1); color:var(--text);">Fold ${it.fold}</span>
        </div>
      <div style="line-height:1.4; color:var(--text); font-family:var(--font);">${it.text || '<em>(teks kosong)</em>'}</div>
    `).join('');
  };

  const content = document.getElementById('cm-sample-content');
  if (!content) return;

  content.innerHTML = `
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap:16px;">
      <div style="background:var(--surface); padding:16px; border-radius:var(--r); border:1px solid rgba(255,255,255,0.05);">
        <div style="font-weight:600; color:var(--green); margin-bottom:12px; display:flex; align-items:center; justify-content:space-between;">
          <span>Aktual Positif → Prediksi Positif (TP)</span>
          <span class="chip chip-green" style="font-size:11px;">${data.TP.length} sampel</span>
        </div>
        <div style="display:flex; flex-direction:column; gap:10px;">
          ${renderList(data.TP, 'var(--green)', 'Tidak ada sampel True Positive.')}
        </div>
      </div>
      <div style="background:var(--surface); padding:16px; border-radius:var(--r); border:1px solid rgba(255,255,255,0.05);">
        <div style="font-weight:600; color:var(--primary); margin-bottom:12px; display:flex; align-items:center; justify-content:space-between;">
          <span>Aktual Negatif → Prediksi Negatif (TN)</span>
          <span class="chip chip-blue" style="font-size:11px;">${data.TN.length} sampel</span>
        </div>
        <div style="display:flex; flex-direction:column; gap:10px;">
          ${renderList(data.TN, 'var(--primary)', 'Tidak ada sampel True Negative.')}
        </div>
      </div>
    </div>
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap:16px; margin-top:16px;">
      <div style="background:var(--surface); padding:16px; border-radius:var(--r); border:1px solid rgba(255,255,255,0.05);">
        <div style="font-weight:600; color:var(--amber); margin-bottom:12px; display:flex; align-items:center; justify-content:space-between;">
          <span>Aktual Negatif → Prediksi Positif (FP)</span>
          <span class="chip chip-amber" style="font-size:11px;">${data.FP.length} sampel</span>
        </div>
        <div style="display:flex; flex-direction:column; gap:10px;">
          ${renderList(data.FP, 'var(--amber)', 'Tidak ada sampel False Positive.')}
        </div>
      </div>
      <div style="background:var(--surface); padding:16px; border-radius:var(--r); border:1px solid rgba(255,255,255,0.05);">
        <div style="font-weight:600; color:var(--red); margin-bottom:12px; display:flex; align-items:center; justify-content:space-between;">
          <span>Aktual Positif → Prediksi Negatif (FN)</span>
          <span class="chip chip-red" style="font-size:11px;">${data.FN.length} sampel</span>
        </div>
        <div style="display:flex; flex-direction:column; gap:10px;">
          ${renderList(data.FN, 'var(--red)', 'Tidak ada sampel False Negative.')}
        </div>
      </div>
    </div>
  `;
}
