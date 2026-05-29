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

  statusEl.innerHTML = `<span style="color:var(--primary);">⏳ Menjalankan ${k}-Fold Cross Validation... Ini mungkin memakan waktu beberapa saat.</span>`;
  resultsContainer.classList.add('hidden');
  document.getElementById('btn-run-cv').disabled = true;

  try {
    const res = await api('/api/cv/run', 'POST', {
      k: k,
      seed: isNaN(seed) ? 42 : seed,
      target_label: targetLabel
    });

    if (res.success) {
      statusEl.innerHTML = `<span style="color:var(--green);">✅ Selesai! Evaluasi silang pada ${res.total_samples} sampel data.</span>`;
      renderCVSummary(res.summary_table);
      renderCVFoldDetails(res.fold_results);
      document.getElementById('cv-report-text').textContent = res.report_text;
      resultsContainer.classList.remove('hidden');
    } else {
      statusEl.innerHTML = `<span style="color:var(--red);">❌ Gagal: ${res.error}</span>`;
      alert(res.error);
    }
  } catch (err) {
    statusEl.innerHTML = `<span style="color:var(--red);">❌ Terjadi kesalahan jaringan.</span>`;
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
    tr.innerHTML = `
      <td>#${idx + 1}</td>
      <td style="font-weight:500; color:var(--primary);">${row.model}</td>
      <td>${(row.accuracy_mean).toFixed(4)} <span class="text-mute text-sm">±${(row.accuracy_std).toFixed(4)}</span></td>
      <td>${(row.precision_mean).toFixed(4)} <span class="text-mute text-sm">±${(row.precision_std).toFixed(4)}</span></td>
      <td>${(row.recall_mean).toFixed(4)} <span class="text-mute text-sm">±${(row.recall_std).toFixed(4)}</span></td>
      <td style="font-weight:600;">${(row.f1_score_mean).toFixed(4)} <span class="text-mute text-sm" style="font-weight:normal">±${(row.f1_score_std).toFixed(4)}</span></td>
    `;
    tbody.appendChild(tr);
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
