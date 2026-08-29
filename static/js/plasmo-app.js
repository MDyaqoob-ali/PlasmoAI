/**
 * PlasmoAI — Client Controller and Diagnostic Lab Interaction Engine
 */

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initMobileNav();
  initHeroScanner();
  initDiagnosticLab();
  initBatchStudio();
  initModelExplorer();
});

/* ==========================================================================
   Theme Management
   ========================================================================== */

function initTheme() {
  const toggleBtn = document.getElementById('themeToggleBtn');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const savedTheme = localStorage.getItem('plasmo_theme') || (prefersDark ? 'dark' : 'light');
  
  applyTheme(savedTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      localStorage.setItem('plasmo_theme', next);
    });
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const toggleBtn = document.getElementById('themeToggleBtn');
  if (toggleBtn) {
    toggleBtn.innerHTML = theme === 'dark' 
      ? '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>'
      : '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';
  }
}

/* ==========================================================================
   Mobile Navigation
   ========================================================================== */

function initMobileNav() {
  const toggle = document.querySelector('.mobile-nav-toggle');
  const navLinks = document.querySelector('.nav-links');
  if (toggle && navLinks) {
    toggle.addEventListener('click', () => {
      navLinks.classList.toggle('open');
    });
  }
}

/* ==========================================================================
   Hero Interactive Scanner
   ========================================================================== */

function initHeroScanner() {
  const scannerImg = document.getElementById('heroScannerImage');
  const laser = document.getElementById('heroLaserLine');
  const statusEl = document.getElementById('heroScannerStatus');
  const resultBox = document.getElementById('heroResultWidget');
  const thumbs = document.querySelectorAll('.hero-sample-thumb');

  if (!scannerImg || !laser || !resultBox) return;

  function runHeroInference(imageSrc, sampleName) {
    laser.classList.add('laser-active');
    statusEl.innerHTML = `<span class="pulse-dot" style="color: var(--color-primary);"></span> Scanning Slide [${sampleName}]...`;

    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageSrc, filename: sampleName })
    })
    .then(r => r.json())
    .then(res => {
      laser.classList.remove('laser-active');
      if (res.success) {
        const data = res.data;
        statusEl.innerHTML = `<span class="pulse-dot" style="color: var(--color-${data.is_parasitized ? 'parasitized' : 'healthy'});"></span> Diagnostic Complete (${data.latency_ms}ms)`;
        
        resultBox.className = `diagnostic-live-result result-${data.is_parasitized ? 'parasitized' : 'healthy'}`;
        resultBox.innerHTML = `
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
              <span class="badge badge-${data.is_parasitized ? 'parasitized' : 'healthy'}">
                ${data.prediction}
              </span>
              <strong style="font-size: 0.9rem;">${data.confidence}% Confidence</strong>
            </div>
            <span style="font-size: 0.8rem; font-family: var(--font-mono); color: var(--text-muted);">${data.latency_ms} ms</span>
          </div>
          <p style="font-size: 0.825rem; margin-bottom: 0.5rem;">${data.clinical_note}</p>
          <div style="display: flex; gap: 0.5rem;">
            <a href="/diagnose?sample=${sampleName}" class="btn btn-sm btn-secondary">Open in Lab Workbench →</a>
          </div>
        `;
      }
    })
    .catch(err => {
      laser.classList.remove('laser-active');
      statusEl.textContent = 'Scan error. Please retry.';
    });
  }

  thumbs.forEach(thumb => {
    thumb.addEventListener('click', () => {
      thumbs.forEach(t => t.classList.remove('active'));
      thumb.classList.add('active');
      const filename = thumb.getAttribute('data-filename');
      const src = `/holdout_images/${filename}`;
      scannerImg.src = src;
      runHeroInference(filename, filename);
    });
  });

  // Run initial scan on first thumb if available
  if (thumbs.length > 0) {
    const initialThumb = thumbs[0];
    initialThumb.classList.add('active');
    const fn = initialThumb.getAttribute('data-filename');
    scannerImg.src = `/holdout_images/${fn}`;
    runHeroInference(fn, fn);
  }
}

/* ==========================================================================
   Full Diagnostic Laboratory Workbench (/form & /diagnose)
   ========================================================================== */

function initDiagnosticLab() {
  const dropzone = document.getElementById('labDropzone');
  const fileInput = document.getElementById('labFileInput');
  const previewImg = document.getElementById('labPreviewImg');
  const heatmapImg = document.getElementById('labHeatmapImg');
  const viewContainer = document.getElementById('labViewContainer');
  const emptyState = document.getElementById('labEmptyState');
  const activeState = document.getElementById('labActiveState');
  const laser = document.getElementById('labLaser');
  const resultCard = document.getElementById('labResultCard');
  const sampleItems = document.querySelectorAll('.lab-sample-item');

  // Sliders
  const brightnessSlider = document.getElementById('brightnessSlider');
  const contrastSlider = document.getElementById('contrastSlider');
  const invertCheck = document.getElementById('invertCheck');
  const resetFiltersBtn = document.getElementById('resetFiltersBtn');

  if (!dropzone || !fileInput) return;

  // Dropzone drag-and-drop
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileSelection(files[0]);
  });

  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileSelection(e.target.files[0]);
  });

  // Clipboard Paste Support
  window.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let item of items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const blob = item.getAsFile();
        handleFileSelection(blob);
        break;
      }
    }
  });

  // Filter Adjustments
  function applyImageFilters() {
    if (!previewImg) return;
    const b = brightnessSlider ? brightnessSlider.value : 100;
    const c = contrastSlider ? contrastSlider.value : 100;
    const inv = invertCheck && invertCheck.checked ? 100 : 0;
    previewImg.style.filter = `brightness(${b}%) contrast(${c}%) invert(${inv}%)`;
  }

  if (brightnessSlider) brightnessSlider.addEventListener('input', applyImageFilters);
  if (contrastSlider) contrastSlider.addEventListener('input', applyImageFilters);
  if (invertCheck) invertCheck.addEventListener('change', applyImageFilters);
  if (resetFiltersBtn) {
    resetFiltersBtn.addEventListener('click', () => {
      if (brightnessSlider) brightnessSlider.value = 100;
      if (contrastSlider) contrastSlider.value = 100;
      if (invertCheck) invertCheck.checked = false;
      applyImageFilters();
    });
  }

  // Sample Click
  sampleItems.forEach(item => {
    item.addEventListener('click', () => {
      const filename = item.getAttribute('data-filename');
      loadAndDiagnoseSample(filename);
    });
  });

  function handleFileSelection(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const base64Data = e.target.result;
      showActiveView(base64Data, file.name);
      executeDiagnosis(base64Data, file.name);
    };
    reader.readAsDataURL(file);
  }

  function loadAndDiagnoseSample(filename) {
    const url = `/holdout_images/${filename}`;
    showActiveView(url, filename);
    executeDiagnosis(filename, filename);
  }

  function showActiveView(src, name) {
    if (emptyState) emptyState.style.display = 'none';
    if (activeState) activeState.style.display = 'block';
    if (previewImg) previewImg.src = src;
    const nameEl = document.getElementById('activeFilename');
    if (nameEl) nameEl.textContent = name;
  }

  function executeDiagnosis(imgPayload, filename) {
    if (laser) laser.classList.add('laser-active');
    
    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imgPayload, filename: filename })
    })
    .then(r => r.json())
    .then(res => {
      if (laser) laser.classList.remove('laser-active');
      if (res.success) {
        renderDiagnosisResult(res.data);
      }
    })
    .catch(err => {
      if (laser) laser.classList.remove('laser-active');
      alert('Inference error: ' + err.message);
    });
  }

  function renderDiagnosisResult(data) {
    if (!resultCard) return;
    resultCard.style.display = 'block';

    const isParasitized = data.is_parasitized;
    const badgeClass = isParasitized ? 'badge-parasitized' : 'badge-healthy';
    const fillClass = isParasitized ? 'parasitized' : 'healthy';

    // Gauge circle calculation (circumference for r=70 is 2*PI*70 = 439.82)
    const radius = 70;
    const circ = 2 * Math.PI * radius;
    const offset = circ - (data.confidence / 100 * circ);

    resultCard.innerHTML = `
      <div class="card card-elevated" style="border-top: 4px solid var(--color-${isParasitized ? 'parasitized' : 'healthy'});">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 1rem;">
          <div>
            <span class="badge ${badgeClass}" style="font-size: 0.9rem; padding: 0.4rem 0.8rem; margin-bottom: 0.5rem;">
              ${data.prediction}
            </span>
            <h3 style="font-size: 1.5rem;">${isParasitized ? 'Malaria Parasite Detected' : 'No Parasites Detected'}</h3>
            <p style="font-size: 0.875rem; color: var(--text-muted); font-family: var(--font-mono);">
              Specimen: ${data.filename} | Latency: ${data.latency_ms} ms
            </p>
          </div>
          <div>
            <a href="/result?data=${encodeURIComponent(JSON.stringify(data))}" class="btn btn-primary" id="viewFullReportBtn">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
              View Full Clinical Lab Report
            </a>
          </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 2rem; align-items: center;">
          <!-- Gauge -->
          <div style="text-align: center;">
            <div class="gauge-container">
              <svg class="gauge-svg" viewBox="0 0 160 160">
                <circle class="gauge-bg" cx="80" cy="80" r="70"></circle>
                <circle class="gauge-fill ${fillClass}" cx="80" cy="80" r="70" 
                        stroke-dasharray="${circ}" stroke-dashoffset="${offset}"></circle>
              </svg>
              <div class="gauge-center-text">
                <div class="gauge-val">${data.confidence}%</div>
                <div class="gauge-label">Confidence</div>
              </div>
            </div>
            <div style="margin-top: 1rem;">
              <span class="badge badge-${data.severity_color}">${data.severity}</span>
            </div>
          </div>

          <!-- Diagnostic Breakdown -->
          <div>
            <h4 style="font-size: 1rem; margin-bottom: 1rem;">Class Probability Distribution</h4>
            
            <div style="margin-bottom: 1rem;">
              <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.25rem;">
                <span style="font-weight: 600;">Parasitized (Plasmodium)</span>
                <span style="font-family: var(--font-mono);">${data.probabilities.Parasitized}%</span>
              </div>
              <div class="layer-bar-wrapper">
                <div class="layer-bar" style="width: ${data.probabilities.Parasitized}%; background: var(--color-parasitized);"></div>
              </div>
            </div>

            <div style="margin-bottom: 1.5rem;">
              <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.25rem;">
                <span style="font-weight: 600;">Uninfected (Normal RBC)</span>
                <span style="font-family: var(--font-mono);">${data.probabilities.Uninfected}%</span>
              </div>
              <div class="layer-bar-wrapper">
                <div class="layer-bar" style="width: ${data.probabilities.Uninfected}%; background: var(--color-healthy);"></div>
              </div>
            </div>

            <div style="background: var(--bg-surface-subtle); padding: 1rem; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
              <strong style="display: block; font-size: 0.8rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.25rem;">Clinical Evaluation Note</strong>
              <p style="font-size: 0.85rem; margin-bottom: 0;">${data.clinical_note}</p>
            </div>
          </div>
        </div>

        <!-- Attention Hotspot / Heatmap Comparison -->
        <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid var(--border-subtle);">
          <h4 style="font-size: 1rem; margin-bottom: 1rem;">Microscopy Morphology & Chromatin Inclusions</h4>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
            <div style="background: var(--bg-surface-subtle); padding: 1rem; border-radius: var(--radius-md); text-align: center;">
              <img src="${data.image_data}" style="max-height: 140px; margin: 0 auto 0.5rem; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
              <span style="font-size: 0.8rem; font-weight: 600; color: var(--text-muted);">Segmented Cell Input</span>
            </div>
            <div style="background: var(--bg-surface-subtle); padding: 1rem; border-radius: var(--radius-md); text-align: center;">
              <img src="${data.heatmap_data}" style="max-height: 140px; margin: 0 auto 0.5rem; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
              <span style="font-size: 0.8rem; font-weight: 600; color: var(--text-muted);">Parasite Chromatin Hotspot</span>
            </div>
          </div>
        </div>
      </div>
    `;

    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Preselected sample from URL query param
  const urlParams = new URLSearchParams(window.location.search);
  const sampleParam = urlParams.get('sample');
  if (sampleParam) {
    loadAndDiagnoseSample(sampleParam);
  }
}

/* ==========================================================================
   Batch Smear & Parasitemia Studio (/batch)
   ========================================================================== */

function initBatchStudio() {
  const batchDropzone = document.getElementById('batchDropzone');
  const batchFileInput = document.getElementById('batchFileInput');
  const runDemoBatchBtn = document.getElementById('runDemoBatchBtn');
  const batchSummaryCard = document.getElementById('batchSummaryCard');
  const batchTableBody = document.getElementById('batchTableBody');
  const batchFilterSelect = document.getElementById('batchFilterSelect');
  const exportCsvBtn = document.getElementById('exportCsvBtn');

  let batchDataGlobal = [];

  if (!batchDropzone || !batchFileInput) return;

  batchDropzone.addEventListener('click', () => batchFileInput.click());
  batchFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) uploadBatchFiles(e.target.files);
  });

  if (runDemoBatchBtn) {
    runDemoBatchBtn.addEventListener('click', () => {
      fetch('/api/sample-images')
        .then(r => r.json())
        .then(res => {
          if (res.success) {
            const all = [...res.data.parasitized, ...res.data.uninfected];
            const filenames = all.map(x => x.filename);
            runBatchInferenceJSON(filenames);
          }
        });
    });
  }

  function uploadBatchFiles(files) {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    if (batchDropzone) batchDropzone.innerHTML = `<div style="padding: 2rem;"><span class="pulse-dot"></span> Processing ${files.length} Microscopy Cell Patches...</div>`;

    fetch('/api/predict-batch', {
      method: 'POST',
      body: formData
    })
    .then(r => r.json())
    .then(res => {
      if (res.success) {
        renderBatchResults(res.data);
      }
    });
  }

  function runBatchInferenceJSON(sampleNames) {
    if (batchDropzone) batchDropzone.innerHTML = `<div style="padding: 2rem;"><span class="pulse-dot"></span> Analyzing Cohort (${sampleNames.length} Patches)...</div>`;
    
    fetch('/api/predict-batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ images: sampleNames })
    })
    .then(r => r.json())
    .then(res => {
      if (res.success) {
        renderBatchResults(res.data);
      }
    });
  }

  function renderBatchResults(data) {
    batchDataGlobal = data.results;
    if (batchSummaryCard) batchSummaryCard.style.display = 'block';

    // Update Summary KPIs
    document.getElementById('batchTotalCells').textContent = data.total_cells;
    document.getElementById('batchInfectedCells').textContent = data.infected_count;
    document.getElementById('batchUninfectedCells').textContent = data.uninfected_count;
    document.getElementById('batchParasitemiaRate').textContent = data.parasitemia_index_percent + '%';
    
    const riskBadge = document.getElementById('batchRiskBadge');
    if (riskBadge) {
      riskBadge.textContent = data.cohort_risk;
      riskBadge.className = `badge badge-${data.cohort_risk_color}`;
    }

    populateTable(batchDataGlobal);
  }

  function populateTable(items) {
    if (!batchTableBody) return;
    batchTableBody.innerHTML = items.map((item, idx) => `
      <tr>
        <td><strong>#${idx + 1}</strong></td>
        <td>
          <img src="${item.image_data}" style="width: 42px; height: 42px; border-radius: var(--radius-sm); object-fit: cover;">
        </td>
        <td style="font-family: var(--font-mono); font-size: 0.85rem;">${item.filename}</td>
        <td>
          <span class="badge badge-${item.is_parasitized ? 'parasitized' : 'healthy'}">
            ${item.prediction}
          </span>
        </td>
        <td>
          <strong>${item.confidence}%</strong>
        </td>
        <td>
          <span style="font-size: 0.825rem; color: var(--text-muted);">${item.severity}</span>
        </td>
      </tr>
    `).join('');
  }

  if (batchFilterSelect) {
    batchFilterSelect.addEventListener('change', (e) => {
      const val = e.target.value;
      if (val === 'all') {
        populateTable(batchDataGlobal);
      } else if (val === 'parasitized') {
        populateTable(batchDataGlobal.filter(x => x.is_parasitized));
      } else if (val === 'uninfected') {
        populateTable(batchDataGlobal.filter(x => !x.is_parasitized));
      }
    });
  }

  if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => {
      if (batchDataGlobal.length === 0) return alert('No batch data to export');
      let csv = "Index,Filename,Prediction,Confidence,Is_Parasitized,Severity\n";
      batchDataGlobal.forEach((item, idx) => {
        csv += `${idx + 1},"${item.filename}","${item.prediction}",${item.confidence},${item.is_parasitized},"${item.severity}"\n`;
      });
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `PlasmoAI_Parasitemia_Cohort_${Date.now()}.csv`;
      a.click();
    });
  }
}

/* ==========================================================================
   Model Explorer Interactive Layer Inspector (/model)
   ========================================================================== */

function initModelExplorer() {
  const layerCards = document.querySelectorAll('.model-layer-card');
  const inspectorBox = document.getElementById('layerInspectorDetails');

  if (!layerCards || !inspectorBox) return;

  layerCards.forEach(card => {
    card.addEventListener('click', () => {
      layerCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');

      const name = card.getAttribute('data-name');
      const type = card.getAttribute('data-type');
      const output = card.getAttribute('data-output');
      const params = card.getAttribute('data-params');
      const desc = card.getAttribute('data-desc');

      inspectorBox.innerHTML = `
        <div class="card card-elevated" style="border-left: 4px solid var(--color-primary);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h3 style="font-size: 1.35rem;">${name}</h3>
            <span class="badge badge-primary">${type}</span>
          </div>
          <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1rem; background: var(--bg-surface-subtle); padding: 1rem; border-radius: var(--radius-md);">
            <div>
              <strong style="display: block; font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Output Shape</strong>
              <span style="font-family: var(--font-mono); font-weight: 700;">${output}</span>
            </div>
            <div>
              <strong style="display: block; font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Trainable Parameters</strong>
              <span style="font-family: var(--font-mono); font-weight: 700;">${params}</span>
            </div>
          </div>
          <p style="font-size: 0.925rem; color: var(--text-secondary); line-height: 1.6;">${desc}</p>
        </div>
      `;
    });
  });
}
