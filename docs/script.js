// ── Network registry ───────────────────────────────────────────────────────
const NETWORKS = {
  'grokking-mnist': 'networks/grokking-mnist/info.json'
};

// ── State ──────────────────────────────────────────────────────────────────
let currentWeights = null;
let isDrawing = false;
let lastX = 0, lastY = 0;

// ── DOM refs ───────────────────────────────────────────────────────────────
const canvas         = document.getElementById('draw-canvas');
const ctx            = canvas.getContext('2d');
const clearBtn       = document.getElementById('clear-btn');
const guessBtn       = document.getElementById('guess-btn');
const netSelect      = document.getElementById('network-select');
const loadStatus     = document.getElementById('load-status');
const barChart       = document.getElementById('bar-chart');
const predDisplay    = document.getElementById('prediction-display');
const notesContent   = document.getElementById('notes-content');
const notebookRender = document.getElementById('notebook-render');

// ── Canvas drawing ─────────────────────────────────────────────────────────
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 18; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#fff';

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const src  = e.touches ? e.touches[0] : e;
  return {
    x: (src.clientX - rect.left) * (canvas.width  / rect.width),
    y: (src.clientY - rect.top)  * (canvas.height / rect.height)
  };
}
canvas.addEventListener('mousedown',  e => { isDrawing = true;  const p = getPos(e); lastX = p.x; lastY = p.y; });
canvas.addEventListener('mousemove',  e => { if (!isDrawing) return; const p = getPos(e); ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(p.x, p.y); ctx.stroke(); lastX = p.x; lastY = p.y; });
canvas.addEventListener('mouseup',    () => { isDrawing = false; ctx.beginPath(); });
canvas.addEventListener('mouseleave', () => { isDrawing = false; ctx.beginPath(); });
canvas.addEventListener('touchstart', e => { e.preventDefault(); isDrawing = true; const p = getPos(e); lastX = p.x; lastY = p.y; });
canvas.addEventListener('touchmove',  e => { e.preventDefault(); if (!isDrawing) return; const p = getPos(e); ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(p.x, p.y); ctx.stroke(); lastX = p.x; lastY = p.y; });
canvas.addEventListener('touchend',   () => { isDrawing = false; ctx.beginPath(); });

clearBtn.addEventListener('click', () => {
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  predDisplay.innerHTML = '<span class="prediction-label">Draw something and press Guess</span>';
  barChart.innerHTML = '';
});

// ── Pyodide ────────────────────────────────────────────────────────────────
let pyodide       = null;
let pyodideReady  = false;
let pyodideBooting = false;
let runCounter    = 0;

const kernelDot   = document.getElementById('kernel-dot');
const kernelLabel = document.getElementById('kernel-label');

function setKernelStatus(status) {
  kernelDot.className = 'nb-kernel-dot ' + status;
  const labels = { idle: 'Python (idle)', busy: 'Python (busy)', error: 'Python (error)', loading: 'Python (loading…)' };
  kernelLabel.textContent = labels[status] || status;
}

async function bootPyodide() {
  if (pyodideReady || pyodideBooting) return;
  pyodideBooting = true;
  setKernelStatus('loading');
  try {
    pyodide = await loadPyodide();
    await pyodide.loadPackage(['numpy', 'matplotlib']);
    // Set up matplotlib to capture plt.show() as base64 PNG printed to stdout
    await pyodide.runPythonAsync(`
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, sys

def _show_capture():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    print('__IMG__' + base64.b64encode(buf.read()).decode() + '__ENDIMG__')
    plt.close('all')
plt.show = _show_capture
`);
    pyodideReady = true;
    setKernelStatus('idle');
  } catch (e) {
    setKernelStatus('error');
    console.error('Pyodide failed:', e);
  }
  pyodideBooting = false;
}

async function runCellCode(code, outputContent, promptEl) {
  if (!pyodideReady) await bootPyodide();
  if (!pyodideReady) {
    outputContent.innerHTML = '<pre class="nb-error-text">Could not start Python runtime.</pre>';
    return;
  }

  runCounter++;
  const thisRun = runCounter;
  promptEl.textContent = 'In [*]:';
  outputContent.innerHTML = '<div class="nb-running-text">Running…</div>';
  setKernelStatus('busy');

  // Redirect stdout/stderr
  pyodide.runPython(`
import sys, io as _io
_buf = _io.StringIO()
sys.stdout = _buf
sys.stderr = _buf
`);

  let rawOutput = '';
  let hasError  = false;

  try {
    await pyodide.loadPackagesFromImports(code);
    await pyodide.runPythonAsync(code);
  } catch (err) {
    hasError = true;
    rawOutput += err.message + '\n';
  }

  try {
    rawOutput = pyodide.runPython('_buf.getvalue()') + rawOutput;
    pyodide.runPython('sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__');
  } catch (_) {}

  promptEl.textContent = `In [${thisRun}]:`;
  renderCellOutput(rawOutput, outputContent, hasError);
  setKernelStatus('idle');
}

function renderCellOutput(text, container, isError) {
  container.innerHTML = '';
  if (!text.trim()) return;

  // Split on image markers embedded in stdout
  const parts = text.split(/(__IMG__[\s\S]*?__ENDIMG__)/);
  parts.forEach(part => {
    const imgMatch = part.match(/^__IMG__([\s\S]*?)__ENDIMG__$/);
    if (imgMatch) {
      const img = document.createElement('img');
      img.src = `data:image/png;base64,${imgMatch[1].trim()}`;
      img.className = 'nb-plot-img';
      container.appendChild(img);
    } else if (part.trim()) {
      const pre = document.createElement('pre');
      pre.className = isError ? 'nb-error-text' : 'nb-stdout';
      pre.textContent = part;
      container.appendChild(pre);
    }
  });
}

// ── Notebook renderer ──────────────────────────────────────────────────────
function autoResize(ta) {
  ta.style.height = 'auto';
  ta.style.height = ta.scrollHeight + 'px';
}

function renderNotebook(nb, title) {
  notebookRender.innerHTML = '';
  document.getElementById('nb-title').textContent = title || 'Notebook';

  const cells = nb.cells || [];

  cells.forEach(cell => {
    const src = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');

    if (cell.cell_type === 'markdown') {
      const div = document.createElement('div');
      div.className = 'nb-md-cell';
      div.innerHTML = marked.parse(src);
      notebookRender.appendChild(div);

    } else if (cell.cell_type === 'code') {

      const wrapper = document.createElement('div');
      wrapper.className = 'nb-code-cell';

      // ── Input ──
      const inputArea = document.createElement('div');
      inputArea.className = 'nb-input-area';

      const inPrompt = document.createElement('div');
      inPrompt.className = 'nb-in-prompt';
      inPrompt.textContent = 'In [ ]:';

      const textarea = document.createElement('textarea');
      textarea.className = 'nb-editor';
      textarea.value = src;
      textarea.spellcheck = false;
      textarea.addEventListener('input', () => autoResize(textarea));
      textarea.addEventListener('focus', () => wrapper.classList.add('selected'));
      textarea.addEventListener('blur',  () => wrapper.classList.remove('selected'));
      textarea.addEventListener('keydown', e => {
        if (e.key === 'Enter' && e.shiftKey) { e.preventDefault(); runBtn.click(); return; }
        if (e.key === 'Tab') {
          e.preventDefault();
          const s = textarea.selectionStart, en = textarea.selectionEnd;
          textarea.value = textarea.value.slice(0, s) + '    ' + textarea.value.slice(en);
          textarea.selectionStart = textarea.selectionEnd = s + 4;
        }
      });

      const runBtn = document.createElement('button');
      runBtn.className = 'nb-run-btn';
      runBtn.innerHTML = '&#9654;';
      runBtn.title = 'Run cell (Shift+Enter)';

      inputArea.appendChild(inPrompt);
      inputArea.appendChild(textarea);
      inputArea.appendChild(runBtn);

      // ── Output ──
      const outputArea = document.createElement('div');
      outputArea.className = 'nb-output-area';

      const outPrompt = document.createElement('div');
      outPrompt.className = 'nb-out-prompt';

      const outputContent = document.createElement('div');
      outputContent.className = 'nb-output-content';

      // Show saved outputs from the notebook file
      (cell.outputs || []).forEach(out => {
        if (out.output_type === 'stream') {
          const text = Array.isArray(out.text) ? out.text.join('') : out.text;
          const pre = document.createElement('pre');
          pre.className = 'nb-stdout'; pre.textContent = text;
          outputContent.appendChild(pre);
        } else if (out.output_type === 'display_data' || out.output_type === 'execute_result') {
          const data = out.data || {};
          if (data['image/png']) {
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${data['image/png']}`;
            img.className = 'nb-plot-img';
            outputContent.appendChild(img);
          } else if (data['text/plain']) {
            const text = Array.isArray(data['text/plain']) ? data['text/plain'].join('') : data['text/plain'];
            const pre = document.createElement('pre');
            pre.className = 'nb-stdout'; pre.textContent = text;
            outputContent.appendChild(pre);
          }
        }
      });

      outputArea.appendChild(outPrompt);
      outputArea.appendChild(outputContent);
      wrapper.appendChild(inputArea);
      wrapper.appendChild(outputArea);
      notebookRender.appendChild(wrapper);

      // Wire up run button
      runBtn.addEventListener('click', async () => {
        wrapper.classList.add('running');
        outPrompt.textContent = '';
        await runCellCode(textarea.value, outputContent, inPrompt);
        wrapper.classList.remove('running');
      });

      // Size textarea after render
      requestAnimationFrame(() => autoResize(textarea));
    }
  });

  // Run all button
  document.getElementById('run-all-btn').onclick = async () => {
    for (const cell of notebookRender.querySelectorAll('.nb-code-cell')) {
      const textarea = cell.querySelector('.nb-editor');
      const runBtn   = cell.querySelector('.nb-run-btn');
      if (textarea && runBtn) runBtn.click();
      // Small gap between cells so UI can update
      await new Promise(r => setTimeout(r, 50));
    }
  };

  // Restart button — clears Pyodide state and re-inits
  document.getElementById('restart-btn').onclick = async () => {
    pyodide = null; pyodideReady = false; pyodideBooting = false;
    setKernelStatus('loading');
    runCounter = 0;
    notebookRender.querySelectorAll('.nb-in-prompt').forEach(p => p.textContent = 'In [ ]:');
    notebookRender.querySelectorAll('.nb-output-content').forEach(el => el.innerHTML = '');
    await bootPyodide();
  };
}

// ── Network loading ────────────────────────────────────────────────────────
async function loadNetwork(key) {
  loadStatus.textContent = 'Loading…';
  guessBtn.disabled = true;
  currentWeights = null;
  notesContent.textContent = 'Loading notes…';
  notebookRender.innerHTML = '<p style="padding:1rem;color:#888">Loading notebook…</p>';

  try {
    const info = await (await fetch(NETWORKS[key])).json();

    document.getElementById('info-arch').textContent    = info.architecture;
    document.getElementById('info-tech').textContent    = info.techniques;
    document.getElementById('info-results').textContent = info.results;

    currentWeights = await (await fetch(info.weights)).json();

    notesContent.innerHTML = info.notes
      ? marked.parse(await (await fetch(info.notes)).text())
      : 'No notes for this network yet.';

    if (info.notebook) {
      const nb = await (await fetch(info.notebook)).json();
      renderNotebook(nb, info.name);
    } else {
      notebookRender.innerHTML = '<p style="padding:1rem;color:#888">No notebook attached.</p>';
    }

    loadStatus.textContent = 'Ready';
    guessBtn.disabled = false;

  } catch (err) {
    loadStatus.textContent = 'Failed to load';
    console.error(err);
  }
}

netSelect.addEventListener('change', () => loadNetwork(netSelect.value));
loadNetwork(netSelect.value);

// ── Network forward pass ───────────────────────────────────────────────────
function dotVecMat(vec, mat) {
  const n = vec.length, m = mat[0].length;
  const out = new Float64Array(m);
  for (let j = 0; j < m; j++) { let s = 0; for (let i = 0; i < n; i++) s += vec[i] * mat[i][j]; out[j] = s; }
  return out;
}
const tanh    = v => Array.from(v).map(x => Math.tanh(x));
const softmax = v => { const a = Array.from(v), mx = Math.max(...a), e = a.map(x => Math.exp(x-mx)), s = e.reduce((a,b)=>a+b,0); return e.map(x=>x/s); };

function predict(pixels) {
  return softmax(dotVecMat(tanh(dotVecMat(pixels, currentWeights.weights_0_1)), currentWeights.weights_1_2));
}

function getPixels() {
  const s = document.createElement('canvas'); s.width = s.height = 28;
  s.getContext('2d').drawImage(canvas, 0, 0, 28, 28);
  const d = s.getContext('2d').getImageData(0,0,28,28).data;
  const p = new Float32Array(784);
  for (let i = 0; i < 784; i++) p[i] = d[i*4] / 255;
  return p;
}

guessBtn.addEventListener('click', () => {
  if (!currentWeights) return;
  const probs = predict(getPixels());
  const pred  = probs.indexOf(Math.max(...probs));
  predDisplay.innerHTML = `<div class="prediction-number">${pred}</div><div class="prediction-conf">${(probs[pred]*100).toFixed(1)}% confidence</div>`;
  barChart.innerHTML = '';
  const top = Math.max(...probs);
  probs.forEach((p, d) => {
    const row = document.createElement('div'); row.className = 'bar-row';
    row.innerHTML = `<span class="bar-digit">${d}</span><div class="bar-track"><div class="bar-fill ${d===pred?'top':''}" style="width:${(p/top*100).toFixed(1)}%"></div></div><span class="bar-pct">${(p*100).toFixed(1)}%</span>`;
    barChart.appendChild(row);
  });
});
