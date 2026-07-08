/* app.js — Viewer application logic (router, render, modal) */

function route(expId, partId) {
  const exp = experiments.find(e => e.id === expId);
  if (!exp) { renderHome(); return; }
  renderExp(exp, partId);
}

function renderHome() {
  const app = document.getElementById('app');
  const cards = experiments.map(e => {
    const hasSubParts = e.parts.every(p => p.id);
    const partsHtml = hasSubParts
      ? `<div class="parts">${e.parts.map(p =>
          `<span onclick="event.stopPropagation();navigate('${e.id}-${p.id}')">${p.id.toUpperCase()}</span>`
        ).join('')}</div>`
      : '';
    const link = `${e.id}${e.parts[0].id ? '-a' : ''}`;
    return `<div class="exp-card" onclick="navigate('${link}')">
      <div class="num">Experiment ${e.num}</div>
      <h3>${esc(e.title)}</h3>
      <p>${e.dataset ? 'Dataset: ' + esc(e.dataset) : ''}</p>
      ${partsHtml}
    </div>`;
  }).join('');

  app.innerHTML = `
    <div class="home-intro">
      <h2>12 Experiments</h2>
      <p>Deep Learning Techniques Lab (U23CS7L1) &mdash; TensorFlow / Keras</p>
    </div>
    <div class="home-grid">${cards}</div>
  `;
}

function renderExp(exp, activePartId) {
  const activePart = activePartId
    ? exp.parts.find(p => p.id === activePartId)
    : exp.parts[0];
  if (!activePart) return;

  const app = document.getElementById('app');
  const tabs = exp.parts.map(p => {
    const pId = `${exp.id}${p.id ? '-' + p.id : ''}`;
    const label = p.id ? p.id.toUpperCase() : p.name;
    return `<div class="tab ${p.id === activePart.id ? 'active' : ''}" onclick="navigate('${pId}')">${esc(label)}</div>`;
  }).join('');

  const images = activePart.images || (activePart.image ? [activePart.image] : []);
  const imgs = images.map(img =>
    `<img src="images/${img}" alt="${esc(img)}" onclick="openModal('images/${img}')">`
  ).join('');

  app.innerHTML = `
    <div class="breadcrumb">
      <a href="#" onclick="event.preventDefault();navigate('')">Home</a>
      <span class="sep">/</span>
      <span>Experiment ${exp.num} &mdash; ${esc(exp.title)}</span>
    </div>

    <div class="exp-header">
      <h2>${esc(exp.title)}</h2>
      <div class="exp-meta">Experiment ${exp.num}${exp.dataset ? ' &middot; ' + esc(exp.dataset) : ''}</div>
    </div>

    <div class="tabs">${tabs}</div>

    <div class="content-split">
      <div>
        <div class="info-panel">
          <h3>Aim</h3>
          <p>${esc(exp.aim)}</p>
          <div class="section">
            <h3>Algorithm</h3>
            <ol>${exp.algorithm.map(s => `<li>${esc(s)}</li>`).join('')}</ol>
          </div>
          <div class="section">
            <h3>Result</h3>
            <p>${esc(activePart.result)}</p>
          </div>
        </div>
      </div>
      <div>
        <div class="code-panel">
          <div class="code-header">
            <span>${esc(activePart.file)}</span>
            <button class="copy-btn" onclick="copyCode(this)" title="Copy code">Copy</button>
          </div>
          <pre><code class="language-python">${esc(activePart.code)}</code></pre>
        </div>
        ${imgs ? `
        <div class="output-panel">
          <h3>Output</h3>
          <div class="output-grid">${imgs}</div>
        </div>` : ''}
      </div>
    </div>
  `;

  document.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
}

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function navigate(hash) {
  if (!hash || hash === '') {
    window.location.hash = '';
    renderHome();
    return;
  }
  window.location.hash = hash;
  const m = hash.match(/^(exp\d+)-?([a-z]?)$/);
  if (m) route(m[1], m[2] || null);
}

function copyCode(btn) {
  const code = btn.closest('.code-panel').querySelector('code');
  navigator.clipboard.writeText(code.textContent).then(() => {
    btn.textContent = 'Copied';
    setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
  });
}

function openModal(src) {
  document.getElementById('modalImg').src = src;
  document.getElementById('modal').classList.add('open');
}

function closeModal() {
  document.getElementById('modal').classList.remove('open');
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
document.getElementById('modal').addEventListener('click', closeModal);

window.addEventListener('hashchange', () => navigate(location.hash.replace('#','')));
window.addEventListener('load', () => {
  const h = location.hash.replace('#','');
  h ? navigate(h) : renderHome();
});
