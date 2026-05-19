// AXON v3 — main.js
// 3D force graph with OrbitControls, parameter sliders, inspector, breathing decay.
'use strict';

// ── DOM refs ────────────────────────────────────────────────────────────────
const promptInput  = document.getElementById('prompt-input');
const startBtn     = document.getElementById('start-btn');
const stopBtn      = document.getElementById('stop-btn');
const clearBtn     = document.getElementById('clear-btn');
const fitBtn       = document.getElementById('fit-btn');
const promptTextEl = document.getElementById('prompt-text');
const generatedEl  = document.getElementById('generated-text');
const cursorEl     = document.getElementById('cursor');
const conceptsList = document.getElementById('active-concepts');
const conceptsEmpty= document.getElementById('concepts-empty');
const tokenBadge   = document.getElementById('token-badge');
const speedBadge   = document.getElementById('speed-badge');
const statNodes    = document.getElementById('stat-nodes');
const statEdges    = document.getElementById('stat-edges');
const statusDot    = document.getElementById('status-dot');
const sidebar      = document.getElementById('sidebar');
const collapseBtn  = document.getElementById('collapse-btn');
const expandBtn    = document.getElementById('expand-btn');
const inspector    = document.getElementById('inspector');
const inspContent  = document.getElementById('inspector-content');
const closeInsp    = document.getElementById('close-inspector');
const exportBtn    = document.getElementById('export-btn');
const paramsToggle = document.getElementById('params-toggle');
const paramsBody   = document.getElementById('params-body');

// ── Slider setup ────────────────────────────────────────────────────────────
const TRACK_BG  = 'rgba(26,39,68,1)';
const TRACK_FG  = '#4f8ef7';

const sliders = {
  temp:   { el: document.getElementById('p-temp'),   val: document.getElementById('v-temp'),   fmt: v => (+v).toFixed(2) },
  topk:   { el: document.getElementById('p-topk'),   val: document.getElementById('v-topk'),   fmt: v => String(v) },
  topp:   { el: document.getElementById('p-topp'),   val: document.getElementById('v-topp'),   fmt: v => (+v).toFixed(2) },
  rep:    { el: document.getElementById('p-rep'),    val: document.getElementById('v-rep'),    fmt: v => (+v).toFixed(2) },
  maxtok: { el: document.getElementById('p-maxtok'), val: document.getElementById('v-maxtok'), fmt: v => String(v) },
  featk:  { el: document.getElementById('p-featk'),  val: document.getElementById('v-featk'),  fmt: v => String(v) },
  thresh: { el: document.getElementById('p-thresh'), val: document.getElementById('v-thresh'), fmt: v => (+v).toFixed(2) },
};

function updateSlider(key) {
  const s   = sliders[key];
  const pct = ((+s.el.value - +s.el.min) / (+s.el.max - +s.el.min)) * 100;
  s.el.style.background =
    `linear-gradient(to right, ${TRACK_FG} ${pct}%, ${TRACK_BG} ${pct}%)`;
  s.val.textContent = s.fmt(s.el.value);
}
Object.keys(sliders).forEach(k => {
  updateSlider(k);
  sliders[k].el.addEventListener('input', () => updateSlider(k));
});

// ── Collapsible params ───────────────────────────────────────────────────────
let paramsOpen = true;
const chevron = paramsToggle.querySelector('.chevron');

paramsToggle.addEventListener('click', () => {
  paramsOpen = !paramsOpen;
  paramsBody.style.display = paramsOpen ? '' : 'none';
  chevron.classList.toggle('open', paramsOpen);
});

function getParams() {
  return {
    temperature:        +sliders.temp.el.value,
    top_k:              +sliders.topk.el.value,
    top_p:              +sliders.topp.el.value,
    repetition_penalty: +sliders.rep.el.value,
    max_tokens:         +sliders.maxtok.el.value,
    feat_k:             +sliders.featk.el.value,
    act_threshold:      +sliders.thresh.el.value,
  };
}

// ── Sidebar collapse ─────────────────────────────────────────────────────────
collapseBtn.addEventListener('click', () => {
  sidebar.classList.add('collapsed');
  expandBtn.style.display = 'flex';
});
expandBtn.addEventListener('click', () => {
  sidebar.classList.remove('collapsed');
  expandBtn.style.display = 'none';
});

// ── Graph state ──────────────────────────────────────────────────────────────
const graphData = { nodes: [], links: [] };
const nodeMap   = new Map();  // id → node
const linkMap   = new Map();  // "a|b" → link
let maxActSeen  = 1;
let totalTokens = 0;
let genStart    = 0;

// ── Node visuals ─────────────────────────────────────────────────────────────
// Glow sprite texture (radial gradient on a canvas)
function makeGlowTex(size = 128) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const cx = c.getContext('2d');
  const g = cx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  g.addColorStop(0.00, 'rgba(255,255,255,1)');
  g.addColorStop(0.15, 'rgba(180,210,255,0.9)');
  g.addColorStop(0.45, 'rgba(79,142,247,0.3)');
  g.addColorStop(1.00, 'rgba(0,0,0,0)');
  cx.fillStyle = g;
  cx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(c);
}
const glowTex = makeGlowTex();

function nodeRadius(act) {
  return 0.6 + 1.5 * Math.log1p(Math.max(0, act));
}

// Color: cool blue at low activation → bright cyan-white at peak
function nodeColor(act) {
  const t = Math.min(1, act / Math.max(1e-6, maxActSeen));
  // lerp from #4f8ef7 (blue) → #7ff5de (teal-white) as activation rises
  const r = Math.round(79  + (127 - 79)  * t);
  const g = Math.round(142 + (245 - 142) * t);
  const b = Math.round(247 + (222 - 247) * t);
  return `rgb(${r},${g},${b})`;
}

function buildNodeObj(node) {
  const grp = new THREE.Group();
  const r   = nodeRadius(node.activation);
  const col = nodeColor(node.activation);

  // Core sphere
  grp.add(new THREE.Mesh(
    new THREE.SphereGeometry(r, 18, 18),
    new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.92 })
  ));

  // Inner glow halo
  const s1 = new THREE.Sprite(new THREE.SpriteMaterial({
    map: glowTex,
    color: 0x4f8ef7,
    transparent: true,
    opacity: 0.75,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  }));
  s1.scale.setScalar(r * 7);
  grp.add(s1);

  // Outer wide glow (only for highly active nodes)
  const t = Math.min(1, node.activation / Math.max(1e-6, maxActSeen));
  if (t > 0.3) {
    const s2 = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex,
      color: 0x34d399,
      transparent: true,
      opacity: 0.18 * t,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }));
    s2.scale.setScalar(r * 22);
    grp.add(s2);
  }

  return grp;
}

// ── Force graph init ─────────────────────────────────────────────────────────
const Graph = ForceGraph3D()(document.getElementById('graph-container'))
  .graphData(graphData)
  .backgroundColor('#07090e')
  .showNavInfo(false)
  .nodeLabel(n => {
    const label = n.label || `Feature ${n.id}`;
    return `
      <div style="
        background:#0d1526;
        border:1px solid #1a2744;
        border-radius:7px;
        padding:6px 10px;
        font:13px/1.5 Inter,sans-serif;
        color:#e2e8f0;
        max-width:220px;
        word-wrap:break-word;
        box-shadow:0 4px 16px rgba(0,0,0,.6);
      ">${label}</div>`;
  })
  .nodeRelSize(1)
  .nodeThreeObject(buildNodeObj)
  .linkOpacity(0.45)
  .linkColor(() => 'rgba(79,142,247,0.35)')
  .linkWidth(l => Math.min(2.2, 0.3 + Math.log1p(l.value || 1) * 0.55));

// Force tuning
Graph.d3Force('charge').strength(-120);
Graph.d3Force('link').distance(55);

// ── OrbitControls ────────────────────────────────────────────────────────────
const controls = Graph.controls();
controls.autoRotate      = true;
controls.autoRotateSpeed = 0.5;
controls.enableDamping   = true;
controls.dampingFactor   = 0.1;
controls.enablePan       = true;
controls.zoomSpeed       = 1.1;

// ── Breathing decay loop ─────────────────────────────────────────────────────
const DECAY = 0.965;
let lastTick = 0;
function decayLoop(now) {
  if (now - lastTick > 160) {
    let any = false;
    for (const n of graphData.nodes) {
      if (n.activation > 0.05) { n.activation *= DECAY; any = true; }
    }
    if (any) Graph.nodeThreeObject(buildNodeObj);
    lastTick = now;
  }
  requestAnimationFrame(decayLoop);
}
requestAnimationFrame(decayLoop);

// ── Fit / Clear ──────────────────────────────────────────────────────────────
fitBtn.addEventListener('click', () => Graph.zoomToFit(700, 60));
clearBtn.addEventListener('click', resetGraph);

function resetGraph() {
  graphData.nodes.length = 0;
  graphData.links.length = 0;
  nodeMap.clear();
  linkMap.clear();
  maxActSeen  = 1;
  totalTokens = 0;
  Graph.graphData({ nodes: [], links: [] });
  conceptsList.innerHTML = '';
  if (conceptsEmpty) conceptsList.appendChild(conceptsEmpty);
  updateFooter(false);
}

// ── Inspector ────────────────────────────────────────────────────────────────
closeInsp.addEventListener('click', () => {
  inspector.classList.remove('visible');
  controls.autoRotate = true;
});

exportBtn.addEventListener('click', () => {
  const payload = {
    nodes: graphData.nodes.map(n => ({
      id: n.id, label: n.label, activation: n.activation,
      peak: n.peak, layer: n.layer
    })),
    links: graphData.links.map(l => ({
      source: typeof l.source === 'object' ? l.source.id : l.source,
      target: typeof l.target === 'object' ? l.target.id : l.target,
      value: l.value,
    })),
  };
  const a = Object.assign(document.createElement('a'), {
    href: 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(payload, null, 2)),
    download: 'axon_graph.json',
  });
  document.body.appendChild(a); a.click(); a.remove();
});

Graph.onBackgroundClick(() => {
  inspector.classList.remove('visible');
  controls.autoRotate = true;   // resume idle spin
});

Graph.onNodeClick(node => {
  controls.autoRotate = false;  // stop spin while inspecting
  inspector.classList.add('visible');

  const related = graphData.links
    .filter(l => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      return s === node.id || t === node.id;
    })
    .map(l => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      const otherId = s === node.id ? t : s;
      const other = nodeMap.get(otherId);
      return other ? { label: other.label || `F${other.id}`, value: l.value } : null;
    })
    .filter(Boolean)
    .sort((a, b) => b.value - a.value)
    .slice(0, 5);

  const coHtml = related.length
    ? related.map(r =>
        `<span class="co-chip" title="${r.label}">${r.label.slice(0,22)} ×${r.value}</span>`
      ).join('')
    : `<span style="color:var(--t3)">none yet</span>`;

  inspContent.innerHTML = `
    <div class="insp-row">
      <span class="insp-lbl">Feature index</span>
      <span class="insp-val mono">${node.id}</span>
    </div>
    <div class="insp-row">
      <span class="insp-lbl">Label</span>
      <span class="insp-val">${node.label || `Feature ${node.id}`}</span>
    </div>
    <div class="insp-row">
      <span class="insp-lbl">Layer</span>
      <span class="insp-val mono">${node.layer ?? 8}</span>
    </div>
    <div class="insp-row">
      <span class="insp-lbl">Activation &nbsp;/&nbsp; Peak</span>
      <span class="insp-val mono">${(node.activation||0).toFixed(3)} &nbsp;/&nbsp; ${(node.peak||0).toFixed(3)}</span>
    </div>
    <div class="insp-row">
      <span class="insp-lbl">Co-activations</span>
      <span class="insp-val">${coHtml}</span>
    </div>
    <div class="insp-row">
      <span class="insp-lbl">Neuronpedia</span>
      <span class="insp-val">
        <a href="https://www.neuronpedia.org/gpt2-small/8-res-jb/${node.id}" target="_blank" rel="noopener">
          Open feature page ↗
        </a>
      </span>
    </div>
  `;

  // Fly camera to node (short tween so controls re-enable fast)
  const dist = 90;
  const dr = 1 + dist / (Math.hypot(node.x||1, node.y||1, node.z||1) || 1);
  Graph.cameraPosition(
    { x: (node.x||0)*dr, y: (node.y||0)*dr, z: (node.z||0)*dr },
    node, 500
  );
  // Belt-and-suspenders: guarantee controls come back after tween
  setTimeout(() => {
    controls.enabled = true;
    controls.update();
  }, 600);
});

// ── Graph update ─────────────────────────────────────────────────────────────
function linkKey(a, b) { return a < b ? `${a}|${b}` : `${b}|${a}`; }

function updateGraph(features) {
  if (!features || !features.length) return;
  let changed = false;

  for (const f of features) {
    if (f.activation > maxActSeen) maxActSeen = f.activation;
    const id = String(f.id);
    let node = nodeMap.get(id);
    if (!node) {
      node = { id, label: f.label, activation: f.activation, peak: f.activation, layer: f.layer };
      nodeMap.set(id, node);
      graphData.nodes.push(node);
      changed = true;
    } else {
      node.activation = Math.max(node.activation, f.activation);
      if (f.activation > (node.peak || 0)) node.peak = f.activation;
      if (f.label && (!node.label || node.label.startsWith('Feature '))) node.label = f.label;
    }
  }

  for (let i = 0; i < features.length; i++) {
    for (let j = i + 1; j < features.length; j++) {
      const a = String(features[i].id), b = String(features[j].id);
      const k = linkKey(a, b);
      const lnk = linkMap.get(k);
      if (lnk) { lnk.value = (lnk.value || 1) + 1; }
      else {
        const nl = { source: a, target: b, value: 1 };
        linkMap.set(k, nl);
        graphData.links.push(nl);
        changed = true;
      }
    }
  }

  if (changed) {
    Graph.graphData({ nodes: graphData.nodes, links: graphData.links });
  } else {
    Graph.nodeThreeObject(buildNodeObj);
  }
  updateFooter(true);
}

function updateFooter(active) {
  statNodes.textContent = `${graphData.nodes.length} nodes`;
  statEdges.textContent = `${graphData.links.length} edges`;
  if (statusDot) {
    statusDot.classList.toggle('off', !active || graphData.nodes.length === 0);
  }
}

function renderConcepts(features) {
  const sorted = [...features].sort((a, b) => b.activation - a.activation).slice(0, 10);
  const maxAct = Math.max(...sorted.map(f => f.activation), 1e-6);

  if (conceptsEmpty) conceptsEmpty.remove();
  conceptsList.innerHTML = '';

  for (const f of sorted) {
    const li   = document.createElement('li');
    const lbl  = Object.assign(document.createElement('span'), {
      className: 'concept-label',
      textContent: f.label || `Feature ${f.id}`,
      title: f.label || `Feature ${f.id}`,
    });
    const bar  = document.createElement('span'); bar.className = 'concept-bar';
    const fill = document.createElement('span'); fill.className = 'concept-bar-inner';
    fill.style.width = `${Math.round((f.activation / maxAct) * 100)}%`;
    bar.appendChild(fill);
    const val  = Object.assign(document.createElement('span'), {
      className: 'concept-val',
      textContent: (f.activation / maxAct).toFixed(2),
    });
    li.append(lbl, bar, val);
    conceptsList.appendChild(li);
  }
}

// ── WebSocket ────────────────────────────────────────────────────────────────
let ws = null;

function setGenerating(on) {
  startBtn.style.display = on ? 'none' : '';
  stopBtn.style.display  = on ? '' : 'none';
  startBtn.disabled      = on;
  cursorEl.style.display = on ? '' : 'none';
  if (statusDot) statusDot.classList.toggle('off', !on);
}

function connectAndGenerate() {
  if (ws && ws.readyState <= 1) { try { ws.close(); } catch (_) {} }

  const prompt = promptInput.value.trim();
  if (!prompt) return;

  promptTextEl.textContent = prompt;
  generatedEl.textContent  = '';
  tokenBadge.style.display = 'inline';
  tokenBadge.textContent   = '0 tok';
  speedBadge.style.display = 'none';
  totalTokens = 0;
  genStart    = performance.now();

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws`);

  ws.onopen = () => {
    setGenerating(true);
    ws.send(JSON.stringify({ prompt, ...getParams() }));
  };

  ws.onmessage = ({ data }) => {
    let msg;
    try { msg = JSON.parse(data); } catch { return; }

    if (msg.status === 'done') {
      setGenerating(false);
      try { ws.close(); } catch (_) {}
      return;
    }
    if (msg.error) {
      generatedEl.textContent += ` [${msg.error}]`;
      return;
    }
    if (typeof msg.token === 'string') {
      generatedEl.textContent += msg.token;
      const tb = generatedEl.closest('.text-box');
      if (tb) tb.scrollTop = tb.scrollHeight;
      totalTokens++;
      tokenBadge.textContent = `${totalTokens} tok`;
      const elapsed = (performance.now() - genStart) / 1000;
      if (elapsed > 0.5) {
        speedBadge.style.display = 'inline';
        speedBadge.textContent   = `${(totalTokens / elapsed).toFixed(1)} tok/s`;
      }
    }
    if (msg.features && msg.features.length) {
      renderConcepts(msg.features);
      updateGraph(msg.features);
    }
  };

  ws.onclose = () => setGenerating(false);
  ws.onerror = () => setGenerating(false);
}

function stopGeneration() {
  if (ws) { try { ws.close(); } catch (_) {} ws = null; }
  setGenerating(false);
}

startBtn.addEventListener('click', connectAndGenerate);
stopBtn.addEventListener('click', stopGeneration);
promptInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); connectAndGenerate(); }
});

// ── Auto-run from URL params (demo.py) ──────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  const p = new URLSearchParams(location.search);
  if (p.get('autorun') === '1') {
    const pr = p.get('prompt');
    if (pr) promptInput.value = pr;
    setTimeout(connectAndGenerate, 800);
  }
});
