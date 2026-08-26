const controls = [
  ['px', 'X', -0.2, 0.3, 0.002, 'positionControls'], ['py', 'Y', -0.3, 0.3, 0.002, 'positionControls'], ['pz', 'Z', 0.2, 0.7, 0.002, 'positionControls'],
  ['rx', 'rx', -3.14, 3.14, 0.01, 'rotationControls'], ['ry', 'ry', -3.14, 3.14, 0.01, 'rotationControls'], ['rz', 'rz', -3.14, 3.14, 0.01, 'rotationControls'],
  ['fx', 'fx', 160, 1000, 1, 'intrinsicControls'], ['fy', 'fy', 160, 1000, 1, 'intrinsicControls'], ['cx', 'cx', 120, 520, 1, 'intrinsicControls'], ['cy', 'cy', 80, 400, 1, 'intrinsicControls'],
];
const orbitControls = [
  ['orbit_azimuth', '方位角', -3.14, 3.14, 0.01, 'orbitControls'],
  ['orbit_elevation', '仰角', -1.45, 1.45, 0.01, 'orbitControls'],
  ['orbit_distance', '距离 / m', 0.12, 1.5, 0.005, 'orbitControls'],
  ['orbit_roll', '画面滚转', -3.14, 3.14, 0.01, 'orbitControls'],
];
const clientDefaults = { view_mode: 'raw', orbit_azimuth: -2.2, orbit_elevation: 0.4, orbit_distance: 0.55, orbit_roll: 0 };
const allControls = [...controls, ...orbitControls];
const state = { defaults: null, frameCount: 1, rendered: false, timer: null, lastRender: null, backendSupportsOrbit: false };
const $ = (id) => document.getElementById(id);

function setStatus(message, kind = 'ready') {
  $('statusText').textContent = message;
  $('statusDot').style.background = kind === 'error' ? '#f27445' : kind === 'busy' ? '#e3b33a' : '#c7f252';
}
function parameterValue(key) { return Number($(`${key}_number`).value); }
function setControlValue(key, value, trigger = true) {
  const range = $(key);
  const number = $(`${key}_number`);
  range.value = value;
  number.value = value;
  range.closest('.parameter').querySelector('output').textContent = Number(value).toFixed(Number(range.step) < 1 ? 3 : 0);
  if (trigger) number.dispatchEvent(new Event('input'));
}
function parameters() {
  const values = Object.fromEntries(allControls.map(([key]) => [key, parameterValue(key)]));
  return { ...values, frame: Number($('frame').value), state_offset: Number($('state_offset').value), mask_kind: $('mask_kind').value, view_mode: $('view_mode').value };
}
function queryString(object) { return new URLSearchParams(object).toString(); }
function scheduleRender() { clearTimeout(state.timer); state.timer = setTimeout(render, 280); }

function createControls(defaults) {
  const template = $('parameterTemplate');
  allControls.forEach(([key, label, min, max, step, destination]) => {
    const node = template.content.cloneNode(true);
    const range = node.querySelector('input[type="range"]');
    const number = node.querySelector('.number');
    const code = node.querySelector('code');
    const output = node.querySelector('output');
    code.textContent = label; range.id = key; range.min = min; range.max = max; range.step = step; range.value = defaults[key];
    number.id = `${key}_number`;
    number.min = min; number.max = max; number.step = step; number.value = defaults[key];
    output.textContent = Number(defaults[key]).toFixed(step < 1 ? 3 : 0);
    const sync = (source, target) => { target.value = source.value; output.textContent = Number(source.value).toFixed(step < 1 ? 3 : 0); scheduleRender(); };
    range.addEventListener('input', () => sync(range, number)); number.addEventListener('input', () => sync(number, range));
    $(destination).appendChild(node);
  });
}

async function loadEpisodes() {
  const dataset = $('dataset').value;
  const response = await fetch(`/api/episodes?${queryString({ dataset })}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '无法读取 episode');
  $('episode').innerHTML = data.episodes.map((episode) => `<option value="${episode}">episode ${String(episode).padStart(6, '0')}</option>`).join('');
}
async function loadInfo() {
  const response = await fetch(`/api/info?${queryString({ dataset: $('dataset').value, episode: $('episode').value })}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '无法读取视频信息');
  state.frameCount = data.frame_count;
  $('frame').max = Math.max(0, data.frame_count - 1);
  clampFrameToState();
}
function clampFrameToState() {
  const firstValidFrame = Math.max(0, -Number($('state_offset').value));
  const lastFrame = Number($('frame').max);
  const frame = Math.max(firstValidFrame, Math.min(lastFrame, Number($('frame').value)));
  $('frame').value = frame;
  $('frameOutput').textContent = String(frame);
}
async function render() {
  if (!$('dataset').value) return;
  setStatus('FK 渲染中…', 'busy');
  const response = await fetch(`/api/render?${queryString({ dataset: $('dataset').value, episode: $('episode').value, ...parameters() })}`);
  const data = await response.json();
  if (!response.ok) { setStatus(data.error || '渲染失败', 'error'); return; }
  $('rawImage').src = data.raw; $('overlayImage').src = data.overlay; $('meshImage').src = data.mesh;
  $('overlayCaption').textContent = `RGB + ${data.mask_kind} mask`;
  $('frameTitle').textContent = `frame ${data.frame}  /  state ${data.state_index}`;
  $('metrics').textContent = `${data.mask_pixels.toLocaleString()} mask pixels · ${data.render_ms} ms · ${state.frameCount} frames`;
  const camera = data.resolved_camera;
  $('resolvedPose').textContent = camera
    ? `pos [${camera.position.map((value) => value.toFixed(4)).join(', ')}]\neuler XYZ [${camera.euler_intrinsic_xyz_rad.map((value) => value.toFixed(4)).join(', ')}]`
    : '当前服务为旧后端：原始 6DoF 自动渲染可用；重启服务后启用 orbit。';
  state.lastRender = data;
  state.rendered = true; setStatus(state.backendSupportsOrbit ? '已更新' : '已更新 · orbit 需重启服务', 'ready');
}
function reset() {
  const defaults = state.defaults; allControls.forEach(([key]) => setControlValue(key, defaults[key]));
  $('state_offset').value = defaults.state_offset; $('stateOffsetValue').textContent = defaults.state_offset; $('mask_kind').value = defaults.mask_kind; $('view_mode').value = defaults.view_mode; $('frame').value = 0; clampFrameToState(); render();
}
function initializeOrbit() {
  if (!state.backendSupportsOrbit) { setStatus('重启后端服务后才能使用 orbit', 'error'); return; }
  if (!state.lastRender) return;
  const orbit = state.lastRender.orbit_equivalent;
  const values = { orbit_azimuth: orbit.azimuth, orbit_elevation: orbit.elevation, orbit_distance: orbit.distance, orbit_roll: 0 };
  Object.entries(values).forEach(([key, value]) => setControlValue(key, value));
  $('view_mode').value = 'orbit'; render();
}
function freezeOrbit() {
  if (!state.backendSupportsOrbit) { setStatus('重启后端服务后才能固化 orbit', 'error'); return; }
  if (!state.lastRender || state.lastRender.view_mode !== 'orbit') { setStatus('请先进入绕左手观察模式', 'error'); return; }
  const camera = state.lastRender.resolved_camera;
  const values = { px: camera.position[0], py: camera.position[1], pz: camera.position[2], rx: camera.euler_intrinsic_xyz_rad[0], ry: camera.euler_intrinsic_xyz_rad[1], rz: camera.euler_intrinsic_xyz_rad[2] };
  Object.entries(values).forEach(([key, value]) => setControlValue(key, value));
  $('view_mode').value = 'raw'; render();
}
async function savePreset() {
  if ($('view_mode').value === 'orbit') { setStatus('保存前请先把 orbit 固化为静态安装位姿', 'error'); return; }
  const name = window.prompt('预设名称（字母、数字、_、-）：', 'head_camera_candidate'); if (!name) return;
  const response = await fetch('/api/presets', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, dataset: $('dataset').value, episode: Number($('episode').value), parameters: parameters() }) });
  const data = await response.json(); setStatus(response.ok ? `已保存 ${data.saved}` : (data.error || '保存失败'), response.ok ? 'ready' : 'error');
}
async function initialize() {
  try {
    setStatus('载入数据集…', 'busy');
    const response = await fetch('/api/datasets'); const data = await response.json(); if (!response.ok) throw new Error(data.error || '无法发现数据集');
    const backendDefaults = data.defaults || {};
    state.backendSupportsOrbit = backendDefaults.view_mode === 'raw' && Number.isFinite(Number(backendDefaults.orbit_distance));
    state.defaults = { ...clientDefaults, ...backendDefaults };
    $('state_offset').value = state.defaults.state_offset; $('stateOffsetValue').textContent = state.defaults.state_offset; $('mask_kind').value = state.defaults.mask_kind; $('view_mode').value = state.defaults.view_mode; $('dataset').innerHTML = data.datasets.map((path) => `<option value="${path}">${path}</option>`).join('');
    const preferred = data.datasets.find((path) => path === '8_13_shake_beaker_1'); if (preferred) $('dataset').value = preferred;
    createControls(state.defaults);
    if (!state.backendSupportsOrbit) {
      $('view_mode').disabled = true; $('initializeOrbit').disabled = true; $('freezeOrbit').disabled = true;
      orbitControls.forEach(([key]) => { $(key).disabled = true; $(`${key}_number`).disabled = true; });
    }
    await loadEpisodes(); await loadInfo(); await render();
  } catch (error) { setStatus(error.message, 'error'); }
}
$('dataset').addEventListener('change', async () => { try { await loadEpisodes(); await loadInfo(); await render(); } catch (error) { setStatus(error.message, 'error'); } });
$('episode').addEventListener('change', async () => { try { await loadInfo(); await render(); } catch (error) { setStatus(error.message, 'error'); } });
$('frame').addEventListener('input', () => { $('frameOutput').textContent = $('frame').value; scheduleRender(); });
$('state_offset').addEventListener('input', () => { $('stateOffsetValue').textContent = $('state_offset').value; clampFrameToState(); scheduleRender(); });
$('mask_kind').addEventListener('change', render); $('renderButton').addEventListener('click', render); $('resetButton').addEventListener('click', reset); $('saveButton').addEventListener('click', savePreset);
$('view_mode').addEventListener('change', render); $('initializeOrbit').addEventListener('click', initializeOrbit); $('freezeOrbit').addEventListener('click', freezeOrbit);
document.addEventListener('keydown', (event) => { if (event.key.toLowerCase() === 'r' && !['INPUT', 'SELECT'].includes(document.activeElement.tagName)) { event.preventDefault(); render(); } });
initialize();
