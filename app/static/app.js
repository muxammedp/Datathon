const $ = (selector) => document.querySelector(selector);
const dom = (tag, className, text) => {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
};
const icon = (name) => {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.append(use);
  svg.setAttribute('aria-hidden', 'true');
  return svg;
};
const timecode = (seconds) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
let apiKey = '';
let selectedFile = null;
let videoUrl = null;
let activeJob = null;
let lastResult = null;
let busy = false;
let ready = false;
let started = 0;
let pollTimer;
let elapsedTimer;
let toastTimer;

function notify(message) {
  $('#toast').textContent = message;
  $('#toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 4200);
}

function showError(message) {
  $('#uploadError').textContent = message;
  $('#uploadError').hidden = false;
}

function updateButton() {
  $('#analyzeButton').disabled = !selectedFile || busy || !ready;
  $('#removeFile').disabled = busy;
  $('#fileInput').disabled = busy;
}

function route() {
  const hash = location.hash.slice(1);
  const page = ['rules', 'api'].includes(hash) ? hash : 'analyze';
  document.querySelectorAll('.page').forEach((el) => { el.hidden = el.id !== `page-${page}`; });
  document.querySelectorAll('[data-page]').forEach((el) => {
    const active = el.dataset.page === page;
    el.classList.toggle('active', active);
    if (active) el.setAttribute('aria-current', 'page');
    else el.removeAttribute('aria-current');
  });
  $('#breadcrumbTitle').textContent = {analyze: 'Обзор и анализ', rules: 'Правила проверки', api: 'API для жюри'}[page];
  if (hash !== 'upload') window.scrollTo({top: 0, behavior: 'instant'});
}

async function health() {
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error();
    const data = await response.json();
    ready = data.engine === 'ready';
    $('#healthPill').className = `health-pill ${ready ? 'ready' : data.engine === 'unavailable' ? 'error' : ''}`;
    $('#healthPill span').textContent = ready ? (data.automatic_verdict_enabled === false ? 'Ручная проверка' : 'Модель готова') : data.engine === 'loading' ? 'Модель загружается' : 'Модель недоступна';
    $('#healthPill').title = data.message;
    $('#qualityNotice').textContent = data.automatic_verdict_enabled
      ? 'Экспериментальный бинарный анализ. Тип события и роль человека требуют ручной проверки; возможны ошибки.'
      : 'Автоматический вердикт приостановлен из-за ложных тревог и пропусков. Выделенный фрагмент не означает нарушение.';
    if (data.engine === 'unavailable') showError(data.message);
  } catch {
    ready = false;
    $('#healthPill').className = 'health-pill error';
    $('#healthPill span').textContent = 'Нет соединения';
  }
  updateButton();
}

function selectFile(file) {
  if (busy || !file) return;
  $('#uploadError').hidden = true;
  if (!/\.(mov|mp4)$/i.test(file.name)) { clearFile(); return showError('Поддерживаются только видео MOV и MP4.'); }
  if (file.size > 150 * 1024 * 1024) { clearFile(); return showError('Файл больше 150 МБ. Выберите более короткий ролик.'); }
  if (file.size === 0) { clearFile(); return showError('Файл пуст. Выберите другую запись.'); }
  selectedFile = file;
  lastResult = null;
  $('#results').hidden = true;
  $('#fileName').textContent = file.name;
  $('#fileSize').textContent = `${(file.size / 1024 / 1024).toFixed(1)} МБ · ${file.name.split('.').pop().toUpperCase()}`;
  $('#selectedFile').hidden = false;
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = URL.createObjectURL(file);
  $('#videoPlayer').src = videoUrl;
  $('#videoPlayer').hidden = false;
  $('#videoFallback').hidden = true;
  updateButton();
}

function clearFile() {
  if (busy) return;
  selectedFile = null;
  $('#fileInput').value = '';
  $('#selectedFile').hidden = true;
  updateButton();
}

function setProgress(percent, title, detail) {
  $('#progressPanel').hidden = false;
  $('#progressTitle').textContent = title;
  $('#progressDetail').textContent = detail;
  $('#progressPercent').textContent = `${percent}%`;
  $('#progressBar').value = percent;
}

function finishBusy() {
  busy = false;
  clearTimeout(pollTimer);
  clearInterval(elapsedTimer);
  $('#progressPanel').hidden = true;
  updateButton();
}

function errorText(data, fallback) {
  return typeof data?.detail === 'string' ? data.detail : fallback;
}

function upload() {
  if (!selectedFile || busy || !ready) return;
  if (!apiKey) {
    $('#accessDialog').showModal();
    $('#apiKey').focus();
    return;
  }
  busy = true;
  lastResult = null;
  $('#results').hidden = true;
  $('#uploadError').hidden = true;
  updateButton();
  started = Date.now();
  setProgress(0, 'Загружаем видео', 'Передача в локальный сервис');
  elapsedTimer = setInterval(() => { $('#elapsed').textContent = timecode((Date.now() - started) / 1000); }, 1000);
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/v1/analyze');
  xhr.setRequestHeader('X-API-Key', apiKey);
  xhr.timeout = 90000;
  xhr.upload.onprogress = (event) => {
    if (event.lengthComputable) setProgress(Math.round(event.loaded / event.total * 100), 'Загружаем видео', 'Передача в локальный сервис');
  };
  xhr.onload = () => {
    let data;
    try { data = JSON.parse(xhr.responseText); } catch { data = {}; }
    if (xhr.status !== 202 || !data.job_id) {
      finishBusy();
      showError(errorText(data, 'Не удалось загрузить видео. Повторите попытку.'));
      if (xhr.status === 401) { apiKey = ''; $('#accessDialog').showModal(); }
      return;
    }
    activeJob = data.job_id;
    setProgress(3, 'Видео принято', 'Начинаем локальную проверку');
    poll(0);
  };
  xhr.onerror = xhr.ontimeout = () => { finishBusy(); showError('Соединение прервано. Проверьте, что локальный сервер запущен.'); };
  const form = new FormData();
  form.append('file', selectedFile);
  xhr.send(form);
}

async function poll(retries) {
  try {
    const response = await fetch(`/api/v1/jobs/${activeJob}`, {headers: {'X-API-Key': apiKey}});
    const job = await response.json();
    if (!response.ok) throw new Error(errorText(job, 'Не удалось получить результат.'));
    if (job.status === 'done') {
      finishBusy();
      renderResult(job);
      return;
    }
    if (job.status === 'failed') {
      finishBusy();
      showError(job.error || 'Анализ завершился ошибкой.');
      return;
    }
    setProgress(job.progress || 3, job.stage || 'Обрабатываем видео', 'Проверяем визуальные признаки и последовательность действий');
    pollTimer = setTimeout(() => poll(0), 1800);
  } catch (error) {
    if (retries < 5) {
      setProgress($('#progressBar').value, 'Восстанавливаем соединение', 'Задание продолжает выполняться на сервере');
      pollTimer = setTimeout(() => poll(retries + 1), 3000);
    } else {
      finishBusy();
      showError(`${error.message} Идентификатор проверки: ${activeJob}`);
    }
  }
}

function seek(timestamp) {
  const video = $('#videoPlayer');
  if (video.readyState >= 1) {
    video.currentTime = timestamp;
    video.play().catch(() => {});
  } else notify('Для этого кодека используйте кадры-доказательства.');
}

function renderResult(job) {
  lastResult = job;
  const inconclusive = job.verdict === 'inconclusive';
  const violations = inconclusive ? (job.review_segments || []) : (job.violations || []);
  const clean = job.verdict === 'no_violation';
  $('#verdictTitle').textContent = inconclusive ? 'Недостаточно оснований для вывода' : clean ? 'Нарушение не обнаружено' : 'Обнаружены признаки возможных нарушений';
  $('#resultFilename').textContent = job.filename;
  $('#videoDuration').textContent = timecode(job.video.duration_sec);
  $('#timelineEnd').textContent = timecode(job.video.duration_sec);
  const summary = $('#resultSummary');
  summary.className = `result-summary${clean ? ' clean' : ''}`;
  summary.replaceChildren(icon(clean ? 'shield' : 'alert'));
  const summaryText = dom('div');
  summaryText.append(dom('strong', '', inconclusive ? 'Автоматический вердикт отключён. Нужен просмотр человеком.' : clean ? 'По проверенным кадрам признаков не найдено' : `Событий для проверки: ${violations.length}`));
  const coverage = job.analysis.whole_video_context ? `Вся запись в одном контексте · ${job.analysis.sampled_frames} выбранных кадров · ` : `${job.analysis.sampled_frames} кадров · `;
  summaryText.append(dom('p', '', `${coverage}${job.analysis.elapsed_sec.toFixed(1)} с · обработано локально`));
  summary.append(summaryText);
  $('#events').replaceChildren();
  $('#timeline').replaceChildren();
  violations.forEach((event, index) => {
    const card = dom('article', 'event-card');
    const top = dom('div', 'event-card-top');
    const timeButton = dom('button', 'event-time', timecode(event.timecode_sec));
    timeButton.title = 'Перейти к моменту в видео';
    timeButton.addEventListener('click', () => seek(event.timecode_sec));
    top.append(timeButton, dom('span', 'event-tag', `${inconclusive ? 'Фрагмент' : 'Событие'} ${String(index + 1).padStart(2, '0')} · требует проверки`));
    card.append(top, dom('h3', '', event.type), dom('p', '', event.description));
    const evidence = dom('div', 'evidence-grid');
    (event.evidence_frames || []).forEach((frame) => {
      const button = dom('button', 'evidence-button');
      button.setAttribute('aria-label', `Увеличить кадр ${timecode(frame.timestamp_sec)}`);
      const image = dom('img');
      image.src = `data:image/jpeg;base64,${frame.image_base64}`;
      image.alt = `Кадр для просмотра на ${timecode(frame.timestamp_sec)}`;
      image.loading = 'lazy';
      button.append(image, dom('span', '', timecode(frame.timestamp_sec)));
      button.addEventListener('click', () => {
        $('#largeFrame').src = image.src;
        $('#frameCaption').textContent = `${event.type} · ${frame.timestamp_sec.toFixed(2)} с`;
        $('#frameDialog').showModal();
      });
      evidence.append(button);
    });
    card.append(evidence);
    if (event.rule) {
      const source = dom('a', 'rule-reference', `НБРК № 231 · ${event.rule.section} ↗`);
      source.href = 'https://adilet.zan.kz/rus/docs/V1900019680';
      source.target = '_blank';
      source.rel = 'noreferrer';
      card.append(source);
      const details = dom('details');
      details.append(dom('summary', '', 'Основание и границы вывода'), dom('p', '', event.rule.text), dom('p', '', event.limitation));
      card.append(details);
    }
    $('#events').append(card);
    const marker = dom('button');
    marker.style.left = `${Math.min(100, Math.max(0, event.timecode_sec / job.video.duration_sec * 100))}%`;
    marker.title = `${timecode(event.timecode_sec)} — ${event.type}`;
    marker.setAttribute('aria-label', marker.title);
    marker.addEventListener('click', () => seek(event.timecode_sec));
    $('#timeline').append(marker);
  });
  if (clean || (inconclusive && !violations.length)) {
    const empty = dom('div', 'empty-result');
    empty.append(dom('h3', '', inconclusive ? 'Просмотрите запись вручную' : 'Проверка завершена'), dom('p', '', inconclusive ? 'Модель не прошла проверку качества. Сервис не подтверждает ни наличие, ни отсутствие нарушения и не назначает человеку роль кассира или клиента.' : 'Модель не обнаружила поддерживаемых событий в выбранных кадрах. Это не подтверждает отсутствие всех возможных нарушений: короткие действия, перекрытия и события за пределами кадра могут остаться незамеченными.'));
    $('#events').append(empty);
  }
  $('#analysisNotice').textContent = `${job.analysis.notice} Результат хранится локально 24 часа.`;
  $('#results').hidden = false;
  $('#results').scrollIntoView({behavior: 'smooth', block: 'start'});
}

async function loadRules() {
  try {
    const response = await fetch('/api/rules');
    if (!response.ok) throw new Error();
    const rules = await response.json();
    $('#rulesList').replaceChildren();
    rules.forEach((rule, index) => {
      const card = dom('article', 'panel rule-card');
      card.append(dom('div', 'rule-number', `ПРАВИЛО ${String(index + 1).padStart(2, '0')} · ${rule.section.toUpperCase()}`), dom('h2', '', rule.title), dom('p', '', rule.rule), dom('p', 'rule-limit', rule.limitation));
      $('#rulesList').append(card);
    });
  } catch { $('#rulesList').textContent = 'Правила не загрузились. Проверьте подключение к серверу и обновите страницу.'; }
}

$('#fileInput').addEventListener('change', (event) => selectFile(event.target.files[0]));
$('#removeFile').addEventListener('click', clearFile);
$('#analyzeButton').addEventListener('click', upload);
['dragenter', 'dragover'].forEach((type) => $('#dropzone').addEventListener(type, (event) => { event.preventDefault(); if (!busy) $('#dropzone').classList.add('dragging'); }));
['dragleave', 'drop'].forEach((type) => $('#dropzone').addEventListener(type, (event) => { event.preventDefault(); $('#dropzone').classList.remove('dragging'); }));
$('#dropzone').addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));
$('#accessButton').addEventListener('click', () => { $('#authError').hidden = true; $('#accessDialog').showModal(); $('#apiKey').focus(); });
$('#closeDialog').addEventListener('click', () => $('#accessDialog').close());
$('#closeFrame').addEventListener('click', () => $('#frameDialog').close());
$('#accessForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const key = $('#apiKey').value.trim();
  const button = $('#accessForm button[type="submit"]');
  button.disabled = true;
  try {
    const response = await fetch('/api/v1/auth', {headers: {'X-API-Key': key}});
    if (!response.ok) throw new Error('Ключ не подошёл. Проверьте содержимое data/api-key.txt.');
    apiKey = key;
    $('#apiKey').value = '';
    $('#accessDialog').close();
    $('#accessButton span').textContent = 'Подключено';
    notify('Доступ подтверждён. Можно запускать проверку.');
  } catch (error) {
    $('#authError').textContent = error.message;
    $('#authError').hidden = false;
  } finally { button.disabled = false; }
});
$('#videoPlayer').addEventListener('error', () => { $('#videoPlayer').hidden = true; $('#videoFallback').hidden = false; });
$('#exportButton').addEventListener('click', () => {
  if (!lastResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult, null, 2)], {type: 'application/json'}));
  const link = dom('a');
  link.href = url;
  link.download = `fortevision-${lastResult.job_id}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$('#deleteButton').addEventListener('click', async () => {
  if (!lastResult) return;
  if (!window.confirm('Удалить этот результат и кадры-доказательства с локального сервера?')) return;
  try {
    const response = await fetch(`/api/v1/jobs/${lastResult.job_id}`, {method: 'DELETE', headers: {'X-API-Key': apiKey}});
    if (!response.ok) throw new Error('Не удалось удалить результат.');
    $('#results').hidden = true;
    lastResult = null;
    $('#events').replaceChildren();
    notify('Результат и кадры удалены.');
  } catch (error) { notify(error.message); }
});
const apiExample = `curl -X POST ${location.origin}/api/v1/analyze \\\n  -H "X-API-Key: YOUR_API_KEY" \\\n  -F "file=@video.mov"`;
$('#curlExample').textContent = apiExample;
$('#copyApi').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(apiExample); notify('Пример запроса скопирован.'); }
  catch { notify('Выделите и скопируйте пример вручную.'); }
});
window.addEventListener('hashchange', route);
window.addEventListener('beforeunload', (event) => { if (busy) { event.preventDefault(); event.returnValue = ''; } });
route();
health();
loadRules();
setInterval(health, 12000);
