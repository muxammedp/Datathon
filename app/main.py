import asyncio
import hmac
import json
import logging
import os
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.engine import video_info
from app.full_video import FullVideoAnalyzer, VERSION as MODEL_NAME
from app.rules import public_rules

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('FORTE_DATA_DIR', ROOT / 'data'))
DATA.mkdir(parents=True, exist_ok=True, mode=0o700)
JOBS = DATA / 'jobs'
JOBS.mkdir(parents=True, exist_ok=True, mode=0o700)
KEY_FILE = DATA / 'api-key.txt'
if os.environ.get('FORTE_API_KEY'):
    API_KEY = os.environ['FORTE_API_KEY']
else:
    if not KEY_FILE.exists():
        try:
            descriptor = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'w', encoding='utf8') as output:
                output.write(secrets.token_urlsafe(32))
        except FileExistsError:
            pass
    API_KEY = KEY_FILE.read_text(encoding='utf8').strip()
if len(API_KEY) < 24:
    raise RuntimeError('FORTE_API_KEY должен содержать не менее 24 символов.')

analyzer = FullVideoAnalyzer()
MODEL_NAME += '-lora'
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='local-inference')
job_lock = threading.RLock()
capacity = threading.BoundedSemaphore(1)
state = {'engine': 'loading', 'message': 'Загрузка локальной модели'}
MAX_BYTES = 150 * 1024 * 1024
RETENTION = 24 * 3600


def save_job(job):
    with job_lock:
        target = JOBS / job['job_id'] / 'result.json'
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(job, ensure_ascii=False), encoding='utf8')
        os.replace(tmp, target)


def load_job(job_id):
    try:
        if str(uuid.UUID(job_id)) != job_id:
            raise ValueError
    except ValueError:
        raise HTTPException(404, 'Проверка не найдена')
    with job_lock:
        path = JOBS / job_id / 'result.json'
        if not path.exists():
            raise HTTPException(404, 'Проверка не найдена')
        return json.loads(path.read_text(encoding='utf8'))


def clear_old_jobs():
    import shutil
    cutoff = time.time() - RETENTION
    for folder in JOBS.iterdir():
        if folder.is_dir() and folder.stat().st_mtime < cutoff:
            shutil.rmtree(folder)


def warmup():
    try:
        analyzer.load()
        message = ('Экспериментальный бинарный анализ готов. Тип нарушения проверяет человек.'
                   if getattr(analyzer, 'automatic_verdict_enabled', False) else
                   'Режим ручной проверки. Автоматический вердикт отключён: качество модели не подтверждено.')
        state.update(engine='ready', message=message)
    except Exception as exc:
        logging.exception('Model initialization failed')
        state.update(engine='unavailable', message=str(exc)[:300])


@asynccontextmanager
async def lifespan(app):
    clear_old_jobs()
    for result in JOBS.glob('*/result.json'):
        job = json.loads(result.read_text(encoding='utf8'))
        if job.get('status') == 'processing':
            job.update(status='failed', error='Сервис был перезапущен. Загрузите видео повторно.')
            save_job(job)
            for upload in result.parent.glob('input.*'):
                upload.unlink(missing_ok=True)
    if os.environ.get('FORTE_SKIP_WARMUP') != '1':
        executor.submit(warmup)
    async def cleanup_loop():
        while True:
            await asyncio.sleep(600)
            await asyncio.to_thread(clear_old_jobs)
    cleanup_task = asyncio.create_task(cleanup_loop())
    yield
    cleanup_task.cancel()
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title='ForteVision API', version='1.1.0', lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware('http')
async def security_headers(request: Request, call_next):
    if request.url.path == '/api/v1/analyze':
        key = request.headers.get('X-API-Key', '')
        if not key or not hmac.compare_digest(key.encode('utf8'), API_KEY.encode('utf8')):
            return JSONResponse({'detail': 'Неверный или отсутствующий X-API-Key'}, status_code=401)
        length = request.headers.get('content-length')
        if length and (not length.isdigit() or int(length) > MAX_BYTES + 1024 * 1024):
            return JSONResponse({'detail': 'Размер запроса превышает допустимый'}, status_code=413)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


def require_key(x_api_key: str | None = Header(default=None)):
    if not x_api_key or not hmac.compare_digest(x_api_key.encode('utf8'), API_KEY.encode('utf8')):
        raise HTTPException(401, 'Неверный или отсутствующий X-API-Key')


@app.get('/')
def index():
    return FileResponse(ROOT / 'app' / 'static' / 'index.html')


@app.get('/api/health')
def health():
    return {'service': 'ForteVision', 'status': 'ok', **state, 'model': MODEL_NAME, 'local_only': True,
            'analysis_mode': 'binary_review' if getattr(analyzer, 'automatic_verdict_enabled', False) else 'manual_review',
            'automatic_verdict_enabled': getattr(analyzer, 'automatic_verdict_enabled', False),
            'max_upload_mb': 150, 'max_duration_sec': 180, 'retention_hours': 24}


@app.get('/api/rules')
def rules():
    return public_rules()


@app.get('/api/v1/auth', dependencies=[Depends(require_key)])
def auth():
    return {'authenticated': True}


def process_job(job_id, path):
    job = load_job(job_id)
    started = time.monotonic()
    try:
        def progress(percent, stage):
            job.update(progress=percent, stage=stage)
            save_job(job)
        result = analyzer.analyze(path, progress)
        job.update(result, status='done', progress=100, stage='Анализ завершён')
    except Exception as exc:
        logging.exception('Video analysis failed for job %s', job_id)
        message = str(exc) if isinstance(exc, (ValueError, RuntimeError, TimeoutError)) else 'Ошибка обработки видео. Проверьте журнал сервера.'
        job.update(status='failed', error=message[:350], stage='Анализ не завершён')
    finally:
        job['processing_sec'] = round(time.monotonic() - started, 2)
        save_job(job)
        path.unlink(missing_ok=True)
        capacity.release()


@app.post('/api/v1/analyze', status_code=202, dependencies=[Depends(require_key)])
async def analyze(request: Request, file: UploadFile = File(...)):
    if state['engine'] != 'ready':
        raise HTTPException(503, state['message'])
    if Path(file.filename or '').suffix.lower() not in ('.mov', '.mp4'):
        raise HTTPException(415, 'Загрузите видео MOV или MP4')
    if not capacity.acquire(blocking=False):
        raise HTTPException(429, 'Сейчас обрабатывается другое видео. Повторите запрос после завершения.', headers={'Retry-After': '30'})
    job_id = str(uuid.uuid4())
    folder = JOBS / job_id
    folder.mkdir()
    path = folder / ('input' + Path(file.filename).suffix.lower())
    submitted = False
    try:
        total = 0
        with path.open('wb') as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(413, 'Размер видео превышает 150 МБ')
                output.write(chunk)
        if total == 0:
            raise HTTPException(422, 'Файл пуст')
        try:
            info = await asyncio.to_thread(video_info, path)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        job = {'job_id': job_id, 'status': 'processing', 'progress': 3, 'stage': 'Видео принято',
               'filename': Path(file.filename.replace('\\', '/')).name[:200],
               'created_at': time.time(), 'size_bytes': total, 'video': info}
        save_job(job)
        executor.submit(process_job, job_id, path)
        submitted = True
        return {'job_id': job_id, 'status': 'processing'}
    finally:
        await file.close()
        if not submitted:
            path.unlink(missing_ok=True)
            folder.rmdir()
            capacity.release()


@app.get('/api/v1/jobs/{job_id}', dependencies=[Depends(require_key)])
def get_job(job_id: str):
    job = load_job(job_id)
    if time.time() - job['created_at'] > RETENTION and job['status'] != 'processing':
        raise HTTPException(410, 'Срок хранения результата истёк')
    if job.get('status') == 'done' and job.get('analysis', {}).get('model', '').startswith('Qwen'):
        job['legacy_result_unreliable'] = True
        job['verdict'] = 'inconclusive'
        job['violations'] = []
        job['review_required'] = True
        job['analysis'] = {**job.get('analysis', {}), 'automatic_verdict_enabled': False,
                           'notice': 'Результат старой версии отозван из-за ложных срабатываний. Запустите новую проверку.'}
    return job


@app.delete('/api/v1/jobs/{job_id}', dependencies=[Depends(require_key)])
def delete_job(job_id: str):
    import shutil
    job = load_job(job_id)
    if job['status'] == 'processing':
        raise HTTPException(409, 'Дождитесь завершения обработки')
    with job_lock:
        shutil.rmtree(JOBS / job_id)
    return {'deleted': True}


app.mount('/static', StaticFiles(directory=ROOT / 'app' / 'static'), name='static')
