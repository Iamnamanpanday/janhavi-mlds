from __future__ import annotations
import asyncio, hashlib, json, logging, os, sqlite3, tempfile, time, multiprocessing
from collections import defaultdict, deque
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib, numpy as np
import shap
from xgboost import XGBClassifier
from .config import ARTIFACT_DIR, DB_PATH, MAX_UPLOAD_BYTES, RATE_LIMIT, SCAN_TIMEOUT
from .features import FEATURES, GROUPS, InvalidPE, PEFeatureExtractor

log=logging.getLogger('scanner'); logging.basicConfig(level=logging.INFO,format='%(message)s')
app=FastAPI(title='Static PE Risk Scanner',version='1.0')
app.add_middleware(CORSMiddleware,allow_origins=os.getenv('CORS_ORIGINS','http://localhost:3000').split(','),allow_methods=['*'],allow_headers=['*'])
hits=defaultdict(deque)
def card():
    try: return json.loads((ARTIFACT_DIR/'model_card.json').read_text())
    except FileNotFoundError: raise HTTPException(503,'Model artifacts missing. Run `make smoke` or train with EMBER.')
def init_db():
    con=sqlite3.connect(DB_PATH); con.execute('CREATE TABLE IF NOT EXISTS scans (id INTEGER PRIMARY KEY,created TEXT DEFAULT CURRENT_TIMESTAMP,sha256 TEXT,filename TEXT,verdict TEXT,score REAL,latency_ms INTEGER)'); con.commit(); con.close()
@app.on_event('startup')
def startup(): init_db(); card()
@app.middleware('http')
async def rate(request:Request, call_next):
    q=hits[request.client.host]; now=time.time()
    while q and q[0]<now-60:q.popleft()
    if len(q)>=RATE_LIMIT:return JSONResponse({'detail':'Rate limit exceeded'},429)
    q.append(now); return await call_next(request)
@app.get('/api/health')
def health(): return {'status':'ok','static_analysis_only':True,'model_version':card()['model_version']}
@app.get('/api/model')
def model(): return card()
@app.get('/api/history')
def history(limit:int=50):
    con=sqlite3.connect(DB_PATH); con.row_factory=sqlite3.Row; rows=[dict(r) for r in con.execute('SELECT * FROM scans ORDER BY id DESC LIMIT ?', (min(max(limit,1),200),))];con.close();return rows
def _parse_worker(path: str, conn):
    """Separate parser process: no execution path, bounded memory where supported."""
    try:
        if os.name != 'nt':
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CPU, (max(1, SCAN_TIMEOUT), max(1, SCAN_TIMEOUT)))
        conn.send(('ok', PEFeatureExtractor().extract_path(Path(path))))
    except Exception as e: conn.send(('error', str(e)))
    finally: conn.close()
def _extract_process(path: str):
    parent, child=multiprocessing.Pipe(False); p=multiprocessing.get_context('spawn').Process(target=_parse_worker,args=(path,child)); p.start(); child.close()
    if not parent.poll(SCAN_TIMEOUT): p.terminate(); p.join(); raise TimeoutError()
    kind,payload=parent.recv(); p.join()
    if kind=='error': raise InvalidPE(payload)
    return payload
async def bounded_extract(data:bytes):
    # The restricted temp file is the only on-disk copy and is removed even on parser failure.
    fd,path=tempfile.mkstemp(prefix='static-pe-',suffix='.bin'); os.close(fd)
    try:
        Path(path).write_bytes(data); os.chmod(path,0o400)
        return await asyncio.to_thread(_extract_process,path)
    finally:
        try: os.chmod(path,0o600); os.unlink(path)
        except FileNotFoundError: pass
def explain(model,x):
    # TreeExplainer exposes the feature-level contribution of this actual prediction.
    values=np.asarray(shap.TreeExplainer(model).shap_values(x.reshape(1,-1))).reshape(-1)
    contrib=values
    top=sorted(({'feature':FEATURES[i],'contribution':round(float(contrib[i]),4)} for i in range(len(x))),key=lambda z:abs(z['contribution']),reverse=True)[:8]
    groups={g:round(float(sum(contrib[i] for i in ids)),4) for g,ids in GROUPS.items()}; return top,groups
async def scan_one(upload:UploadFile):
    start=time.perf_counter(); data=await upload.read(MAX_UPLOAD_BYTES+1)
    if not data: raise HTTPException(400,'Empty upload')
    if len(data)>MAX_UPLOAD_BYTES: raise HTTPException(413,'File exceeds maximum upload size')
    digest=hashlib.sha256(data).hexdigest()
    try: x,meta=await bounded_extract(data)
    except InvalidPE as e: raise HTTPException(400,str(e))
    except TimeoutError: raise HTTPException(408,'PE parser timed out')
    finally: del data
    m=XGBClassifier();m.load_model(ARTIFACT_DIR/'model.json'); cal=joblib.load(ARTIFACT_DIR/'calibrator.pkl'); raw=m.predict_proba(x.reshape(1,-1))[:,1]; score=float(cal.predict(raw)[0]); c=card(); threshold=c['threshold_at_0_1pct_fpr']
    verdict='Malicious' if score>=threshold else ('Suspicious' if score>=threshold*.6 else 'Benign'); top,groups=explain(m,x)
    result={'sha256':digest,'verdict':verdict,'score':round(score,4),'threshold':threshold,'confidence_band':'high' if abs(score-threshold)>.2 else 'borderline','top_features':top,'group_contributions':groups,'model_version':c['model_version'],'latency_ms':round((time.perf_counter()-start)*1000),'file_metadata':meta,'disclaimer':'Probabilistic static detector. Files are never executed.'}
    con=sqlite3.connect(DB_PATH);con.execute('INSERT INTO scans(sha256,filename,verdict,score,latency_ms) VALUES (?,?,?,?,?)',(digest,upload.filename or '',verdict,score,result['latency_ms']));con.commit();con.close();log.info(json.dumps({'event':'scan','sha256':digest,'size':meta['size'],'verdict':verdict}));return result
@app.post('/api/scan')
async def scan(file:UploadFile=File(...)): return await scan_one(file)
@app.post('/api/scan/batch')
async def batch(files:list[UploadFile]=File(...)): return {'results':[await scan_one(f) for f in files]}
