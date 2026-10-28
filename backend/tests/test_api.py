import os, tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
@pytest.fixture(scope='session',autouse=True)
def artifacts():
    d=tempfile.mkdtemp();os.environ['ARTIFACT_DIR']=d
    from app import train
    train.ARTIFACT_DIR=Path(d); import sys
    old=sys.argv;sys.argv=['train','--smoke'];train.main();sys.argv=old
def test_non_pe_rejected():
    from app.main import app
    r=TestClient(app).post('/api/scan',files={'file':('x.exe',b'not a pe')}); assert r.status_code==400
def test_empty_rejected():
    from app.main import app
    assert TestClient(app).post('/api/scan',files={'file':('x.exe',b'')}).status_code==400
def test_model_schema():
    from app.main import app
    j=TestClient(app).get('/api/model').json();assert {'model_version','variants','threshold_at_0_1pct_fpr'}<=j.keys()
def test_pe_response_schema():
    from app.main import app
    from test_features import minimal_pe
    r=TestClient(app).post('/api/scan',files={'file':('sample.exe',minimal_pe())})
    assert r.status_code==200 and {'sha256','verdict','score','top_features','group_contributions'}<=r.json().keys()
