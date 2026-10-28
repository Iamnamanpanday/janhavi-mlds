import pytest, struct
from app.features import PEFeatureExtractor, InvalidPE, constraints, GROUPS
def minimal_pe():
    b=bytearray(512); b[:2]=b'MZ'; struct.pack_into('<I',b,0x3c,0x80); b[0x80:0x84]=b'PE\0\0'
    struct.pack_into('<HHIIIHH',b,0x84,0x14c,0,0,0,0,0xe0,0x102)
    struct.pack_into('<H',b,0x98,0x10b); return bytes(b)
def test_training_serving_feature_parity(tmp_path):
    data=minimal_pe(); path=tmp_path/'sample.exe';path.write_bytes(data)
    a,ma=PEFeatureExtractor().extract_bytes(data); b,mb=PEFeatureExtractor().extract_path(path)
    assert (a==b).all() and ma==mb
def test_corrupt_rejected():
    with pytest.raises(InvalidPE): PEFeatureExtractor().extract_bytes(b'MZ'+b'\0'*62)
def test_constraints_reference_groups():
    assert constraints()==list(GROUPS.values())
