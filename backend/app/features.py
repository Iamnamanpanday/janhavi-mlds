"""Small, stable static PE feature schema shared by training and serving."""
from __future__ import annotations
import math, os, struct
from pathlib import Path
import pefile
import numpy as np

FEATURES = ["file_size_log", "entropy", "sections", "mean_section_entropy", "imports", "exports", "timestamp_age", "entrypoint", "rwx_sections", "overlay_ratio", "suspicious_strings", "mz_header"]
GROUPS = {"header": [0, 6, 7, 11], "sections": [2, 3, 8, 9], "imports": [4, 5], "strings": [10], "packer_indicators": [1, 8, 9, 10]}

class InvalidPE(ValueError): pass

def _entropy(data: bytes) -> float:
    if not data: return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / len(data)
    return float(-(p * np.log2(p)).sum())

class PEFeatureExtractor:
    version = "static-pe-v1"
    def extract_bytes(self, data: bytes) -> tuple[np.ndarray, dict]:
        if len(data) < 64 or data[:2] != b"MZ": raise InvalidPE("Not a Windows PE file (missing MZ header)")
        try:
            pe = pefile.PE(data=data, fast_load=True)
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'], pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']])
        except Exception as e: raise InvalidPE("Corrupt or unsupported PE file") from e
        secs = pe.sections
        sec_ent = [s.get_entropy() for s in secs] or [0.0]
        imports = sum(len(x.imports) for x in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []))
        exports = len(getattr(getattr(pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", []))
        rwx = sum(bool(s.Characteristics & 0xE0000000 == 0xE0000000) for s in secs)
        end = max([s.PointerToRawData + s.SizeOfRawData for s in secs] or [len(data)])
        overlay = max(0, len(data) - end) / max(1, len(data))
        strings = sum(data.lower().count(x) for x in (b"powershell", b"cmd.exe", b"http://", b"https://", b"virtualalloc"))
        values = np.array([math.log1p(len(data)), _entropy(data), len(secs), float(np.mean(sec_ent)), imports, exports, max(0, 2100 - (pe.FILE_HEADER.TimeDateStamp and 1970 + pe.FILE_HEADER.TimeDateStamp // 31557600)), pe.OPTIONAL_HEADER.AddressOfEntryPoint, rwx, overlay, strings, 1], dtype=np.float32)
        meta = {"size": len(data), "sections": len(secs), "entropy": round(_entropy(data), 3), "imports": imports}
        return values, meta
    def extract_path(self, path: Path) -> tuple[np.ndarray, dict]: return self.extract_bytes(path.read_bytes())

def constraints() -> list[list[int]]: return list(GROUPS.values())
