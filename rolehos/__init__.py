"""RoleHOS reproducibility entry points. Vendored research model is isolated here."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
vendor = str(ROOT / "third_party")
if "mmseg" in sys.modules and not str(
    getattr(sys.modules["mmseg"], "__file__", "")
).startswith(vendor):
    raise RuntimeError(
        "Import rolehos before any other mmseg installation in this process."
    )
sys.path.insert(0, vendor)
__version__ = "0.1.0"
