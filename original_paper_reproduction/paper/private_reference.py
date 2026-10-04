"""Optional read-only archive used only for comparisons."""
from pathlib import Path
import os
REFERENCE = Path(os.environ.get("ICTC_PRIVATE_REFERENCE", str(Path(__file__).resolve().parent / "reference"))).resolve()
