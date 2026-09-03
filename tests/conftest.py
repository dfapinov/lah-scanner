import os
import sys
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESS_DIR = PROJECT_ROOT / "src" / "process"
sys.path.insert(0, str(PROCESS_DIR))

