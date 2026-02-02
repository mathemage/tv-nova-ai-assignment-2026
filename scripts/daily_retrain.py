"""
Orchestration: daily retrain of Task 2 model.
1) Fetches new dataset URL (from env or mock), downloads data.
2) Runs Task 2 training pipeline.
3) Saves model under models/v_YYYYMMDD/.
4) Keeps last 7 versions; sets current symlink to latest.
"""
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MAX_VERSIONS = 7
CURRENT_LINK = MODELS_DIR / "current"


def get_data_url() -> str:
    """Get CSV URL from env or use mock (synthetic)."""
    url = os.environ.get("NOVATV_DATA_URL")
    if url:
        return url
    # Mock: use local synthetic or existing data
    if (DATA_DIR / "data.csv").exists():
        return str(DATA_DIR / "data.csv")
    return "synthetic"


def download_data(url: str) -> Path:
    if url == "synthetic":
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_synthetic_data.py")], check=True, cwd=str(ROOT))
        return DATA_DIR / "data_synthetic.csv"
    if url.startswith("http"):
        from src.data import download_data as dl
        path = dl(save_dir=str(DATA_DIR), filename="data.csv")
        return Path(path)
    return Path(url)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = datetime.utcnow().strftime("%Y%m%d")
    out_subdir = MODELS_DIR / f"v_{version}"
    if out_subdir.exists():
        shutil.rmtree(out_subdir)
    out_subdir.mkdir(parents=True)

    url = get_data_url()
    print(f"Data source: {url}")
    data_path = download_data(url)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    epochs = os.environ.get("TRAIN_EPOCHS", "50")
    patience = os.environ.get("TRAIN_PATIENCE", "8")
    cmd = [
        sys.executable, "-m", "src.train_task2",
        "--data_path", str(data_path),
        "--model", "mlp_large",
        "--out_dir", str(out_subdir),
        "--epochs", epochs,
        "--patience", patience,
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    # Copy artifact names to current (task2_best.pt etc. are in out_subdir)
    # Service reads from MODEL_DIR; we set current -> v_YYYYMMDD
    if CURRENT_LINK.exists():
        CURRENT_LINK.unlink()
    CURRENT_LINK.symlink_to(out_subdir.name)

    # Prune old versions (keep last MAX_VERSIONS)
    versions = sorted([d.name for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("v_")])
    for name in versions[:-MAX_VERSIONS]:
        old = MODELS_DIR / name
        if old.is_dir() and not old.is_symlink():
            shutil.rmtree(old, ignore_errors=True)
    print(f"Saved model to {out_subdir}, current -> {out_subdir.name}")


if __name__ == "__main__":
    main()
