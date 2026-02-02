"""
Orchestration: rollback to previous model version.
Switches the 'current' symlink to the previous v_YYYYMMDD directory.
Run after a failed deployment or bad model.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CURRENT_LINK = MODELS_DIR / "current"


def main():
    if not CURRENT_LINK.exists():
        print("No 'current' link found. Nothing to rollback.")
        return
    versions = sorted([d.name for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("v_")])
    if len(versions) < 2:
        print("Only one version exists. Cannot rollback.")
        return
    current_target = CURRENT_LINK.resolve().name
    idx = next((i for i, v in enumerate(versions) if v == current_target), None)
    if idx is None or idx == 0:
        print("Current is already the oldest version or not in list. Cannot rollback.")
        return
    prev = versions[idx - 1]
    CURRENT_LINK.unlink()
    CURRENT_LINK.symlink_to(prev)
    print(f"Rolled back: current -> {prev}")


if __name__ == "__main__":
    main()
