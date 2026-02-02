"""Test daily retrain script with synthetic data and minimal epochs."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_daily_retrain_smoke():
    """Run daily_retrain with synthetic data; should complete without error."""
    # Ensure synthetic data exists
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_synthetic_data.py")],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "daily_retrain.py"),
        ],
        cwd=str(ROOT),
        env={"NOVATV_DATA_URL": "", "TRAIN_EPOCHS": "2", "TRAIN_PATIENCE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert (ROOT / "models").exists()
    versions = list((ROOT / "models").glob("v_*"))
    assert len(versions) >= 1
    assert (versions[0] / "task2_best.pt").exists()
