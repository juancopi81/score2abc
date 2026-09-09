"""Exercise playback timing against the same optional local renderer as the UI."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_playback_timeline():
    node = shutil.which("node")
    if not node:
        pytest.skip("Optional Node.js is unavailable")
    env = os.environ.copy()
    renderer = shutil.which("abc2svg")
    if renderer:
        env["SCORE2ABC_TEST_RENDERER"] = str(Path(renderer).resolve().parent)
    result = subprocess.run(
        [
            node,
            "--test",
            str(Path(__file__).with_name("review_playback.test.js")),
            str(Path(__file__).with_name("review_audio.test.js")),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
