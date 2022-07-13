import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
def test_run_summarization():
    """
    A simple tests that fails if the run_summarization.py script returns non-zero exit code.
    """
    cwd = Path(__file__).parent
    script_filepath = cwd / ".." / ".." / "scripts" / "run_summarization.py"
    config_filepath = cwd / ".." / ".." / "test_fixtures" / "conf" / "summarization.json"
    _ = subprocess.run(
        [
            "python",
            script_filepath,
            config_filepath,
        ],
        capture_output=True,
        check=True,
    )
