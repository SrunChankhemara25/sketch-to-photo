"""Project entry point. Run with `python app.py` or `streamlit run app.py`."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit = get_script_run_ctx(suppress_warning=True) is not None
    except ImportError:
        in_streamlit = False

    if not in_streamlit:
        os.execv(
            sys.executable,
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]],
        )

    from src.ui import render_app
    render_app()


if __name__ == "__main__":
    main()
