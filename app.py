"""Streamlit entry point.

    streamlit run app.py

Streamlit needs a script path, so this file exists to give it one. The UI itself lives in
`src/interview_ai/ui/app.py` with the rest of the package.
"""
from __future__ import annotations

from interview_ai.ui.app import main

main()
