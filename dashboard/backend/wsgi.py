from __future__ import annotations

from dashboard.backend.app import create_app

# WSGI entrypoint for Gunicorn
app = create_app()
