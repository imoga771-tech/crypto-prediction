"""
Flask application entrypoint.

This file now delegates to the application factory in crypto_analysis.__init__.py
and keeps a simple runnable script interface for local development.

Run options:
- python -m flask --app crypto_analysis run
- python -m crypto_analysis.app
- python crypto_analysis/app.py  (works when executed from project root)
"""

import os

from crypto_analysis import create_app

app = create_app()


if __name__ == "__main__":
    debug_env = os.getenv("FLASK_DEBUG", "1")
    debug = True if str(debug_env).lower() in ("1", "true", "yes", "y") else False
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))

    app.run(debug=debug, host=host, port=port)