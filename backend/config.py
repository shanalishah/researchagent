"""Backend configuration: load `backend/.env` into the process environment.

Kept dependency-free (no python-dotenv) so importing the backend never
requires extra packages. The extracted pipeline reads everything via
os.getenv, so loading these into os.environ is all that's needed.
"""

import os
import pathlib

_ENV_PATH = pathlib.Path(__file__).resolve().parent / ".env"


def load_env() -> bool:
    """Load KEY=VALUE pairs from backend/.env into os.environ.

    Existing environment variables win (so real env / Cloud Run config is not
    overridden by a stray local .env). Returns True if the file was found.
    """
    if not _ENV_PATH.is_file():
        return False
    for raw in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def has_r2_credentials() -> bool:
    return all(os.getenv(k) for k in
               ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET"))
