"""Environment-backed application secrets and notification settings.

Set these values in the process environment before launching Digital Eye.
No credentials are stored in source control.
"""
import os
from pathlib import Path


def _load_local_env():
    """Load simple KEY=VALUE entries from a project-local .env if present.

    Existing process environment variables always take precedence.
    """
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()

DASHBOARD_PASSWORD = os.getenv("DIGITAL_EYE_PASSWORD", "")
EMAIL_FROM = os.getenv("DIGITAL_EYE_EMAIL_FROM", "")
EMAIL_APP_PASSWORD = os.getenv("DIGITAL_EYE_EMAIL_APP_PASSWORD", "")
EMAIL_TO = os.getenv("DIGITAL_EYE_EMAIL_TO", "")


def email_is_configured():
    return all((EMAIL_FROM, EMAIL_APP_PASSWORD, EMAIL_TO))
