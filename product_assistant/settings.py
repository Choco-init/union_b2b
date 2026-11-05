"""Read credentials without changing the user's environment or logging values."""
import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def api_key() -> str | None:
    aliases = ('OPENAI_API_KEY', 'OPENAI_API', 'OPEN_AI')
    environment = {key.upper(): value for key, value in os.environ.items() if key.upper() in aliases}
    # interpolate=False prevents unrelated environment variables entering a credential.
    file_values = {key.upper(): value for key, value in
                   dotenv_values(ROOT / '.env', encoding='utf-8-sig', interpolate=False).items()
                   if key.upper() in aliases}
    for values in (environment, file_values):
        for name in aliases:
            if value := values.get(name):
                return value.strip()
    return None
