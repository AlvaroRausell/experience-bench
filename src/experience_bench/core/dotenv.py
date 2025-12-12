from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_dotenv_if_present() -> Optional[Path]:
    """Load environment variables from a `.env` file if one can be found.

    - Searches from the current working directory upwards.
    - Does not override existing environment variables.

    Returns the resolved path to the `.env` file loaded, or None.
    """

    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore
    except Exception:
        return None

    env_path = find_dotenv(filename=".env", usecwd=True)
    if not env_path:
        return None

    p = Path(env_path).resolve()
    if not p.exists():
        return None

    load_dotenv(dotenv_path=str(p), override=False)
    return p
