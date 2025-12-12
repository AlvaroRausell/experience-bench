from __future__ import annotations

import re


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_first_code_block(text: str) -> str | None:
    m = _CODE_BLOCK_RE.search(text or "")
    if not m:
        return None
    code = m.group(1)
    return code.strip() + "\n"
