from __future__ import annotations

import os


from experience_bench.core.dotenv import load_dotenv_if_present


def test_load_dotenv_if_present_loads_from_cwd(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from_env_file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    loaded = load_dotenv_if_present()

    assert loaded is not None
    assert os.environ.get("OPENROUTER_API_KEY") == "from_env_file"


def test_load_dotenv_if_present_does_not_override_existing_env(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from_env_file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "from_shell")

    load_dotenv_if_present()

    assert os.environ.get("OPENROUTER_API_KEY") == "from_shell"
