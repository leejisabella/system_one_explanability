import os

from sysone.env import load_env


def test_env_file_fills_only_unset_nonblank_variables(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text('# comment\nSYSONE_A=from-file\nSYSONE_B="quoted"\nSYSONE_BLANK=\nSYSONE_SHELL=from-file\nnot a line\n')
    for key in ("SYSONE_A", "SYSONE_B", "SYSONE_BLANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SYSONE_SHELL", "from-shell")
    load_env(path)
    assert os.environ["SYSONE_A"] == "from-file" and os.environ["SYSONE_B"] == "quoted"
    assert "SYSONE_BLANK" not in os.environ
    assert os.environ["SYSONE_SHELL"] == "from-shell"
    for key in ("SYSONE_A", "SYSONE_B"):
        monkeypatch.delenv(key)


def test_missing_env_file_is_fine(tmp_path):
    load_env(tmp_path / "absent")
