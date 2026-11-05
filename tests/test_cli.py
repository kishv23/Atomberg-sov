import json
import shutil
from pathlib import Path

from sov.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_analyze_command_on_sample_data(tmp_path, monkeypatch, capsys):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    shutil.copy(ROOT / "config.yaml", tmp_path / "config.yaml")
    monkeypatch.chdir(tmp_path)
    text = (tmp_path / "config.yaml").read_text().replace('backend: "auto"', 'backend: "lexicon"')
    (tmp_path / "config.yaml").write_text(text)

    assert main(["analyze"]) == 0
    out = capsys.readouterr().out
    assert "Atomberg" in out and "mention_share" in out
    insights = json.loads((tmp_path / "out" / "insights.json").read_text())
    assert insights and all({"kind", "text"} <= set(i) for i in insights)
