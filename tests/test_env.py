import json
import shutil

from env.create import CONFIGS, create_config
from env.extract import extract_env


def test_extract_env_keys():
    e = extract_env()
    assert e["device"] in ("cuda", "cpu")
    assert "torch" in e


def test_create_config_sections_and_idempotent():
    name = "_pytest_tmp"
    cfg_dir = CONFIGS / name
    shutil.rmtree(cfg_dir, ignore_errors=True)
    try:
        out = create_config(3, 4, name=name)
        assert out == cfg_dir / "config.json"
        cfg = json.loads(out.read_text())
        assert set(cfg) >= {"task", "env", "benchmark", "correctness", "perf"}
        assert cfg["task"] == {"level": 3, "problem_id": 4, "name": "4_LeNet5"}
        assert cfg["perf"]["min_speedup_vs_compile"] == 1.05

        # idempotent: second call keeps the existing file untouched
        before = out.read_text()
        out2 = create_config(3, 4, name=name)
        assert out2 == out and out.read_text() == before
    finally:
        shutil.rmtree(cfg_dir, ignore_errors=True)
