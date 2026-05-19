import importlib.util
from pathlib import Path

_v = Path(__file__).resolve().parents[1] / "skills" / "validate_skills.py"
_spec = importlib.util.spec_from_file_location("validate_skills", _v)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_all_skill_metadata_valid():
    errors = _mod.validate()
    assert not errors, "skill.json validation failed:\n" + "\n".join(errors)
