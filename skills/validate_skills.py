"""Validate every skills/<category>/<folder>/skill.json against
skills/schema/skill.schema.json plus the cross-field invariants the
schema cannot express (id == '<category>.<folder>', entry points at the
sibling SKILL.md, category == parent dir). Importable by tests and
runnable standalone: python3 skills/validate_skills.py
"""

import json
import sys
from pathlib import Path

import jsonschema

SKILLS = Path(__file__).resolve().parent
SCHEMA = SKILLS / "schema" / "skill.schema.json"


def validate() -> list[str]:
    schema = json.loads(SCHEMA.read_text())
    errors: list[str] = []
    metas = sorted(SKILLS.glob("*/*/skill.json"))
    if not metas:
        return ["no skill.json found under skills/*/*/"]
    for meta in metas:
        rel = meta.relative_to(SKILLS)
        category, folder = rel.parts[0], rel.parts[1]
        try:
            data = json.loads(meta.read_text())
        except json.JSONDecodeError as e:
            errors.append(f"{rel}: invalid JSON ({e})")
            continue
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            errors.append(f"{rel}: schema: {e.message}")
            continue
        if data["id"] != f"{category}.{folder}":
            errors.append(f"{rel}: id {data['id']!r} != '{category}.{folder}'")
        if data["category"] != category:
            errors.append(f"{rel}: category {data['category']!r} != dir {category!r}")
        expected_entry = f"skills/{category}/{folder}/SKILL.md"
        if data["entry"] != expected_entry:
            errors.append(f"{rel}: entry {data['entry']!r} != {expected_entry!r}")
        if not (meta.parent / "SKILL.md").is_file():
            errors.append(f"{rel}: sibling SKILL.md missing")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print(f"OK: all skill.json valid ({len(list(SKILLS.glob('*/*/skill.json')))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
