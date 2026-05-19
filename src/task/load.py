"""Pull one KernelBench task by (level, problem_id).

The parquet files in data/ are the source of truth. A task is a single
row: the reference Model code plus its input generators.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Task:
    level: int
    problem_id: int
    name: str
    code: str


def load_task(level: int, problem_id: int) -> Task:
    parquet = DATA / f"level_{level}.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(f"dataset missing: {parquet}")
    df = pd.read_parquet(parquet)
    rows = df[df.problem_id == problem_id]
    if rows.empty:
        raise KeyError(f"no problem_id={problem_id} in level_{level}")
    row = rows.iloc[0]
    return Task(
        level=level, problem_id=int(row["problem_id"]), name=str(row["name"]), code=str(row["code"])
    )
