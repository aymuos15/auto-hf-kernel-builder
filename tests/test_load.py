import pytest

from task.load import Task, load_task


def test_load_task_basic():
    t = load_task(3, 4)
    assert isinstance(t, Task)
    assert (t.level, t.problem_id, t.name) == (3, 4, "4_LeNet5")
    assert "class Model" in t.code


def test_unknown_problem_id():
    with pytest.raises(KeyError):
        load_task(3, 99999)


def test_unknown_level():
    with pytest.raises(FileNotFoundError):
        load_task(99, 1)
