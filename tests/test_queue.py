import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worker.queue import Queue  # noqa: E402


def test_enqueue_claim_complete(tmp_path):
    q = Queue(str(tmp_path / "q.db"))
    jid = q.enqueue("/some/configs/X/config.json")
    assert q.stats() == {"pending": 1}

    job = q.claim("w1")
    assert job is not None and job["id"] == jid and job["state"] == "pending"
    assert q.stats() == {"leased": 1}
    assert q.claim("w2") is None  # nothing else to take

    q.complete(jid, "w1", {"passed": True, "error_class": None})
    assert q.stats() == {"done": 1}
    done = q.get(jid)
    assert done is not None and done["verdict"] is not None


def test_fifo_order(tmp_path):
    q = Queue(str(tmp_path / "q.db"))
    a = q.enqueue("a")
    b = q.enqueue("b")
    j1 = q.claim("w")
    j2 = q.claim("w")
    assert j1 is not None and j1["id"] == a
    assert j2 is not None and j2["id"] == b


def test_lease_expiry_reclaims_crashed_job(tmp_path):
    q = Queue(str(tmp_path / "q.db"))
    jid = q.enqueue("c")
    q.claim("dead", lease_secs=0.05)  # worker takes it then "crashes"
    assert q.claim("alive") is None  # lease still valid
    time.sleep(0.06)
    again = q.claim("alive")  # expired lease -> reclaimable
    assert again is not None and again["id"] == jid


def test_completed_job_not_reclaimed(tmp_path):
    q = Queue(str(tmp_path / "q.db"))
    jid = q.enqueue("d")
    q.claim("w", lease_secs=0.01)
    q.complete(jid, "w", {"passed": True})
    time.sleep(0.02)
    assert q.claim("w2") is None  # done jobs never re-leased
