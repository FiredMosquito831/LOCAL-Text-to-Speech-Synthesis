"""Batch queue behaviour, including the dataclass regression fixed in 0.1.0."""

from __future__ import annotations

import numpy as np

from core.batch_queue import BatchJob, BatchQueue, JobPriority, JobStatus


class FakeManager:
    def __init__(self):
        self.adapters = []
        self.calls = []

    def set_adapter(self, adapter):
        self.adapters.append(adapter)

    def synthesize(self, text, voice=None, progress_callback=None, **settings):
        self.calls.append({"text": text, "voice": voice, "settings": settings})
        if progress_callback:
            progress_callback(50, "halfway")
            progress_callback(100, "done")
        return np.zeros(2400, dtype=np.float32), 24000


def test_batch_job_can_be_constructed_with_defaults():
    job = BatchJob(text="hello")
    assert job.id
    assert job.status is JobStatus.PENDING
    assert job.priority is JobPriority.NORMAL
    assert job.settings == {}


def test_add_job_preserves_all_fields():
    queue = BatchQueue()
    job_id = queue.add_job(
        "hello",
        voice="ref.wav",
        adapter="test-adapter",
        output_path="out.wav",
        priority=JobPriority.HIGH,
        nfe_steps=8,
    )
    job = queue.get_job(job_id)
    assert job is not None
    assert job.text == "hello"
    assert job.voice == "ref.wav"
    assert job.adapter == "test-adapter"
    assert job.output_path == "out.wav"
    assert job.priority is JobPriority.HIGH
    assert job.settings == {"nfe_steps": 8}
    assert queue.get_pending_count() == 1


def test_add_batch_assigns_distinct_output_paths(tmp_path):
    queue = BatchQueue()
    ids = queue.add_batch(["one", "two"], output_dir=str(tmp_path / "out"))
    assert len(ids) == 2
    assert len(set(ids)) == 2
    jobs = queue.get_all_jobs()
    outputs = sorted(job.output_path for job in jobs)
    assert outputs == [
        str(tmp_path / "out" / "output_0000.wav"),
        str(tmp_path / "out" / "output_0001.wav"),
    ]


def test_process_job_runs_manager_and_saves_audio(tmp_path):
    manager = FakeManager()
    queue = BatchQueue(inference_manager=manager)
    job_id = queue.add_job(
        "hello",
        voice="ref.wav",
        adapter="adapter.pt",
        output_path=str(tmp_path / "job.wav"),
    )
    job = queue.get_job(job_id)
    queue._process_job(job)

    assert job.status is JobStatus.COMPLETED
    assert job.progress == 100
    assert job.duration_s == 0.1
    assert (tmp_path / "job.wav").exists()
    assert manager.adapters == ["adapter.pt"]
    assert manager.calls[0]["voice"] == "ref.wav"


def test_process_job_without_manager_fails_instead_of_silently_completing():
    queue = BatchQueue()
    job = BatchJob(text="hello")
    queue._process_job(job)
    assert job.status is JobStatus.FAILED
    assert "inference manager" in (job.error or "").lower()


def test_cancel_and_statistics():
    queue = BatchQueue()
    first = queue.add_job("a")
    queue.add_job("b")
    assert queue.cancel_job(first)
    stats = queue.get_statistics()
    assert stats.cancelled_jobs == 1
    assert stats.total_jobs == 2
