"""Stage 4 embedding cache tests. The real fastembed model is never loaded
here -- embeddings._get_model is monkeypatched to a small deterministic
fake, so these tests are fast, offline, and exercise the caching logic
(which is what's actually under test), not embedding quality (see
test_deduplication_candidates.py / the pipeline smoke test for similarity
behavior against the real model)."""
from __future__ import annotations

from dataclasses import replace

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.database import Database
from exampapersorter.deduplication import embeddings
from exampapersorter.schemas import Question


def q(question_id, text, qtype="short_answer"):
    return Question(
        question_id=question_id, paper_id="p1", section_id=None,
        source_filename="paper.pdf", source_file_hash="hash1",
        question_number="1", question_text=text, question_type=qtype,
        source_pages=[1], extraction_confidence=0.9,
    )


class FakeModel:
    """Deterministic stand-in for fastembed.TextEmbedding: encodes each
    text as [len(text)] so distinct texts get distinct (but reproducible)
    vectors, with no ML/network involved."""

    def __init__(self):
        self.calls = 0
        self.last_batch = None

    def embed(self, texts):
        self.calls += 1
        self.last_batch = list(texts)
        return [[float(len(t))] for t in texts]


def install_fake_model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(embeddings, "_get_model", lambda config: fake)
    return fake


def test_computes_and_caches_embeddings_for_uncached_questions(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    fake = install_fake_model(monkeypatch)
    result = embeddings.get_or_compute_embeddings([q("q1", "Explain glycolysis.")], DEFAULT_CONFIG, db)
    assert "q1" in result
    assert fake.calls == 1
    cached = db.get_question_embedding("q1", DEFAULT_CONFIG.embedding_model, DEFAULT_CONFIG.embedding_model_version)
    assert cached is not None
    assert cached.vector == result["q1"].vector
    db.close()


def test_rerun_reuses_cached_embedding_without_recomputing(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    fake = install_fake_model(monkeypatch)
    questions = [q("q1", "Explain glycolysis."), q("q2", "What is glycogen?")]

    embeddings.get_or_compute_embeddings(questions, DEFAULT_CONFIG, db)
    assert fake.calls == 1

    embeddings.get_or_compute_embeddings(questions, DEFAULT_CONFIG, db)
    assert fake.calls == 1  # second run: nothing new needed to be computed
    db.close()


def test_only_uncached_questions_are_sent_to_the_model(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    fake = install_fake_model(monkeypatch)
    embeddings.get_or_compute_embeddings([q("q1", "Explain glycolysis.")], DEFAULT_CONFIG, db)
    assert fake.calls == 1

    # q1 already cached; q2 is new -- only q2 should be sent to the model.
    embeddings.get_or_compute_embeddings(
        [q("q1", "Explain glycolysis."), q("q2", "What is glycogen?")], DEFAULT_CONFIG, db
    )
    assert fake.calls == 2
    assert fake.last_batch == ["[short_answer] What is glycogen?"]
    db.close()


def test_changed_question_text_invalidates_the_cached_embedding(tmp_path, monkeypatch):
    """Simulates a genuine Stage 2 re-extraction changing a question's text
    under the same question_id -- content_hash mismatch must force
    recomputation, never silently reuse a stale vector."""
    db = Database(tmp_path / "t.db")
    fake = install_fake_model(monkeypatch)
    embeddings.get_or_compute_embeddings([q("q1", "Explain glycolysis.")], DEFAULT_CONFIG, db)
    assert fake.calls == 1

    embeddings.get_or_compute_embeddings([q("q1", "Explain glycolysis in detail.")], DEFAULT_CONFIG, db)
    assert fake.calls == 2  # text changed -> recomputed, not reused
    db.close()


def test_embedding_model_version_bump_invalidates_the_cache(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    fake = install_fake_model(monkeypatch)
    embeddings.get_or_compute_embeddings([q("q1", "Explain glycolysis.")], DEFAULT_CONFIG, db)
    assert fake.calls == 1

    bumped_config = replace(DEFAULT_CONFIG, embedding_model_version="v2")
    embeddings.get_or_compute_embeddings([q("q1", "Explain glycolysis.")], bumped_config, db)
    assert fake.calls == 2  # different version -> cache miss, recomputed

    # both versions' vectors remain independently retrievable
    assert db.get_question_embedding("q1", DEFAULT_CONFIG.embedding_model, "v1") is not None
    assert db.get_question_embedding("q1", DEFAULT_CONFIG.embedding_model, "v2") is not None
    db.close()


def test_embedding_model_name_change_invalidates_the_cache(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    fake = install_fake_model(monkeypatch)
    embeddings.get_or_compute_embeddings([q("q1", "Explain glycolysis.")], DEFAULT_CONFIG, db)
    assert fake.calls == 1

    different_model_config = replace(DEFAULT_CONFIG, embedding_model="some/other-model")
    embeddings.get_or_compute_embeddings([q("q1", "Explain glycolysis.")], different_model_config, db)
    assert fake.calls == 2
    db.close()


def test_resume_after_partial_batch_only_computes_the_remainder(tmp_path, monkeypatch):
    """Simulates a crash after embedding some, but not all, questions in a
    batch -- a fresh run over the full question list must only compute the
    ones that never got persisted."""
    db = Database(tmp_path / "t.db")
    fake = install_fake_model(monkeypatch)
    partial_batch = [q("q1", "Explain glycolysis."), q("q2", "What is glycogen?")]
    embeddings.get_or_compute_embeddings(partial_batch, DEFAULT_CONFIG, db)  # "crash" happens after this
    assert fake.calls == 1

    full_batch = partial_batch + [q("q3", "Explain vitamin D deficiency.")]
    result = embeddings.get_or_compute_embeddings(full_batch, DEFAULT_CONFIG, db)
    assert fake.last_batch == ["[short_answer] Explain vitamin D deficiency."]
    assert set(result) == {"q1", "q2", "q3"}
    db.close()
