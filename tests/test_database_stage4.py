from exampapersorter.database import Database
from exampapersorter.schemas import CanonicalQuestion, DedupPairDecision, Question, QuestionEmbedding


def q(question_id, text, paper_id="p1", paper_year=None, qtype="short_answer"):
    return Question(
        question_id=question_id, paper_id=paper_id, section_id=None,
        source_filename="paper.pdf", source_file_hash="hash1", paper_year=paper_year,
        question_number="1", question_text=text, question_type=qtype,
        source_pages=[1], extraction_confidence=0.9,
    )


# --- question_embeddings ---


def test_get_question_embedding_returns_none_when_uncached(tmp_path):
    db = Database(tmp_path / "t.db")
    assert db.get_question_embedding("q1", "m", "v1") is None
    db.close()


def test_question_embedding_round_trips(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_question_embedding(QuestionEmbedding(question_id="q1", model="m", version="v1", content_hash="h1", vector=[0.1, 0.2, 0.3]))
    loaded = db.get_question_embedding("q1", "m", "v1")
    assert loaded.vector == [0.1, 0.2, 0.3]
    assert loaded.content_hash == "h1"
    db.close()


def test_question_embedding_is_scoped_to_model_and_version(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_question_embedding(QuestionEmbedding(question_id="q1", model="m", version="v1", content_hash="h1", vector=[1.0]))
    assert db.get_question_embedding("q1", "m", "v2") is None
    assert db.get_question_embedding("q1", "other-model", "v1") is None
    db.close()


def test_saving_embedding_again_overwrites_stale_content_hash(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_question_embedding(QuestionEmbedding(question_id="q1", model="m", version="v1", content_hash="h1", vector=[0.1]))
    db.save_question_embedding(QuestionEmbedding(question_id="q1", model="m", version="v1", content_hash="h2", vector=[0.9]))
    loaded = db.get_question_embedding("q1", "m", "v1")
    assert loaded.content_hash == "h2"
    assert loaded.vector == [0.9]
    db.close()


# --- dedup_pair_decisions ---


def test_get_dedup_pair_decision_returns_none_when_uncached(tmp_path):
    db = Database(tmp_path / "t.db")
    assert db.get_dedup_pair_decision("q1", "q2", "deterministic") is None
    db.close()


def test_exact_duplicate_decision_round_trips_under_deterministic_sentinel(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_dedup_pair_decision(
        DedupPairDecision(question_id_a="q1", question_id_b="q2", decision_source="exact_duplicate", verdict="same_question", confidence=1.0)
    )
    loaded = db.get_dedup_pair_decision("q1", "q2", "deterministic")
    assert loaded.decision_source == "exact_duplicate"
    assert loaded.verdict == "same_question"
    db.close()


def test_llm_judge_decision_round_trips_under_its_model_identifier(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_dedup_pair_decision(
        DedupPairDecision(question_id_a="q1", question_id_b="q2", decision_source="llm_judge", verdict="uncertain", confidence=0.4, reason="unclear"),
        model_used="openrouter-pool-v1",
    )
    loaded = db.get_dedup_pair_decision("q1", "q2", "openrouter-pool-v1")
    assert loaded.verdict == "uncertain"
    assert loaded.reason == "unclear"
    # not visible under a different model identifier -- a model/pool change invalidates the cache
    assert db.get_dedup_pair_decision("q1", "q2", "openrouter-pool-v2") is None
    db.close()


def test_pair_decisions_from_different_sources_do_not_collide(tmp_path):
    """An exact_duplicate decision (sentinel model) and an llm_judge
    decision (real model_used) for the SAME pair must be independently
    retrievable, never overwrite each other."""
    db = Database(tmp_path / "t.db")
    db.save_dedup_pair_decision(
        DedupPairDecision(question_id_a="q1", question_id_b="q2", decision_source="exact_duplicate", verdict="same_question", confidence=1.0)
    )
    db.save_dedup_pair_decision(
        DedupPairDecision(question_id_a="q1", question_id_b="q2", decision_source="llm_judge", verdict="different_question", confidence=0.7),
        model_used="openrouter-pool-v1",
    )
    assert db.get_dedup_pair_decision("q1", "q2", "deterministic").verdict == "same_question"
    assert db.get_dedup_pair_decision("q1", "q2", "openrouter-pool-v1").verdict == "different_question"
    db.close()


def test_list_dedup_pair_decisions_filters_by_verdict(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_dedup_pair_decision(DedupPairDecision(question_id_a="q1", question_id_b="q2", decision_source="llm_judge", verdict="uncertain", confidence=0.4), model_used="m")
    db.save_dedup_pair_decision(DedupPairDecision(question_id_a="q3", question_id_b="q4", decision_source="llm_judge", verdict="same_question", confidence=0.9), model_used="m")
    uncertain = db.list_dedup_pair_decisions(verdict="uncertain")
    assert len(uncertain) == 1
    assert uncertain[0].question_id_a == "q1"
    assert len(db.list_dedup_pair_decisions()) == 2
    db.close()


# --- canonical_questions / question_canonical_links ---


def test_replace_canonical_state_persists_and_hydrates_derived_fields(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_questions([q("q1", "Explain glycolysis.", paper_id="p1", paper_year="2022"), q("q2", "Explain glycolysis.", paper_id="p2", paper_year="2023")])
    cq = CanonicalQuestion(
        canonical_question_id="canonical_1", canonical_question_text="Explain glycolysis.",
        question_type="short_answer", representative_question_id="q1", dedup_confidence=1.0, dedup_status="exact_duplicate",
    )
    db.replace_canonical_state([cq], {"q1": "canonical_1", "q2": "canonical_1"})

    detail = db.get_canonical_question_detail("canonical_1")
    assert detail.occurrence_count == 2
    assert detail.unique_paper_count == 2
    assert detail.years == ["2022", "2023"]
    assert detail.source_question_ids == ["q1", "q2"]
    db.close()


def test_replace_canonical_state_is_a_full_replace_not_an_incremental_patch(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_questions([q("q1", "A"), q("q2", "B")])
    db.replace_canonical_state(
        [CanonicalQuestion(canonical_question_id="c1", canonical_question_text="A", question_type="short_answer", representative_question_id="q1", dedup_status="singleton")],
        {"q1": "c1"},
    )
    assert db.get_canonical_question_id_for_question("q1") == "c1"

    # a fresh run over a different grouping fully supersedes the old state
    db.replace_canonical_state(
        [CanonicalQuestion(canonical_question_id="c2", canonical_question_text="B", question_type="short_answer", representative_question_id="q2", dedup_status="singleton")],
        {"q2": "c2"},
    )
    assert db.get_canonical_question_id_for_question("q1") is None  # old link gone
    assert db.get_canonical_question_id_for_question("q2") == "c2"
    assert len(db.list_canonical_questions()) == 1
    db.close()


def test_canonical_state_survives_reopening_the_database(tmp_path):
    """Persistence round trip: canonical groups and links must be readable
    from a freshly-opened Database pointed at the same file, not just the
    in-memory connection that wrote them."""
    db_path = tmp_path / "t.db"
    db = Database(db_path)
    db.save_questions([q("q1", "Explain glycolysis.", paper_id="p1", paper_year="2022"), q("q2", "Explain glycolysis.", paper_id="p2", paper_year="2023")])
    db.replace_canonical_state(
        [CanonicalQuestion(canonical_question_id="c1", canonical_question_text="Explain glycolysis.", question_type="short_answer", representative_question_id="q1", dedup_status="exact_duplicate", dedup_confidence=1.0)],
        {"q1": "c1", "q2": "c1"},
    )
    db.close()

    reopened = Database(db_path)
    detail = reopened.get_canonical_question_detail("c1")
    assert detail is not None
    assert detail.occurrence_count == 2
    assert reopened.get_canonical_question_id_for_question("q2") == "c1"
    reopened.close()


def test_get_all_questions_spans_every_paper(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_questions([q("q1", "A", paper_id="p1"), q("q2", "B", paper_id="p2")])
    all_questions = db.get_all_questions()
    assert {qq.question_id for qq in all_questions} == {"q1", "q2"}
    db.close()
