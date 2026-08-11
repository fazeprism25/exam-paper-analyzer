from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.schemas import Topic, TopicExtractionResult
from exampapersorter.validation import looks_letter_spaced, reconcile_resolution_status, validate_topics


def make_result(topics: list[Topic]) -> TopicExtractionResult:
    return TopicExtractionResult(topics=topics)


def test_clean_topic_list_passes():
    topics = [
        Topic(id="ch1", name="Carbohydrate Metabolism", level=1, parent_id=None, source_pages=[10]),
        Topic(id="ch1_1", name="Glycolysis", level=2, parent_id="ch1", source_pages=[12]),
        Topic(id="ch2", name="Lipid Metabolism", level=1, parent_id=None, source_pages=[40]),
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert report.passed
    assert report.topic_count == 3
    assert report.issues == []


def test_duplicate_topic_id_is_error():
    topics = [
        Topic(id="ch1", name="A", level=1, parent_id=None, source_pages=[1]),
        Topic(id="ch1", name="B", level=1, parent_id=None, source_pages=[2]),
        Topic(id="ch3", name="C", level=1, parent_id=None, source_pages=[3]),
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert not report.passed
    assert any(i.code == "duplicate_topic_id" for i in report.issues)


def test_duplicate_topic_name_is_warning_not_blocking():
    topics = [
        Topic(id="ch1", name="Overview", level=1, parent_id=None, source_pages=[1]),
        Topic(id="ch2", name="overview", level=1, parent_id=None, source_pages=[2]),
        Topic(id="ch3", name="C", level=1, parent_id=None, source_pages=[3]),
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert report.passed  # warnings alone don't fail validation
    assert any(i.code == "duplicate_topic_name" and i.severity == "warning" for i in report.issues)


def test_empty_name_is_error():
    topics = [
        Topic(id="ch1", name="   ", level=1, parent_id=None, source_pages=[1]),
        Topic(id="ch2", name="B", level=1, parent_id=None, source_pages=[2]),
        Topic(id="ch3", name="C", level=1, parent_id=None, source_pages=[3]),
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert not report.passed
    assert any(i.code == "empty_name" for i in report.issues)


def test_unknown_parent_reference_is_error():
    topics = [
        Topic(id="ch1", name="A", level=1, parent_id=None, source_pages=[1]),
        Topic(id="ch1_1", name="B", level=2, parent_id="does_not_exist", source_pages=[2]),
        Topic(id="ch2", name="C", level=1, parent_id=None, source_pages=[3]),
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert not report.passed
    assert any(i.code == "unknown_parent" for i in report.issues)


def test_self_parent_is_error():
    topics = [
        Topic(id="ch1", name="A", level=1, parent_id="ch1", source_pages=[1]),
        Topic(id="ch2", name="B", level=1, parent_id=None, source_pages=[2]),
        Topic(id="ch3", name="C", level=1, parent_id=None, source_pages=[3]),
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert not report.passed
    assert any(i.code == "self_parent" for i in report.issues)


def test_parent_cycle_is_error():
    topics = [
        Topic(id="a", name="A", level=1, parent_id="b", source_pages=[1]),
        Topic(id="b", name="B", level=1, parent_id="c", source_pages=[2]),
        Topic(id="c", name="C", level=1, parent_id="a", source_pages=[3]),
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert not report.passed
    assert any(i.code == "parent_cycle" for i in report.issues)


def test_nested_level_without_parent_id_is_warning():
    topics = [
        Topic(id="ch1", name="Chapter 1", level=1, parent_id=None, source_pages=[10]),
        Topic(id="ch1_1", name="Subtopic", level=2, parent_id=None, source_pages=[10]),  # level implies nesting, no link
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert report.passed
    issue = next(i for i in report.issues if i.code == "missing_parent_for_nested_level")
    assert issue.topic_id == "ch1_1"


def test_suspiciously_small_topic_list_is_warning():
    topics = [Topic(id="ch1", name="A", level=1, parent_id=None, source_pages=[1])]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert report.passed
    assert any(i.code == "suspiciously_small_topic_list" for i in report.issues)


def test_resolved_topic_with_numeric_name_is_inconsistent_error():
    # resolution_status defaults to "resolved" -- claiming resolved with a
    # numeric-only name means the self-report doesn't match its own content.
    topics = [
        Topic(id="ch1", name="1", level=1, parent_id=None, source_pages=[10]),
        Topic(id="ch2", name="Lipid Metabolism", level=1, parent_id=None, source_pages=[25]),
        Topic(id="ch3", name="12.", level=1, parent_id=None, source_pages=[30]),
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert not report.passed
    flagged = {i.topic_id for i in report.issues if i.code == "inconsistent_resolution_status"}
    assert flagged == {"ch1", "ch3"}


def test_unresolved_topic_is_warning_not_blocking():
    topics = [
        Topic(id="ch1", name="12", level=1, parent_id=None, source_pages=[30], resolution_status="name_unresolved"),
        Topic(id="ch2", name="Lipid Metabolism", level=1, parent_id=None, source_pages=[25]),
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert report.passed  # a review warning, not a blocking error
    issue = next(i for i in report.issues if i.code == "unresolved_topic")
    assert issue.topic_id == "ch1"


def test_source_page_outside_evidence_range_is_warning():
    topics = [
        Topic(id="ch1", name="A", level=1, parent_id=None, source_pages=[12]),
        Topic(id="ch2", name="B", level=1, parent_id=None, source_pages=[999]),  # never shown to the LLM
        Topic(id="ch3", name="C", level=1, parent_id=None, source_pages=[20]),
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG, valid_page_range=(10, 25))
    assert report.passed  # a grounding warning, not a blocking error
    issue = next(i for i in report.issues if i.code == "source_page_out_of_range")
    assert issue.topic_id == "ch2"


def test_looks_letter_spaced_detects_garbled_ocr_text():
    assert looks_letter_spaced("B i o m o l ecu l es and t he ce ll")
    assert looks_letter_spaced("M e t abo li s m o f ca r bohyd r a t es")


def test_looks_letter_spaced_accepts_normal_titles():
    assert not looks_letter_spaced("Biomolecules and the Cell")
    assert not looks_letter_spaced("Metabolism of Carbohydrates")
    assert not looks_letter_spaced("DNA Replication and Repair")


def test_looks_letter_spaced_ignores_short_names():
    assert not looks_letter_spaced("Enzymes")
    assert not looks_letter_spaced("Vitamins")


def test_reconcile_downgrades_resolved_garbled_names():
    topics = [
        Topic(id="ch1", name="B i o m o l ecu l es and t he ce ll", level=1, parent_id=None, source_pages=[10]),
        Topic(id="ch2", name="Lipid Metabolism", level=1, parent_id=None, source_pages=[10]),
    ]
    result = reconcile_resolution_status(topics)
    garbled = next(t for t in result if t.id == "ch1")
    clean = next(t for t in result if t.id == "ch2")
    assert garbled.resolution_status == "name_unresolved"
    assert garbled.name_evidence_source is None
    assert clean.resolution_status == "resolved"


def test_resolved_topic_with_garbled_name_is_inconsistent_error():
    topics = [
        Topic(id="ch1", name="B i o m o l ecu l es and t he ce ll", level=1, parent_id=None, source_pages=[10]),
        Topic(id="ch2", name="Lipid Metabolism", level=1, parent_id=None, source_pages=[10]),
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert not report.passed
    flagged = {i.topic_id for i in report.issues if i.code == "inconsistent_resolution_status"}
    assert flagged == {"ch1"}


def test_root_topic_with_parent_is_error():
    topics = [
        Topic(id="a", name="A", level=1, parent_id=None, source_pages=[1]),
        Topic(id="b", name="B", level=1, parent_id="a", source_pages=[2]),  # level=1 but claims a parent
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert not report.passed
    flagged = {i.topic_id for i in report.issues if i.code == "root_has_parent"}
    assert flagged == {"b"}


def test_child_level_not_parent_level_plus_one_is_error():
    topics = [
        Topic(id="a", name="A", level=1, parent_id=None, source_pages=[1]),
        Topic(id="b", name="B", level=3, parent_id="a", source_pages=[2]),  # skips level 2
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert not report.passed
    flagged = {i.topic_id for i in report.issues if i.code == "inconsistent_child_level"}
    assert flagged == {"b"}


def test_correct_parent_child_levels_pass_cleanly():
    topics = [
        Topic(id="a", name="A", level=1, parent_id=None, source_pages=[1]),
        Topic(id="b", name="B", level=2, parent_id="a", source_pages=[2]),
        Topic(id="c", name="C", level=3, parent_id="b", source_pages=[3]),
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert report.passed
    assert not any(i.code in ("root_has_parent", "inconsistent_child_level", "missing_parent_for_nested_level") for i in report.issues)


def test_suspiciously_large_topic_list_is_warning():
    topics = [
        Topic(id=f"ch{i}", name=f"Topic {i}", level=1, parent_id=None, source_pages=[i])
        for i in range(DEFAULT_CONFIG.max_expected_topics + 5)
    ]
    report = validate_topics(make_result(topics), DEFAULT_CONFIG)
    assert report.passed
    assert any(i.code == "suspiciously_large_topic_list" for i in report.issues)
