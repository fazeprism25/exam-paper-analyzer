import pytest

from exampapersorter.topic_extraction.index_parsing import (
    IndexParsingError,
    parse_topics_from_json,
    parse_topics_from_outline_text,
)


def by_name(topics):
    return {t.name: t for t in topics}


# --- Markdown headings ---


def test_markdown_headings_assign_level_from_hash_count():
    text = "# Section A\n## Chapter 1\n## Chapter 2\n# Section B\n## Chapter 3\n"
    topics = parse_topics_from_outline_text(text)
    n = by_name(topics)
    assert n["Section A"].level == 1
    assert n["Chapter 1"].level == 2
    assert n["Chapter 2"].level == 2
    assert n["Section B"].level == 1
    assert n["Chapter 1"].parent_id == n["Section A"].id
    assert n["Chapter 2"].parent_id == n["Section A"].id
    assert n["Chapter 3"].parent_id == n["Section B"].id
    assert n["Section A"].parent_id is None


def test_markdown_ignores_non_heading_prose_lines():
    text = "# Chapter 1\nSome descriptive prose that isn't a heading.\n# Chapter 2\n"
    topics = parse_topics_from_outline_text(text)
    assert [t.name for t in topics] == ["Chapter 1", "Chapter 2"]


def test_markdown_with_only_hash_characters_and_no_real_headings_raises():
    with pytest.raises(IndexParsingError):
        parse_topics_from_outline_text("# \n#\n")


# --- Indented plain text outline ---


def test_indented_outline_assigns_level_from_indentation_depth():
    text = "Section A\n  Chapter 1\n  Chapter 2\nSection B\n  Chapter 3\n"
    topics = parse_topics_from_outline_text(text)
    n = by_name(topics)
    assert n["Section A"].level == 1
    assert n["Chapter 1"].level == 2
    assert n["Chapter 1"].parent_id == n["Section A"].id
    assert n["Chapter 3"].parent_id == n["Section B"].id


def test_indented_outline_supports_three_levels():
    text = "Section\n  Group\n    Leaf 1\n    Leaf 2\n"
    topics = parse_topics_from_outline_text(text)
    n = by_name(topics)
    assert n["Group"].parent_id == n["Section"].id
    assert n["Leaf 1"].parent_id == n["Group"].id
    assert n["Leaf 2"].parent_id == n["Group"].id


def test_indented_outline_strips_bullet_and_numeric_markers():
    text = "- Chapter One\n- Chapter Two\n"
    topics = parse_topics_from_outline_text(text)
    assert {t.name for t in topics} == {"Chapter One", "Chapter Two"}

    text2 = "1. Chapter One\n2. Chapter Two\n"
    topics2 = parse_topics_from_outline_text(text2)
    assert {t.name for t in topics2} == {"Chapter One", "Chapter Two"}


def test_blank_lines_are_skipped():
    text = "Section A\n\n  Chapter 1\n\n\nSection B\n"
    topics = parse_topics_from_outline_text(text)
    assert [t.name for t in topics] == ["Section A", "Chapter 1", "Section B"]


def test_duplicate_names_get_distinct_ids():
    text = "Chapter One\nChapter One\n"
    topics = parse_topics_from_outline_text(text)
    ids = [t.id for t in topics]
    assert len(ids) == len(set(ids))


def test_empty_outline_text_raises():
    with pytest.raises(IndexParsingError):
        parse_topics_from_outline_text("   \n\n  ")


# --- JSON: nested tree format ---


def test_json_nested_tree_derives_level_and_parent_from_structure():
    data = [
        {
            "name": "Section A",
            "children": [
                {"name": "Chapter 1"},
                {"name": "Chapter 2", "children": [{"name": "Subtopic 2.1"}]},
            ],
        },
        {"name": "Section B"},
    ]
    topics = parse_topics_from_json(data)
    n = by_name(topics)
    assert n["Section A"].level == 1
    assert n["Chapter 1"].level == 2
    assert n["Chapter 1"].parent_id == n["Section A"].id
    assert n["Subtopic 2.1"].level == 3
    assert n["Subtopic 2.1"].parent_id == n["Chapter 2"].id
    assert n["Section B"].parent_id is None


def test_json_nested_tree_respects_explicit_ids():
    data = [{"id": "sec_a", "name": "Section A", "children": [{"id": "ch_1", "name": "Chapter 1"}]}]
    topics = parse_topics_from_json(data)
    n = by_name(topics)
    assert n["Section A"].id == "sec_a"
    assert n["Chapter 1"].id == "ch_1"
    assert n["Chapter 1"].parent_id == "sec_a"


# --- JSON: flat format ---


def test_json_flat_format_uses_explicit_level_and_derives_parent():
    data = [
        {"name": "Section A", "level": 1},
        {"name": "Chapter 1", "level": 2},
        {"name": "Chapter 2", "level": 2},
    ]
    topics = parse_topics_from_json(data)
    n = by_name(topics)
    assert n["Chapter 1"].parent_id == n["Section A"].id
    assert n["Chapter 2"].parent_id == n["Section A"].id


def test_json_flat_format_respects_explicit_parent_id():
    data = [
        {"id": "a", "name": "Section A", "level": 1},
        {"id": "b", "name": "Other Section", "level": 1},
        {"id": "c", "name": "Chapter Under Other", "level": 2, "parent_id": "b"},
    ]
    topics = parse_topics_from_json(data)
    n = by_name(topics)
    # Must NOT be reassigned to the nearest preceding level-1 topic ("a")
    assert n["Chapter Under Other"].parent_id == "b"


def test_json_flat_format_rejects_non_integer_level():
    with pytest.raises(IndexParsingError):
        parse_topics_from_json([{"name": "X", "level": "not-a-number"}])


def test_json_flat_format_rejects_level_below_one():
    with pytest.raises(IndexParsingError):
        parse_topics_from_json([{"name": "X", "level": 0}])


def test_json_missing_name_raises():
    with pytest.raises(IndexParsingError):
        parse_topics_from_json([{"level": 1}])


def test_json_top_level_not_a_list_raises():
    with pytest.raises(IndexParsingError):
        parse_topics_from_json({"name": "not a list"})


def test_json_empty_list_raises():
    with pytest.raises(IndexParsingError):
        parse_topics_from_json([])
