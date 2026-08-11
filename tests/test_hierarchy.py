from exampapersorter.schemas import Topic
from exampapersorter.topic_extraction.hierarchy import derive_parent_ids


def t(id, level, parent_id=None, name=None):
    return Topic(id=id, name=name or id, level=level, parent_id=parent_id, source_pages=[10])


def test_simple_two_level_nesting_gets_parent_assigned():
    topics = [
        t("section", 1),
        t("chapter_a", 2),
        t("chapter_b", 2),
    ]
    result = derive_parent_ids(topics)
    by_id = {r.id: r for r in result}
    assert by_id["chapter_a"].parent_id == "section"
    assert by_id["chapter_b"].parent_id == "section"
    assert by_id["section"].parent_id is None


def test_three_level_nesting_assigns_nearest_ancestor():
    topics = [
        t("section", 1),
        t("group", 2),
        t("leaf_1", 3),
        t("leaf_2", 3),
    ]
    result = derive_parent_ids(topics)
    by_id = {r.id: r for r in result}
    assert by_id["group"].parent_id == "section"
    assert by_id["leaf_1"].parent_id == "group"
    assert by_id["leaf_2"].parent_id == "group"


def test_stack_resets_on_new_top_level_section():
    topics = [
        t("section_a", 1),
        t("group_a", 2),
        t("leaf_a1", 3),
        t("section_b", 1),
        t("group_b", 2),
        t("leaf_b1", 3),
    ]
    result = derive_parent_ids(topics)
    by_id = {r.id: r for r in result}
    assert by_id["group_a"].parent_id == "section_a"
    assert by_id["leaf_a1"].parent_id == "group_a"
    assert by_id["group_b"].parent_id == "section_b"
    assert by_id["leaf_b1"].parent_id == "group_b"
    # leaf_a1's subtree must not bleed into section_b's
    assert by_id["group_b"].parent_id != "group_a"


def test_pre_existing_parent_id_is_preserved_not_overwritten():
    topics = [
        t("section", 1),
        t("other_section", 1),
        t("chapter", 2, parent_id="other_section"),  # model already linked this one, elsewhere
    ]
    result = derive_parent_ids(topics)
    by_id = {r.id: r for r in result}
    # must NOT be reassigned to the nearest preceding "section"
    assert by_id["chapter"].parent_id == "other_section"


def test_level_gap_assigns_nearest_lower_ancestor_available():
    # level jumps 1 -> 3 with nothing at level 2 in between
    topics = [
        t("section", 1),
        t("leaf", 3),
    ]
    result = derive_parent_ids(topics)
    by_id = {r.id: r for r in result}
    assert by_id["leaf"].parent_id == "section"


def test_no_eligible_ancestor_leaves_parent_id_none():
    # first topic in the list already claims to be nested -- nothing precedes it
    topics = [t("orphan", 2)]
    result = derive_parent_ids(topics)
    assert result[0].parent_id is None


def test_root_level_topics_never_get_a_parent_assigned():
    topics = [t("a", 1), t("b", 1), t("c", 1)]
    result = derive_parent_ids(topics)
    assert all(r.parent_id is None for r in result)
