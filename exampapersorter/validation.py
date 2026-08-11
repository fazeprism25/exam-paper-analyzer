"""Deterministic validation of an LLM-extracted topic list.

Nothing here calls the LLM or "fixes" anything -- it only flags problems.
`error`-severity issues mean the topic list should not be accepted as-is;
`warning`-severity issues are surfaced for the human to judge (e.g. a
suspiciously small topic count might be a genuinely short book, or might
mean extraction only caught a subsection of the TOC).
"""
from __future__ import annotations

import re

from exampapersorter.config import Config
from exampapersorter.schemas import Topic, TopicExtractionResult, ValidationIssue, ValidationReport

# Textbook OCR sometimes renders decorative/letter-spaced heading fonts as
# individually spaced characters -- e.g. "Biomolecules and the Cell" comes
# out as "B i o m o l ecu l es and t he ce ll". This is NOT specific to any
# one book: it's a generic artifact of how PDF text extraction handles
# wide letter-tracking in display fonts, and it matters here because a
# model can echo this text verbatim and self-report "resolved" without
# recognizing it's unreadable -- observed directly (see extract.py). This
# heuristic acts as a deterministic safety net independent of whether the
# model's own self-assessment was followed.
_COMMON_SHORT_WORDS = {
    "a", "an", "of", "in", "to", "is", "on", "at", "by", "or", "if", "it", "as", "be", "we",
}


def looks_letter_spaced(text: str) -> bool:
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    alpha_tokens = [t for t in tokens if any(c.isalpha() for c in t)]
    if len(alpha_tokens) < 4:
        return False  # too short to judge reliably one way or the other
    short_count = sum(1 for t in alpha_tokens if len(t) <= 2 and t.lower() not in _COMMON_SHORT_WORDS)
    return (short_count / len(alpha_tokens)) > 0.4


def reconcile_resolution_status(topics: list[Topic]) -> list[Topic]:
    """Deterministic correction pass: downgrade any topic the model marked
    "resolved" whose name is actually letter-spaced/garbled. Independent
    of the model's own self-report -- see looks_letter_spaced.

    Clears name_evidence_source/name_source_pages too, since whatever the
    model thought it read there wasn't actually legible.
    """
    corrected = []
    for topic in topics:
        if topic.resolution_status == "resolved" and looks_letter_spaced(topic.name):
            corrected.append(
                topic.model_copy(
                    update={
                        "resolution_status": "name_unresolved",
                        "name_evidence_source": None,
                        "name_source_pages": [],
                    }
                )
            )
        else:
            corrected.append(topic)
    return corrected

# A topic name that's just digits/punctuation (e.g. "1", "12.") is almost
# certainly an extraction error -- the LLM likely mis-mapped a table's
# chapter-number column onto the name field instead of the title column.
# Observed directly: on a garbled, letter-spaced OCR'd contents table, the
# model produced 43 chapters named "1".."43" instead of their real titles.
# Used here as a cross-check against the model's own self-reported
# resolution_status: Topic.resolution_status is the primary signal (set
# deliberately by the extraction/recovery prompts), but a topic claiming
# "resolved" with a numeric-only name means that self-report doesn't match
# its own content -- worth flagging as an error, not silently trusted.
_NUMERIC_ONLY_NAME = re.compile(r"^[\d\s.\-]+$")


def validate_topics(
    result: TopicExtractionResult, config: Config, valid_page_range: tuple[int, int] | None = None
) -> ValidationReport:
    """valid_page_range, if given, is the (start_page, end_page) of the evidence
    actually shown to the LLM. A topic citing a source_page outside that range
    is a grounding failure -- the model referenced a page it was never shown --
    flagged so it doesn't pass silently."""
    issues: list[ValidationIssue] = []
    topics = result.topics

    seen_ids: dict[str, int] = {}
    seen_names: dict[str, int] = {}
    for topic in topics:
        seen_ids[topic.id] = seen_ids.get(topic.id, 0) + 1
        seen_names[topic.name.strip().lower()] = seen_names.get(topic.name.strip().lower(), 0) + 1

    for topic_id, count in seen_ids.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="duplicate_topic_id",
                    message=f"Topic id '{topic_id}' appears {count} times",
                    topic_id=topic_id,
                )
            )

    for name, count in seen_names.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="duplicate_topic_name",
                    message=f"Topic name '{name}' appears {count} times",
                )
            )

    all_ids = set(seen_ids.keys())
    topics_by_id = {t.id: t for t in topics}
    for topic in topics:
        if not topic.name or not topic.name.strip():
            issues.append(
                ValidationIssue(severity="error", code="empty_name", message="Topic has an empty name", topic_id=topic.id)
            )
        elif topic.resolution_status == "resolved" and _NUMERIC_ONLY_NAME.match(topic.name.strip()):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="inconsistent_resolution_status",
                    message=(
                        f"Topic '{topic.id}' is marked resolved but its name is only digits/punctuation "
                        f"('{topic.name.strip()}') -- the self-reported status doesn't match its own content"
                    ),
                    topic_id=topic.id,
                )
            )
        elif topic.resolution_status == "resolved" and looks_letter_spaced(topic.name):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="inconsistent_resolution_status",
                    message=(
                        f"Topic '{topic.id}' is marked resolved but its name looks letter-spaced/garbled "
                        f"('{topic.name.strip()}') -- the self-reported status doesn't match its own content"
                    ),
                    topic_id=topic.id,
                )
            )
        elif topic.resolution_status != "resolved":
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="unresolved_topic",
                    message=f"Topic '{topic.id}' (name '{topic.name}') has resolution_status={topic.resolution_status!r}",
                    topic_id=topic.id,
                )
            )
        if topic.parent_id is not None:
            if topic.parent_id == topic.id:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="self_parent",
                        message=f"Topic '{topic.id}' lists itself as its own parent",
                        topic_id=topic.id,
                    )
                )
            elif topic.parent_id not in all_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="unknown_parent",
                        message=f"Topic '{topic.id}' references parent_id '{topic.parent_id}' which does not exist",
                        topic_id=topic.id,
                    )
                )
            elif topic.level == 1:
                # level=1 is defined (by the extraction prompt) as top-level --
                # a level-1 topic claiming a parent contradicts its own level,
                # and it's not this code's call to decide which field is right.
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="root_has_parent",
                        message=f"Topic '{topic.id}' has level=1 (top-level) but also declares parent_id '{topic.parent_id}'",
                        topic_id=topic.id,
                    )
                )
            else:
                parent = topics_by_id[topic.parent_id]
                if topic.level != parent.level + 1:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="inconsistent_child_level",
                            message=(
                                f"Topic '{topic.id}' has level={topic.level} under parent '{parent.id}' "
                                f"(level={parent.level}) -- expected level={parent.level + 1}"
                            ),
                            topic_id=topic.id,
                        )
                    )
        elif topic.level > 1:
            # level > 1 claims this topic is nested under something, but
            # parent_id is null -- the hierarchy DEPTH was recorded without
            # the actual parent-child LINK, an internally inconsistent
            # output worth flagging rather than silently accepting.
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="missing_parent_for_nested_level",
                    message=f"Topic '{topic.id}' has level={topic.level} (implies nesting) but parent_id is null",
                    topic_id=topic.id,
                )
            )

    if valid_page_range is not None:
        low, high = valid_page_range
        for topic in topics:
            out_of_range = [p for p in topic.source_pages if p < low or p > high]
            if out_of_range:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="source_page_out_of_range",
                        message=(
                            f"Topic '{topic.id}' cites source_pages {out_of_range} outside the "
                            f"evidence range {low}-{high} it was extracted from"
                        ),
                        topic_id=topic.id,
                    )
                )

    cycle_ids = _find_cycle_members(topics)
    for topic_id in cycle_ids:
        issues.append(
            ValidationIssue(
                severity="error",
                code="parent_cycle",
                message=f"Topic '{topic_id}' is part of a parent-reference cycle",
                topic_id=topic_id,
            )
        )

    topic_count = len(topics)
    if topic_count < config.min_expected_topics:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="suspiciously_small_topic_list",
                message=f"Only {topic_count} topics extracted (below configured minimum {config.min_expected_topics})",
            )
        )
    if topic_count > config.max_expected_topics:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="suspiciously_large_topic_list",
                message=f"{topic_count} topics extracted (above configured maximum {config.max_expected_topics})",
            )
        )

    passed = not any(issue.severity == "error" for issue in issues)
    return ValidationReport(passed=passed, topic_count=topic_count, issues=issues)


def _find_cycle_members(topics: list) -> set[str]:
    parent_of = {t.id: t.parent_id for t in topics}
    cycle_members: set[str] = set()

    for start_id in parent_of:
        visited: list[str] = []
        current = start_id
        while current is not None:
            if current in visited:
                if current in visited[visited.index(current):]:
                    cycle_members.update(visited[visited.index(current):])
                break
            if current not in parent_of:
                break
            visited.append(current)
            current = parent_of[current]

    return cycle_members
