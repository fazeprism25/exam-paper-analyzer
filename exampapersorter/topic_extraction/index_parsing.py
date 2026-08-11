"""Deterministic parsers that turn a user-supplied index/syllabus/topic
list into the same Topic hierarchy Stage 1 produces from a textbook's own
table of contents -- see topic_authority.py, which is what actually calls
these. No LLM call, no embedding call: these formats are already
structured enough that inferring level/parent_id from a hierarchy the
user themselves wrote down is a plain parsing problem, not a judgment
call worth spending model inference on.
"""
from __future__ import annotations

import re

from exampapersorter.schemas import Topic
from exampapersorter.topic_extraction.hierarchy import derive_parent_ids


class IndexParsingError(ValueError):
    """Raised when a supplied index/syllabus file can't be parsed into a
    topic hierarchy -- caller (topic_authority.py) wraps this into a
    TopicAuthorityError for clear user-facing reporting."""


_MARKDOWN_HEADER = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Strips a leading bullet/enumeration marker from an outline line before it
# becomes a topic name -- e.g. "- Lipid metabolism", "1.2 Lipid metabolism",
# "a) Lipid metabolism" all become just "Lipid metabolism". Indentation (not
# this marker) is what actually drives level in _parse_indented_outline.
_LEADING_MARKER = re.compile(r"^(?:[-*•]\s+|\d+(?:\.\d+)*[.)]\s+|[A-Za-z][.)]\s+)")


def _slugify(name: str, used: dict[str, int]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "topic"
    count = used.get(base, 0)
    used[base] = count + 1
    return base if count == 0 else f"{base}_{count + 1}"


def parse_topics_from_outline_text(text: str) -> list[Topic]:
    """Accepts either:
      - Markdown headings (# / ## / ### ...), level = number of '#'s, or
      - Plain indented text, one topic per non-blank line, where each
        additional level of leading-whitespace indentation is one level
        deeper than its nearest less-indented predecessor (like a Python/
        YAML outline -- any consistent indent unit works, tabs or spaces).
    A file is treated as Markdown if ANY non-blank line starts with '#';
    otherwise it's treated as a plain indented outline.
    """
    lines = text.splitlines()
    non_blank = [ln for ln in lines if ln.strip()]
    if not non_blank:
        raise IndexParsingError("File contains no topic lines")

    if any(ln.lstrip().startswith("#") for ln in non_blank):
        return _parse_markdown_headings(lines)
    return _parse_indented_outline(lines)


def _parse_markdown_headings(lines: list[str]) -> list[Topic]:
    used: dict[str, int] = {}
    topics: list[Topic] = []
    for raw in lines:
        if not raw.strip():
            continue
        m = _MARKDOWN_HEADER.match(raw.strip())
        if not m:
            continue  # non-heading prose lines (notes, separators) are ignored, not an error
        level = len(m.group(1))
        name = m.group(2).strip()
        topics.append(Topic(id=_slugify(name, used), name=name, level=level, source_pages=[]))
    if not topics:
        raise IndexParsingError("No '#'-style headings found even though '#' appeared in the file")
    return derive_parent_ids(topics)


def _parse_indented_outline(lines: list[str]) -> list[Topic]:
    used: dict[str, int] = {}
    topics: list[Topic] = []
    indent_stack: list[int] = []  # leading-whitespace lengths seen so far, strictly increasing
    for raw in lines:
        if not raw.strip():
            continue
        stripped = raw.rstrip("\n")
        indent = len(stripped) - len(stripped.lstrip(" \t"))
        name = _LEADING_MARKER.sub("", stripped.strip(), count=1).strip()
        if not name:
            continue

        while indent_stack and indent_stack[-1] >= indent:
            indent_stack.pop()
        indent_stack.append(indent)
        level = len(indent_stack)

        topics.append(Topic(id=_slugify(name, used), name=name, level=level, source_pages=[]))
    if not topics:
        raise IndexParsingError("No topic lines could be parsed")
    return derive_parent_ids(topics)


def parse_topics_from_json(data: object) -> list[Topic]:
    """Accepts either:
      - a nested tree: a list of {"name": ..., "children": [...]} nodes
        (children optional/empty for a leaf), or
      - a flat list of {"name": ..., "level": int, "parent_id": str|None}
        records (parent_id optional -- filled in from level+order via
        derive_parent_ids exactly like Stage 1's own LLM output is, see
        topic_extraction/hierarchy.py).
    Detected by whether every top-level item declares "level" explicitly.
    """
    if not isinstance(data, list) or not data:
        raise IndexParsingError("Top-level JSON must be a non-empty list of topic nodes")

    if all(isinstance(item, dict) and "level" in item for item in data):
        return _parse_flat_json(data)
    return _parse_nested_json(data)


def _parse_flat_json(data: list[dict]) -> list[Topic]:
    used: dict[str, int] = {}
    topics: list[Topic] = []
    for item in data:
        name = item.get("name")
        if not name or not str(name).strip():
            raise IndexParsingError(f"Topic entry missing a non-empty 'name': {item}")
        try:
            level = int(item["level"])
        except (TypeError, ValueError) as exc:
            raise IndexParsingError(f"Topic '{name}' has a non-integer 'level': {item.get('level')!r}") from exc
        if level < 1:
            raise IndexParsingError(f"Topic '{name}' has level={level}, but level must be >= 1")
        topic_id = item.get("id") or _slugify(str(name), used)
        topics.append(
            Topic(id=str(topic_id), name=str(name).strip(), level=level, parent_id=item.get("parent_id"), source_pages=[])
        )
    return derive_parent_ids(topics)


def _parse_nested_json(
    data: list, parent_id: str | None = None, level: int = 1, used: dict[str, int] | None = None
) -> list[Topic]:
    if used is None:
        used = {}
    topics: list[Topic] = []
    for item in data:
        if not isinstance(item, dict):
            raise IndexParsingError(f"Expected a topic object, got {item!r}")
        name = item.get("name")
        if not name or not str(name).strip():
            raise IndexParsingError(f"Topic entry missing a non-empty 'name': {item}")
        topic_id = str(item.get("id") or _slugify(str(name), used))
        topics.append(Topic(id=topic_id, name=str(name).strip(), level=level, parent_id=parent_id, source_pages=[]))
        children = item.get("children") or []
        if children:
            topics.extend(_parse_nested_json(children, parent_id=topic_id, level=level + 1, used=used))
    return topics
