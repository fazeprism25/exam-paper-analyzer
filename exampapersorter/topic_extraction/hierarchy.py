"""Deterministic reconstruction of parent/child links from `level` + order.

Observed directly on a real run: the extraction call produced internally
consistent `level` values for all 64 topics (1/2/3, matching the book's own
SECTION -> chapter-group -> chapter structure) but left `parent_id` null for
every non-root topic. The model committed to the nesting *depth* of each
topic but never wrote down the actual link.

This reconstructs that link the same way outline formats do (Markdown
headings, indented lists, `docutils` section trees): read topics in the
order the extraction call produced them -- which mirrors the evidence's own
reading order, since the model is walking a table top to bottom -- and take
a topic's parent to be the nearest PRECEDING topic with a strictly lower
level. That's the only assumption this makes; it never inspects `id`/`name`
naming conventions (rejected earlier as fragile and textbook-specific), and
it never invents a relationship the topic list's own level/order doesn't
already imply.

An already-populated parent_id is left untouched -- if the model did supply
a real link, it's trusted over a derived one.
"""
from __future__ import annotations

from exampapersorter.schemas import Topic


def derive_parent_ids(topics: list[Topic]) -> list[Topic]:
    stack: list[Topic] = []  # ancestor path so far, strictly increasing level
    result: list[Topic] = []

    for topic in topics:
        while stack and stack[-1].level >= topic.level:
            stack.pop()

        if topic.parent_id is None and topic.level > 1 and stack:
            topic = topic.model_copy(update={"parent_id": stack[-1].id})

        stack.append(topic)
        result.append(topic)

    return result
