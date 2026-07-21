"""utils.json_utils.extract_json failure modes."""

import pytest

from utils.json_utils import extract_json


def test_direct_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_direct_array():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_json_fence_object():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_bare_fence_object():
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_fence_with_nested_object_is_not_truncated():
    # The trailing fence anchors the non-greedy body to the last brace.
    assert extract_json('```json\n{"plan": [{"step": 1}]}\n```') == {
        "plan": [{"step": 1}]
    }


def test_object_with_leading_and_trailing_prose():
    assert extract_json('Here is the plan:\n{"a": 1}\nThanks!') == {"a": 1}


def test_array_takes_precedence_over_object():
    # An unfenced object whose value is an array: the array regex runs first, so
    # the *inner array* is returned, not the wrapping object. This documents the
    # precedence rule callers must know about.
    text = 'result: {"plan": [{"step": 1}, {"step": 2}]}'
    assert extract_json(text) == [{"step": 1}, {"step": 2}]


def test_unparseable_raises_with_truncated_payload():
    with pytest.raises(ValueError, match="Could not extract JSON"):
        extract_json("there is absolutely no json here")
