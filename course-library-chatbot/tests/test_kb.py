"""Run with:  python -m pytest tests   (needs: pip install pytest)"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kb import KnowledgeBase  # noqa: E402


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def test_library_has_all_18_documents(kb):
    assert len(kb.documents) == 18


def test_every_suggested_question_is_answerable_from_the_right_books(kb):
    for topic in kb.suggestions["topics"]:
        for q in topic["questions"]:
            passages, in_scope = kb.search(q)
            assert in_scope, q
            assert any(p["doc"] in topic["docs"] for p in passages[:3]), q


@pytest.mark.parametrize("q", [
    "What is the capital of France?", "Who won the 2022 football world cup?",
    "write me a poem about love", "how do I bake bread", "who is the president of zambia",
])
def test_unrelated_questions_are_out_of_scope(kb, q):
    assert kb.search(q)[1] is False


def test_follow_ups_do_not_repeat_asked_questions(kb):
    q = "How can session tokens be attacked?"
    passages, _ = kb.search(q)
    follow = kb.follow_ups(q, passages, asked=[q], n=3)
    assert len(follow) == 3 and q not in follow


def test_search_results_carry_page_numbers(kb):
    passages, _ = kb.search("What is a buffer overflow?")
    assert all(isinstance(p["page"], int) and p["page"] >= 1 for p in passages)
