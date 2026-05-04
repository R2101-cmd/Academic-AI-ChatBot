from backend.modules.flashcard import FlashcardModule
from backend.modules.graph_cot import GraphCoTModule
from backend.modules.nlp import NLPProcessor
from backend.modules.quiz import QuizModule
from backend.modules.verification import VerificationModule


def test_nlp_detects_academic_quiz_mode():
    processed = NLPProcessor().preprocess("Create an MCQ quiz on backpropagation")

    assert processed.is_academic is True
    assert processed.mode == "quiz"
    assert "backpropagation" in processed.topic


def test_nlp_rejects_non_academic_query():
    processed = NLPProcessor().preprocess("Recommend a movie for tonight")

    assert processed.is_academic is False
    assert processed.rejection_reason is not None


def test_graph_cot_returns_prerequisite_path():
    path = GraphCoTModule().build_path("backpropagation")

    assert path[0] == "Algebra"
    assert path[-1] == "Backpropagation"
    assert "Chain Rule" in path


def test_quiz_and_flashcards_are_structured():
    graph_path = ["Calculus", "Derivatives", "Chain Rule"]
    context = "The chain rule differentiates composite functions. Derivatives measure rate of change."

    quiz = QuizModule().generate("chain rule", context, graph_path, "moderate")
    cards = FlashcardModule().generate("chain rule", context, graph_path)

    assert len(quiz) > 0
    assert len(quiz[0]["options"]) == 4
    assert 0 <= quiz[0]["correct_index"] <= 3
    assert len(cards) > 0
    assert {"front", "back", "hint"} <= set(cards[0])


def test_verification_scores_grounded_answer():
    result = VerificationModule().verify(
        topic="chain rule",
        explanation="The chain rule explains how derivatives work through composite functions.",
        context="The chain rule differentiates composite functions using derivatives.",
        graph_path=["Calculus", "Derivatives", "Chain Rule"],
    )

    assert result["verified"] is True
    assert result["score"] > 0
