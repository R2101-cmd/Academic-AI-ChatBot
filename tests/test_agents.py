"""
Unit Tests for Multi-Agent Pipeline
Tests: RetrieverAgent, ReasonerAgent, VerifierAgent, RLPersonalizationAgent
"""

import pytest
import numpy as np
import faiss
from unittest.mock import patch, MagicMock

from backend.agents.retriever_agent import RetrieverAgent
from backend.agents.reasoner_agent import ReasonerAgent
from backend.agents.verifier_agent import VerifierAgent
from backend.agents.rl_personalization import RLPersonalizationAgent
from backend.graph.cognitive_graph import CognitiveGraphEngine
from backend.utils.validators import detect_request_mode, extract_topic_query, is_academic_query


class FakeSentenceTransformer:
    """Small deterministic embedding stub for offline tests."""

    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    def encode(self, texts, convert_to_numpy=True, **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for index, token in enumerate(text.lower().split()):
                slot = index % self.dimension
                vector[slot] += (sum(ord(char) for char in token) % 17) + 1
            vectors.append(vector)

        result = np.vstack(vectors).astype(np.float32)
        return result if convert_to_numpy else result.tolist()

    def get_sentence_embedding_dimension(self):
        return self.dimension


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        "Calculus is the study of change and motion.",
        "The chain rule is a fundamental rule in calculus.",
        "Backpropagation is used in neural networks.",
        "Gradients measure the rate of change.",
        "Neural networks are inspired by biological neurons.",
    ]


@pytest.fixture
def mock_faiss_index(sample_documents):
    """Create a mock FAISS index"""
    model = FakeSentenceTransformer()
    embeddings = model.encode(sample_documents)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    
    return index


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model"""
    return FakeSentenceTransformer()


@pytest.fixture
def cognitive_graph():
    """Create a sample cognitive graph"""
    graph = CognitiveGraphEngine()
    graph.build_sample_graph()
    return graph


# ============================================================
# RETRIEVER AGENT TESTS
# ============================================================

class TestRetrieverAgent:
    """Test suite for RetrieverAgent"""

    def test_retriever_initialization(self, mock_faiss_index, sample_documents, 
                                     mock_embedding_model, cognitive_graph):
        """Test RetrieverAgent initialization"""
        retriever = RetrieverAgent(
            mock_faiss_index, 
            sample_documents, 
            mock_embedding_model,
            cognitive_graph
        )
        
        assert retriever.index is not None
        assert retriever.documents == sample_documents
        assert retriever.model is not None
        assert retriever.graph is not None

    def test_retrieve_returns_dict(self, mock_faiss_index, sample_documents,
                                   mock_embedding_model, cognitive_graph):
        """Test that retrieve() returns correct dictionary structure"""
        retriever = RetrieverAgent(
            mock_faiss_index, 
            sample_documents, 
            mock_embedding_model,
            cognitive_graph
        )
        
        result = retriever.retrieve("What is backpropagation?", k=1)
        
        assert isinstance(result, dict)
        assert "semantic_docs" in result
        assert "graph_path" in result
        assert "combined_context" in result

    def test_retrieve_semantic_docs_not_empty(self, mock_faiss_index, 
                                              sample_documents, mock_embedding_model,
                                              cognitive_graph):
        """Test that semantic_docs are retrieved"""
        retriever = RetrieverAgent(
            mock_faiss_index, 
            sample_documents, 
            mock_embedding_model,
            cognitive_graph
        )
        
        result = retriever.retrieve("neural networks", k=1)
        
        assert len(result["semantic_docs"]) > 0
        assert isinstance(result["semantic_docs"], list)

    def test_retrieve_graph_path_not_empty(self, mock_faiss_index, sample_documents,
                                           mock_embedding_model, cognitive_graph):
        """Test that graph_path is extracted"""
        retriever = RetrieverAgent(
            mock_faiss_index, 
            sample_documents, 
            mock_embedding_model,
            cognitive_graph
        )
        
        result = retriever.retrieve("explain backpropagation", k=1)
        
        assert len(result["graph_path"]) > 0
        assert isinstance(result["graph_path"], list)

    def test_retrieve_combined_context(self, mock_faiss_index, sample_documents,
                                       mock_embedding_model, cognitive_graph):
        """Test that combined_context is properly formatted"""
        retriever = RetrieverAgent(
            mock_faiss_index, 
            sample_documents, 
            mock_embedding_model,
            cognitive_graph
        )
        
        result = retriever.retrieve("calculus", k=1)
        
        assert isinstance(result["combined_context"], str)
        assert len(result["combined_context"]) > 0

    def test_retrieve_with_different_k_values(self, mock_faiss_index, 
                                              sample_documents, mock_embedding_model,
                                              cognitive_graph):
        """Test retrieve with different k values"""
        retriever = RetrieverAgent(
            mock_faiss_index, 
            sample_documents, 
            mock_embedding_model,
            cognitive_graph
        )
        
        result_k1 = retriever.retrieve("calculus", k=1)
        result_k2 = retriever.retrieve("calculus", k=2)
        
        assert len(result_k1["semantic_docs"]) <= 1
        assert len(result_k2["semantic_docs"]) <= 2

    def test_extract_graph_path_backprop_keyword(self, mock_faiss_index, 
                                                 sample_documents, mock_embedding_model,
                                                 cognitive_graph):
        """Test graph path extraction for backprop queries"""
        retriever = RetrieverAgent(
            mock_faiss_index, 
            sample_documents, 
            mock_embedding_model,
            cognitive_graph
        )
        
        result = retriever.retrieve("explain backpropagation please", k=1)
        path = result["graph_path"]
        
        assert "Backpropagation" in path
        assert "Calculus" in path or "Chain Rule" in path

    def test_extract_graph_path_default_fallback(self, mock_faiss_index,
                                                 sample_documents, mock_embedding_model,
                                                 cognitive_graph):
        """Test graph path uses default for unrecognized queries"""
        retriever = RetrieverAgent(
            mock_faiss_index, 
            sample_documents, 
            mock_embedding_model,
            cognitive_graph
        )
        
        result = retriever.retrieve("random unknown topic xyz", k=1)
        path = result["graph_path"]
        
        # Should return default path
        assert len(path) > 0

    def test_query_normalization_for_artificial_intelligence(self, mock_faiss_index,
                                                             sample_documents, mock_embedding_model,
                                                             cognitive_graph):
        retriever = RetrieverAgent(
            mock_faiss_index,
            sample_documents,
            mock_embedding_model,
            cognitive_graph
        )

        normalized = retriever._normalize_query("what is artificial intelligent")
        assert "artificial intelligence" in normalized


# ============================================================
# REASONER AGENT TESTS
# ============================================================

class TestReasonerAgent:
    """Test suite for ReasonerAgent"""

    def test_reasoner_initialization(self):
        """Test ReasonerAgent initialization"""
        reasoner = ReasonerAgent(model_name="tinyllama")
        
        assert reasoner.model == "tinyllama"

    @patch('requests.post')
    def test_generate_explanation_success(self, mock_post):
        """Test successful explanation generation"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Step 1: Explanation"}
        mock_post.return_value = mock_response
        
        reasoner = ReasonerAgent()
        
        result = reasoner.generate_explanation(
            query="What is calculus?",
            context="Calculus is the study of change",
            graph_path=["Algebra", "Calculus"],
            difficulty="moderate"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0

    @patch('requests.post')
    def test_generate_explanation_with_different_difficulties(self, mock_post):
        """Test explanation generation with different difficulty levels"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Explanation"}
        mock_post.return_value = mock_response
        
        reasoner = ReasonerAgent()
        
        for difficulty in ["basic", "moderate", "advanced"]:
            result = reasoner.generate_explanation(
                query="Explain gradients",
                context="Gradients measure change",
                graph_path=["Calculus", "Gradient"],
                difficulty=difficulty
            )
            
            assert isinstance(result, str)

    @patch('requests.post')
    def test_generate_explanation_request_structure(self, mock_post):
        """Test that API request has correct structure"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Test"}
        mock_post.return_value = mock_response
        
        reasoner = ReasonerAgent()
        reasoner.generate_explanation(
            query="test",
            context="test",
            graph_path=["A", "B"],
            difficulty="moderate"
        )
        
        # Verify request was made
        mock_post.assert_called_once()
        
        # Verify request structure
        call_args = mock_post.call_args
        assert "json" in call_args.kwargs
        request_json = call_args.kwargs["json"]
        assert "model" in request_json
        assert "prompt" in request_json
        assert "stream" in request_json

    @patch('requests.post')
    def test_generate_explanation_with_graph_path_in_prompt(self, mock_post):
        """Test that graph path is included in prompt"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Explanation"}
        mock_post.return_value = mock_response
        
        reasoner = ReasonerAgent()
        graph_path = ["Calculus", "Chain Rule", "Backprop"]
        
        reasoner.generate_explanation(
            query="test",
            context="test context",
            graph_path=graph_path,
            difficulty="moderate"
        )
        
        # Check that path is in prompt
        call_args = mock_post.call_args
        prompt = call_args.kwargs["json"]["prompt"]
        
        assert "Calculus -> Chain Rule -> Backprop" in prompt

    @patch('requests.post')
    def test_generate_explanation_handles_request_error(self, mock_post):
        """Test error handling for API failures"""
        mock_post.side_effect = Exception("Connection error")
        
        reasoner = ReasonerAgent()
        
        result = reasoner.generate_explanation(
            query="test",
            context="test",
            graph_path=["A", "B"],
            difficulty="moderate"
        )
        
        assert "retrieved study notes" in result.lower()

    @patch('requests.post')
    def test_generate_explanation_falls_back_on_prompt_echo(self, mock_post):
        """Prompt echo should be converted into a grounded fallback answer."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": (
                "User question: What is artificial intelligence?\n"
                "Retrieved context: Artificial intelligence is the field of computer science.\n"
                "Return only the explanation text."
            )
        }
        mock_post.return_value = mock_response

        reasoner = ReasonerAgent()

        result = reasoner.generate_explanation(
            query="What is artificial intelligence?",
            context="Artificial intelligence is the field of computer science focused on building intelligent systems.",
            graph_path=["Computer Science", "Artificial Intelligence", "Applications"],
            difficulty="moderate"
        )

        assert "artificial intelligence is the field of computer science" in result.lower()
        assert "user question:" not in result.lower()

    @patch('requests.post')
    def test_generate_flashcards_returns_structured_cards(self, mock_post):
        """Test flashcard generation parsing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": '[{"front":"What is calculus?","back":"Study of change","hint":"Think derivatives"}]'
        }
        mock_post.return_value = mock_response

        reasoner = ReasonerAgent()
        cards = reasoner.generate_flashcards(
            query="make flashcards on calculus",
            context="Calculus is the study of change.",
            graph_path=["Algebra", "Calculus"],
        )

        assert isinstance(cards, list)
        assert cards[0]["front"] == "What is calculus?"

    @patch('requests.post')
    def test_generate_quiz_returns_structured_questions(self, mock_post):
        """Test quiz generation parsing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": '[{"question":"What is a gradient?","options":["Slope","Color","Sound","Shape"],"correct_index":0,"explanation":"A gradient measures rate of change."}]'
        }
        mock_post.return_value = mock_response

        reasoner = ReasonerAgent()
        quiz = reasoner.generate_quiz(
            query="generate mcq quiz on gradients",
            context="A gradient measures rate of change.",
            graph_path=["Calculus", "Gradient"],
        )

        assert isinstance(quiz, list)
        assert quiz[0]["correct_index"] == 0


# ============================================================
# VERIFIER AGENT TESTS
# ============================================================

class TestVerifierAgent:
    """Test suite for VerifierAgent"""

    @patch.object(ReasonerAgent, 'generate_explanation')
    def test_verifier_initialization(self, mock_generate, mock_embedding_model):
        """Test VerifierAgent initialization"""
        reasoner = ReasonerAgent()
        verifier = VerifierAgent(reasoner, mock_embedding_model)
        
        assert verifier.reasoner is not None
        assert verifier.model is not None

    @patch.object(ReasonerAgent, 'generate_explanation')
    def test_verify_returns_tuple(self, mock_generate, mock_embedding_model):
        """Test that verify() returns correct tuple structure"""
        mock_generate.return_value = "Test explanation with some content"
        
        reasoner = ReasonerAgent()
        verifier = VerifierAgent(reasoner, mock_embedding_model)
        
        result = verifier.verify(
            query="test",
            context="test context",
            graph_path=["A", "B"],
            difficulty="moderate"
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], str)  # explanation
        assert isinstance(result[1], bool)  # is_verified
        assert isinstance(result[2], (float, int))  # similarity_score

    @patch.object(ReasonerAgent, 'generate_explanation')
    def test_verify_similarity_threshold(self, mock_generate, mock_embedding_model):
        """Test similarity threshold logic"""
        # Same explanation both times → high similarity
        mock_generate.return_value = "Same explanation for both calls"
        
        reasoner = ReasonerAgent()
        verifier = VerifierAgent(reasoner, mock_embedding_model)
        
        _, is_verified, similarity = verifier.verify(
            query="test",
            context="context",
            graph_path=["same", "calls"],
            difficulty="moderate"
        )
        
        # High similarity should pass verification
        assert similarity > 0.7
        assert is_verified is True

    @patch.object(ReasonerAgent, 'generate_explanation')
    def test_verify_generates_two_variants(self, mock_generate, mock_embedding_model):
        """Test that verify generates two variants"""
        reasoner = ReasonerAgent()
        verifier = VerifierAgent(reasoner, mock_embedding_model)
        
        mock_generate.return_value = "Explanation"
        
        verifier.verify(
            query="test",
            context="context",
            graph_path=["A", "B"],
            difficulty="moderate"
        )
        
        # Should call generate_explanation twice
        assert mock_generate.call_count == 2

    @patch.object(ReasonerAgent, 'generate_explanation')
    def test_verify_returns_explanation(self, mock_generate, mock_embedding_model):
        """Test that verify returns first explanation"""
        expected_explanation = "This is the generated explanation"
        mock_generate.return_value = expected_explanation
        
        reasoner = ReasonerAgent()
        verifier = VerifierAgent(reasoner, mock_embedding_model)
        
        explanation, _, _ = verifier.verify(
            query="test",
            context="context",
            graph_path=["A", "B"],
            difficulty="moderate"
        )
        
        assert explanation == expected_explanation


# ============================================================
# RL PERSONALIZATION AGENT TESTS
# ============================================================

class TestRLPersonalizationAgent:
    """Test suite for RLPersonalizationAgent"""

    @pytest.fixture
    def rl_agent(self, tmp_path):
        """Create RL agent with temporary database"""
        db_path = str(tmp_path / "test_session.db")
        return RLPersonalizationAgent(db_path=db_path)

    def test_rl_initialization(self, rl_agent):
        """Test RLPersonalizationAgent initialization"""
        assert rl_agent.db_path is not None

    def test_track_performance(self, rl_agent):
        """Test performance tracking"""
        rl_agent.track_performance(
            concept="Calculus",
            quiz_score=0.85,
            engagement_time=20.5
        )
        
        # Should not raise exception

    def test_track_multiple_performances(self, rl_agent):
        """Test tracking multiple performances"""
        concepts = ["Calculus", "Gradient", "Backprop"]
        scores = [0.6, 0.75, 0.9]
        
        for concept, score in zip(concepts, scores):
            rl_agent.track_performance(concept, score, 15.0)
        
        # Should not raise exception

    def test_get_difficulty_default(self, rl_agent):
        """Test difficulty recommendation for new concept"""
        difficulty = rl_agent.get_difficulty("UnknownConcept")
        
        assert difficulty == "moderate"

    def test_get_difficulty_basic(self, rl_agent):
        """Test difficulty recommendation for low performance"""
        # Track low scores
        rl_agent.track_performance("Calculus", 0.4, 10.0)
        rl_agent.track_performance("Calculus", 0.5, 12.0)
        
        difficulty = rl_agent.get_difficulty("Calculus")
        
        assert difficulty == "basic"

    def test_get_difficulty_moderate(self, rl_agent):
        """Test difficulty recommendation for medium performance"""
        rl_agent.track_performance("Gradient", 0.65, 15.0)
        rl_agent.track_performance("Gradient", 0.75, 18.0)
        
        difficulty = rl_agent.get_difficulty("Gradient")
        
        assert difficulty == "moderate"

    def test_get_difficulty_advanced(self, rl_agent):
        """Test difficulty recommendation for high performance"""
        rl_agent.track_performance("Backprop", 0.85, 20.0)
        rl_agent.track_performance("Backprop", 0.95, 22.0)
        
        difficulty = rl_agent.get_difficulty("Backprop")
        
        assert difficulty == "advanced"

    def test_get_difficulty_progression(self, rl_agent):
        """Test difficulty progression as performance improves"""
        concept = "Calculus"
        
        # Start with low performance
        rl_agent.track_performance(concept, 0.5, 15.0)
        assert rl_agent.get_difficulty(concept) == "basic"
        
        # Improve performance
        rl_agent.track_performance(concept, 0.7, 18.0)
        assert rl_agent.get_difficulty(concept) == "moderate"
        
        # Further improvement
        rl_agent.track_performance(concept, 0.9, 20.0)
        assert rl_agent.get_difficulty(concept) == "advanced"


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestAgentsIntegration:
    """Integration tests for agents working together"""

    @patch('requests.post')
    def test_retriever_reasoner_pipeline(self, mock_post, mock_faiss_index,
                                        sample_documents, mock_embedding_model,
                                        cognitive_graph):
        """Test Retriever → Reasoner pipeline"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Step-by-step explanation"}
        mock_post.return_value = mock_response
        
        # Retrieve
        retriever = RetrieverAgent(
            mock_faiss_index, 
            sample_documents, 
            mock_embedding_model,
            cognitive_graph
        )
        retrieval_result = retriever.retrieve("explain backpropagation", k=1)
        
        # Reason
        reasoner = ReasonerAgent()
        explanation = reasoner.generate_explanation(
            query="explain backpropagation",
            context=retrieval_result["combined_context"],
            graph_path=retrieval_result["graph_path"],
            difficulty="moderate"
        )
        
        assert len(explanation) > 0

    @patch('requests.post')
    def test_full_pipeline_with_verification(self, mock_post, mock_faiss_index,
                                            sample_documents, mock_embedding_model,
                                            cognitive_graph):
        """Test full Retriever → Reasoner → Verifier pipeline"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Complete explanation"}
        mock_post.return_value = mock_response
        
        # Retrieve
        retriever = RetrieverAgent(
            mock_faiss_index, 
            sample_documents, 
            mock_embedding_model,
            cognitive_graph
        )
        retrieval = retriever.retrieve("neural networks", k=1)
        
        # Reason
        reasoner = ReasonerAgent()
        
        # Verify
        verifier = VerifierAgent(reasoner, mock_embedding_model)
        explanation, is_verified, similarity = verifier.verify(
            query="neural networks",
            context=retrieval["combined_context"],
            graph_path=retrieval["graph_path"],
            difficulty="moderate"
        )
        
        assert isinstance(explanation, str)
        assert isinstance(is_verified, bool)
        assert 0 <= similarity <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


class TestStudyModeValidators:
    """Tests for explicit study-mode request detection."""

    def test_detect_quiz_mode(self):
        assert detect_request_mode("generate an mcq quiz on backpropagation") == "quiz"

    def test_detect_flashcards_mode(self):
        assert detect_request_mode("make flashcards for calculus") == "flashcards"

    def test_extract_topic_query(self):
        cleaned = extract_topic_query("generate flashcards on backpropagation")
        assert "backpropagation" in cleaned

    def test_is_academic_query_accepts_general_academic_topics(self):
        assert is_academic_query("What is artificial intelligence?") is True
        assert is_academic_query("Explain the causes of World War 1") is True
        assert is_academic_query("Solve this calculus derivative") is True

    def test_is_academic_query_now_accepts_broad_topics(self):
        assert is_academic_query("What is the weather today?") is True
        assert is_academic_query("Recommend a movie for tonight") is True
        assert is_academic_query("Can you help with my relationship?") is True
