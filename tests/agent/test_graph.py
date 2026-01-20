"""
Tests for the agent graph and routing logic.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.agent.config import AgentConfig, LLMConfig, LLMProvider
from src.agent.state import AgentState, create_initial_state
from src.agent.graph import route_after_verification, create_agent_graph, Agent


class TestRouteAfterVerification:
    """Tests for the routing logic after verification."""

    def test_route_to_end_on_success(self):
        """Test routing to END when verification succeeds."""
        state = create_initial_state(question="Test")
        state["status"] = "success"
        state["iteration_count"] = 1

        result = route_after_verification(state)
        assert result == "end"

    def test_route_to_end_on_cycle(self):
        """Test routing to END when cycle detected."""
        state = create_initial_state(question="Test")
        state["status"] = "cycle_detected"
        state["iteration_count"] = 2

        result = route_after_verification(state)
        assert result == "end"

    def test_route_to_end_on_max_retries(self):
        """Test routing to END when max retries exceeded."""
        state = create_initial_state(question="Test", max_retries=3)
        state["status"] = "running"
        state["iteration_count"] = 3

        result = route_after_verification(state)
        assert result == "end"

    def test_route_to_reflect_when_running(self):
        """Test routing to reflect when still running."""
        state = create_initial_state(question="Test", max_retries=5)
        state["status"] = "running"
        state["iteration_count"] = 1

        result = route_after_verification(state)
        assert result == "reflect"

    def test_route_considers_max_retries(self):
        """Test that routing respects max_retries setting."""
        state = create_initial_state(question="Test", max_retries=2)
        state["status"] = "running"

        # At iteration 1, should still reflect
        state["iteration_count"] = 1
        assert route_after_verification(state) == "reflect"

        # At iteration 2, should end
        state["iteration_count"] = 2
        assert route_after_verification(state) == "end"


class TestAgentGraph:
    """Tests for the full agent graph."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AgentConfig(
            generator_llm=LLMConfig(provider=LLMProvider.OPENAI),
            max_retries=3,
            enable_entity_extraction=False,  # Skip for faster tests
            solver_timeout_ms=5000,
            trace_mode="OFF",
        )

    def test_graph_creation(self, config):
        """Test that graph can be created."""
        graph = create_agent_graph(config)
        assert graph is not None

    def test_graph_with_entity_extraction(self, config):
        """Test graph creation with entity extraction enabled."""
        config.enable_entity_extraction = True
        graph = create_agent_graph(config)
        assert graph is not None


class TestAgent:
    """Tests for the Agent class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AgentConfig(
            generator_llm=LLMConfig(provider=LLMProvider.OPENAI),
            max_retries=2,
            enable_entity_extraction=False,
            solver_timeout_ms=5000,
            trace_mode="OFF",
        )

    def test_agent_initialization(self, config):
        """Test that agent can be initialized."""
        agent = Agent(config)
        assert agent.config == config
        assert agent._graph is None  # Lazy init

    def test_agent_graph_lazy_init(self, config):
        """Test that graph is lazily initialized."""
        agent = Agent(config)
        assert agent._graph is None

        # Access graph property
        graph = agent.graph
        assert graph is not None
        assert agent._graph is not None

    @patch("src.agent.nodes.generate_asp.create_llm_provider")
    def test_agent_solve_with_mock(self, mock_create_provider, config):
        """Test agent solve with mocked LLM."""
        # Configure mock to return valid ASP
        mock_provider = MagicMock()
        mock_provider.generate.return_value = """```asp
quantity(john, apples, 10).
quantity(mary, apples, 5).
total(T) :- quantity(john, apples, A), quantity(mary, apples, B), T = @add(A, B).
final_answer(T) :- total(T).
```"""
        mock_create_provider.return_value = mock_provider

        agent = Agent(config)
        result = agent.solve("John has 10 apples. Mary has 5. How many total?")

        assert result["status"] == "success"
        assert result["final_answer"] == 15
