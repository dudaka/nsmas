"""
NS-MAS LangGraph Agent Module - Phase 4

This module implements the Generator -> Verifier -> Reflector loop
for neuro-symbolic mathematical reasoning.

Architecture:
    - Generator: LLM translates word problems to ASP code
    - Verifier: Clingo solver validates ASP code
    - Reflector: Error-specific coaching for self-correction

Usage:
    from src.agent import Agent, AgentConfig, LLMConfig, LLMProvider

    config = AgentConfig(
        generator_llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o"),
        max_retries=5,
    )
    agent = Agent(config)
    result = agent.solve("John has 10 apples. Mary has 5. How many total?")
    print(result.final_answer)  # 15

Reference: Phase 4 Implementation Clarifications
"""

from .config import AgentConfig, LLMConfig, LLMProvider
from .state import AgentState, VerificationResult
from .graph import Agent, create_agent_graph

__all__ = [
    # Configuration
    "AgentConfig",
    "LLMConfig",
    "LLMProvider",
    # State
    "AgentState",
    "VerificationResult",
    # Agent
    "Agent",
    "create_agent_graph",
]
