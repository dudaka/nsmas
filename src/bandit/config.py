"""
Configuration dataclasses for the Bandit Router.

Provides BanditConfig following the pattern from agent/config.py.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from src.agent.config import LLMProvider


@dataclass
class BanditConfig:
    """
    Configuration for the Contextual Bandit Router.

    Design Decisions (from Phase 5 research):
    - Binary action space: Fast (0) vs Slow (1)
    - SquareCB exploration for adaptive exploration
    - Hybrid features: regex + MiniLM embeddings
    - Offline policy evaluation (no online learning during inference)

    Attributes:
        exploration_algo: VW exploration algorithm ("squarecb" or "epsilon")
        gamma_scale: SquareCB greediness parameter (higher = more greedy)
        epsilon: Epsilon for epsilon-greedy exploration (fallback)

        alpha_incorrect: Loss penalty for incorrect answer (high to prevent cheap failures)
        beta_cost: Cost weight in loss function

        fast_provider: LLM provider for fast path
        fast_model: Model identifier for fast path
        fast_temperature: Temperature for fast path (0.0 for deterministic)
        fast_max_tokens: Max tokens for fast path response

        embedding_model: Sentence transformer model for semantic features
        embedding_dim: Dimension of embedding vectors
        use_onnx: Whether to use ONNX runtime for faster inference

        model_dir: Directory for persisting VW model
        model_filename: Filename for VW model

        training_subset_size: Number of samples for oracle construction
        enable_online_learning: Whether to update policy during inference
    """

    # VW Configuration
    exploration_algo: Literal["squarecb", "epsilon"] = "squarecb"
    gamma_scale: float = 10.0
    epsilon: float = 0.2

    # Loss Function Parameters
    # Loss = alpha * I_incorrect + beta * cost
    alpha_incorrect: float = 10.0  # High penalty for failure
    beta_cost: float = 1.0  # Normalized cost weight

    # Fast Path Configuration
    fast_provider: LLMProvider = LLMProvider.OPENAI
    fast_model: str = "gpt-4o-mini"
    fast_temperature: float = 0.0
    fast_max_tokens: int = 256
    fast_api_key: Optional[str] = None

    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    use_onnx: bool = True
    cache_embeddings: bool = True

    # Model Persistence
    model_dir: str = ".bandit"
    model_filename: str = "policy.vw"

    # Training Configuration
    training_subset_size: int = 2000
    enable_online_learning: bool = False  # Disabled by default for safety

    # Action Space (fixed for Phase 5)
    action_names: tuple = field(default_factory=lambda: ("fast_path", "slow_path"))

    # Normalized costs for each action (slow = 1.0 baseline)
    fast_cost: float = 0.1  # ~10x cheaper than slow
    slow_cost: float = 1.0

    def get_fast_api_key(self) -> str:
        """Get API key for fast path from config or environment."""
        if self.fast_api_key:
            return self.fast_api_key

        env_var = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        }.get(self.fast_provider)

        if env_var:
            key = os.environ.get(env_var)
            if key:
                return key

        raise ValueError(
            f"No API key found for {self.fast_provider.value}. "
            f"Set {env_var} environment variable or pass fast_api_key to BanditConfig."
        )

    @property
    def model_path(self) -> Path:
        """Full path to the VW model file."""
        return Path(self.model_dir) / self.model_filename

    def get_vw_args(self, load_existing: bool = True) -> str:
        """
        Build VW command line arguments.

        Args:
            load_existing: Whether to load existing model if available

        Returns:
            VW argument string
        """
        args = [
            "--cb_explore_adf",
            f"--{self.exploration_algo}",
            f"--gamma_scale {self.gamma_scale}",
            "--interactions ::",  # Quadratic features between namespaces
            "--quiet",
            "--save_resume",
        ]

        if self.exploration_algo == "epsilon":
            args.append(f"--epsilon {self.epsilon}")

        if load_existing and self.model_path.exists():
            args.append(f"-i {self.model_path}")

        return " ".join(args)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.exploration_algo not in ("squarecb", "epsilon"):
            raise ValueError(
                f"Invalid exploration_algo: {self.exploration_algo}. "
                "Must be 'squarecb' or 'epsilon'."
            )

        if self.gamma_scale <= 0:
            raise ValueError(f"gamma_scale must be positive, got {self.gamma_scale}")

        if not 0 <= self.epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")

        if self.alpha_incorrect < 0:
            raise ValueError(
                f"alpha_incorrect must be non-negative, got {self.alpha_incorrect}"
            )

        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}"
            )

        if self.training_subset_size <= 0:
            raise ValueError(
                f"training_subset_size must be positive, got {self.training_subset_size}"
            )
