"""
Base types and protocols for the tracing infrastructure.

Defines TraceSpan (compatible with LangSmith RunTree) and
TracerProtocol for different backends.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from uuid import uuid4


@dataclass
class TraceSpan:
    """
    A single span in a trace, compatible with LangSmith RunTree schema.

    Attributes:
        id: Unique identifier for the span
        name: Name of the operation (e.g., "generate_asp", "verify_asp")
        run_type: Type of run ("chain", "llm", "tool")
        inputs: Input data for the operation
        outputs: Output data from the operation
        start_time: When the operation started
        end_time: When the operation completed
        error: Error message if operation failed
        parent_id: ID of parent span for nested operations
        metadata: Additional metadata (model, tokens, etc.)
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    run_type: str = "chain"  # "chain", "llm", "tool"
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "run_type": self.run_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }


@runtime_checkable
class TracerProtocol(Protocol):
    """Protocol defining the tracer interface."""

    def start_span(
        self,
        name: str,
        inputs: Dict[str, Any],
        run_type: str = "chain",
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        """
        Start a new trace span.

        Args:
            name: Name of the operation
            inputs: Input data for the operation
            run_type: Type of run ("chain", "llm", "tool")
            parent_id: ID of parent span for nesting
            metadata: Additional metadata

        Returns:
            New TraceSpan object
        """
        ...

    def end_span(
        self,
        span: TraceSpan,
        outputs: Dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        """
        End a trace span.

        Args:
            span: The span to end
            outputs: Output data from the operation
            error: Error message if operation failed
        """
        ...

    def flush(self) -> None:
        """Flush any buffered traces to storage."""
        ...
