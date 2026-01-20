"""
Factory for creating tracers based on configuration.

Supports switching between LOCAL, CLOUD, and OFF modes
via TRACE_MODE environment variable.
"""

import logging
import os
from typing import Any, Dict, Optional

from .base import TraceSpan, TracerProtocol
from .local_tracer import LocalJSONLTracer

logger = logging.getLogger(__name__)


class NullTracer:
    """
    No-op tracer when tracing is disabled.

    All methods are no-ops that return minimal valid values.
    """

    def start_span(
        self,
        name: str,
        inputs: Dict[str, Any],
        run_type: str = "chain",
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        """Return a minimal span object."""
        return TraceSpan(name=name)

    def end_span(
        self,
        span: TraceSpan,
        outputs: Dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        """No-op."""
        pass

    def flush(self) -> None:
        """No-op."""
        pass


class CloudTracer:
    """
    Cloud tracer that wraps LangSmith client.

    Falls back to LocalJSONLTracer if LangSmith is not configured.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: str = "ns-mas-agent",
    ):
        """
        Initialize the cloud tracer.

        Args:
            api_key: LangSmith API key (falls back to LANGSMITH_API_KEY env var)
            project_name: LangSmith project name
        """
        self.api_key = api_key or os.environ.get("LANGSMITH_API_KEY")
        self.project_name = project_name
        self._client = None
        self._fallback = None

        if not self.api_key:
            logger.warning(
                "LANGSMITH_API_KEY not set, falling back to local tracer"
            )
            self._fallback = LocalJSONLTracer()

    @property
    def client(self):
        """Lazy initialization of LangSmith client."""
        if self._client is None and self.api_key:
            try:
                from langsmith import Client
                self._client = Client(api_key=self.api_key)
                logger.info(f"LangSmith client initialized for project: {self.project_name}")
            except ImportError:
                logger.warning("langsmith not installed, falling back to local tracer")
                self._fallback = LocalJSONLTracer()
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith: {e}, falling back to local")
                self._fallback = LocalJSONLTracer()
        return self._client

    def start_span(
        self,
        name: str,
        inputs: Dict[str, Any],
        run_type: str = "chain",
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        """Start a trace span."""
        if self._fallback:
            return self._fallback.start_span(name, inputs, run_type, parent_id, metadata)

        span = TraceSpan(
            name=name,
            inputs=inputs,
            run_type=run_type,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        # LangSmith integration would go here
        return span

    def end_span(
        self,
        span: TraceSpan,
        outputs: Dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        """End a trace span."""
        if self._fallback:
            return self._fallback.end_span(span, outputs, error)

        span.end_time = span.end_time or __import__("datetime").datetime.utcnow()
        span.outputs = outputs
        span.error = error
        # LangSmith integration would go here

    def flush(self) -> None:
        """Flush traces."""
        if self._fallback:
            return self._fallback.flush()
        # LangSmith flushes automatically


def create_tracer(
    mode: Optional[str] = None,
    base_dir: str = ".traces",
    session_id: Optional[str] = None,
    **kwargs,
) -> TracerProtocol:
    """
    Factory function to create tracer based on configuration.

    Args:
        mode: Tracing mode ("CLOUD", "LOCAL", "OFF")
              Falls back to TRACE_MODE env var, then "LOCAL"
        base_dir: Base directory for local traces
        session_id: Optional session ID for local tracer
        **kwargs: Additional arguments passed to tracer

    Returns:
        Configured tracer instance

    Modes:
        - OFF: NullTracer (no tracing)
        - LOCAL: LocalJSONLTracer (write to local files)
        - CLOUD: CloudTracer (send to LangSmith, fallback to local)
    """
    # Determine mode from argument, env var, or default
    mode = mode or os.environ.get("TRACE_MODE", "LOCAL")
    mode = mode.upper()

    logger.info(f"Creating tracer with mode: {mode}")

    if mode == "OFF":
        return NullTracer()
    elif mode == "LOCAL":
        return LocalJSONLTracer(base_dir=base_dir, session_id=session_id)
    elif mode == "CLOUD":
        return CloudTracer(**kwargs)
    else:
        logger.warning(f"Unknown trace mode '{mode}', defaulting to LOCAL")
        return LocalJSONLTracer(base_dir=base_dir, session_id=session_id)
