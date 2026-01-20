"""
Hybrid tracing infrastructure for NS-MAS agent.

Supports local JSONL tracing and cloud LangSmith integration,
switchable via TRACE_MODE environment variable.

Modes:
    - LOCAL: Write traces to .traces/{year}/{month}/{day}/session_{uuid}/
    - CLOUD: Send traces to LangSmith (requires LANGSMITH_API_KEY)
    - OFF: Disable tracing entirely
"""

from .base import TraceSpan, TracerProtocol
from .local_tracer import LocalJSONLTracer
from .factory import create_tracer, NullTracer

__all__ = [
    "TraceSpan",
    "TracerProtocol",
    "LocalJSONLTracer",
    "NullTracer",
    "create_tracer",
]
