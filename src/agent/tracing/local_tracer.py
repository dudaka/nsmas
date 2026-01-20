"""
Local JSONL tracer for development and offline use.

Writes traces to local files with LangSmith-compatible schema,
enabling later rehydration to the cloud.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .base import TraceSpan

logger = logging.getLogger(__name__)


class LocalJSONLTracer:
    """
    Local JSONL tracer with LangSmith-compatible schema.

    Directory structure:
        .traces/{year}/{month}/{day}/session_{uuid}/traces.jsonl

    Features:
        - Append-only JSONL for crash safety
        - Credential scrubbing before writing
        - Session-based organization
        - Human-readable format
    """

    # Patterns for credentials to scrub
    SENSITIVE_PATTERNS = [
        re.compile(r"sk-[a-zA-Z0-9]{32,}"),  # OpenAI API keys
        re.compile(r"sk-ant-[a-zA-Z0-9-]+"),  # Anthropic API keys
        re.compile(r"anthropic-[a-zA-Z0-9]+"),  # Legacy Anthropic
        re.compile(r"key-[a-zA-Z0-9]{32,}"),  # Generic API keys
    ]

    def __init__(
        self,
        base_dir: str = ".traces",
        session_id: Optional[str] = None,
    ):
        """
        Initialize the local tracer.

        Args:
            base_dir: Base directory for traces
            session_id: Optional session ID (generated if not provided)
        """
        self.session_id = session_id or str(uuid4())[:8]
        self.base_dir = Path(base_dir)
        self._buffer: List[TraceSpan] = []
        self._setup_session_dir()

        logger.info(f"Local tracer initialized: {self.trace_file}")

    def _setup_session_dir(self) -> None:
        """Create session directory with date hierarchy."""
        now = datetime.utcnow()
        self.session_dir = (
            self.base_dir
            / str(now.year)
            / f"{now.month:02d}"
            / f"{now.day:02d}"
            / f"session_{now.strftime('%H%M%S')}_{self.session_id}"
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.session_dir / "traces.jsonl"

        # Write session metadata
        metadata_file = self.session_dir / "metadata.json"
        metadata = {
            "session_id": self.session_id,
            "created_at": now.isoformat(),
            "trace_file": str(self.trace_file),
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _scrub_credentials(self, data: Any) -> Any:
        """
        Remove sensitive credentials from trace data.

        Recursively processes dicts, lists, and strings.

        Args:
            data: Data to scrub

        Returns:
            Scrubbed data with credentials replaced
        """
        if isinstance(data, dict):
            return {k: self._scrub_credentials(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._scrub_credentials(item) for item in data]
        elif isinstance(data, str):
            result = data
            for pattern in self.SENSITIVE_PATTERNS:
                result = pattern.sub("[REDACTED]", result)
            return result
        else:
            return data

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
        span = TraceSpan(
            name=name,
            inputs=self._scrub_credentials(inputs),
            run_type=run_type,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        logger.debug(f"Started span: {name} ({span.id[:8]})")
        return span

    def end_span(
        self,
        span: TraceSpan,
        outputs: Dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        """
        End a trace span and add to buffer.

        Args:
            span: The span to end
            outputs: Output data from the operation
            error: Error message if operation failed
        """
        span.end_time = datetime.utcnow()
        span.outputs = self._scrub_credentials(outputs)
        span.error = error

        self._buffer.append(span)
        logger.debug(f"Ended span: {span.name} ({span.id[:8]})")

        # Auto-flush after each span for reliability
        self.flush()

    def flush(self) -> None:
        """Write buffered spans to JSONL file."""
        if not self._buffer:
            return

        with open(self.trace_file, "a", encoding="utf-8") as f:
            for span in self._buffer:
                record = span.to_dict()
                f.write(json.dumps(record) + "\n")

        logger.debug(f"Flushed {len(self._buffer)} spans to {self.trace_file}")
        self._buffer.clear()

    def read_traces(self) -> List[Dict[str, Any]]:
        """
        Read all traces from the current session.

        Returns:
            List of trace records
        """
        if not self.trace_file.exists():
            return []

        traces = []
        with open(self.trace_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))
        return traces
