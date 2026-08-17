"""Central observability setup for Helix — Logfire.

Logfire (OpenTelemetry-based) traces the whole stack in one place: the FastAPI
app, Celery workers, HTTPX calls, the ingestion pipeline, LLM calls, and system
metrics.

Export targets (independent, both optional):

* **Local UI** — if ``LOCAL_OTLP_ENDPOINT`` is set (e.g. a Jaeger container),
  spans are exported there over OTLP. No account needed; view at the backend's
  UI (Jaeger: http://localhost:16686).
* **Logfire cloud** — if ``LOGFIRE_TOKEN`` is set, spans also go to
  logfire.pydantic.dev.

With neither set, Logfire just prints spans to the console.
"""

from __future__ import annotations

import os

_logfire_configured = False


def configure_logfire(service_name: str):
    """Idempotently configure Logfire and return the module."""
    global _logfire_configured
    import logfire

    if not _logfire_configured:
        processors = []

        # Local OTLP backend (e.g. Jaeger) for a self-hosted trace UI.
        endpoint = os.getenv("LOCAL_OTLP_ENDPOINT")
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            processors.append(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

        logfire.configure(
            service_name=service_name,
            send_to_logfire="if-token-present",  # cloud export only with a token
            environment=os.getenv("ENVIRONMENT", "development"),
            additional_span_processors=processors,
        )
        _logfire_configured = True

    return logfire
