# ![LLAMATOR](assets/LLAMATOR.svg)

MCP server for llamator: automate LLM red teaming workflows

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Chat](https://img.shields.io/badge/chat-gray.svg?logo=telegram)](https://t.me/llamator)

## Overview

This repository provides a production-oriented service wrapper around **LLAMATOR** for automated LLM red teaming.
It exposes two integration surfaces:

- **HTTP API (FastAPI)** for job submission, job state retrieval, and artifacts access.
- **MCP server (Streamable HTTP transport)** for agent/tooling integrations, enabling LLAMATOR runs to be invoked as
  tools.

Execution is asynchronous and is orchestrated via **ARQ + Redis**. Artifacts are uploaded to **MinIO** and are retrieved
through presigned URLs (returned as JSON; the API does not redirect).

## Capabilities

- Asynchronous test runs with durable state persisted in Redis.
- Request persistence with secret redaction:
    - API keys are not stored in plaintext.
    - Stored payloads include only boolean markers (e.g. `api_key_present`).
- Artifacts lifecycle management:
    - Worker creates job-local artifacts under `LLAMATOR_MCP_ARTIFACTS_ROOT/<job_id>/...`.
    - Artifacts are uploaded to MinIO as an archive named `artifacts.zip`.
    - HTTP API can list available objects under a job prefix and resolve presigned download links.
- Optional API-key protection for both HTTP and MCP interfaces via `X-API-Key`.
- OpenAPI schema (Swagger UI) with API-key authorization support.
- Prometheus metrics exposed at `/metrics`.

## Deployment (Docker Compose)

Requirements:

- Docker
- Docker Compose

Start the full stack:

```bash
docker compose up --build
```

Default service endpoints:

- HTTP API: `http://localhost:8000`
- MinIO S3 endpoint: `http://localhost:9000`
- MinIO console: `http://localhost:9001`

Healthcheck:

```bash
curl -sS http://localhost:8000/v1/health
```

## Configuration

All configuration is provided via environment variables prefixed with `LLAMATOR_MCP_`.
A complete reference is available in `DOCUMENTATION.md`.

Typical local setup:

```bash
cp .env.example .env
```

## HTTP API usage

### Create a run

```bash
curl -sS -X POST "http://localhost:8000/v1/tests/runs"   -H "Content-Type: application/json"   -H "X-API-Key: <optional>"   -d '{
    "tested_model": {
      "kind": "openai",
      "base_url": "http://host.docker.internal:1235/v1",
      "model": "llm",
      "api_key": "lm-studio"
    },
    "run_config": { "enable_reports": false },
    "plan": { "preset_name": "owasp:llm10", "num_threads": 1 }
  }'
```

The response contains:

- `job_id` (uuid4 hex, 32 characters)
- `status` (`queued | running | succeeded | failed`)
- `created_at` (UTC timestamp)

### Retrieve job state

```bash
curl -sS "http://localhost:8000/v1/tests/runs/<job_id>"   -H "X-API-Key: <optional>"
```

### Artifacts

List objects available for a job:

```bash
curl -sS "http://localhost:8000/v1/tests/runs/<job_id>/artifacts"   -H "X-API-Key: <optional>"
```

Resolve a presigned download URL for a specific object:

```bash
curl -sS "http://localhost:8000/v1/tests/runs/<job_id>/artifacts/<path>"   -H "X-API-Key: <optional>"
```

The download endpoint returns a JSON payload containing `download_url` and does not emit redirects.

## MCP interface

The MCP server is mounted into the FastAPI application (default mount path: `/mcp`) and uses Streamable HTTP transport.

Exposed tools:

- `create_llamator_run`: submits a job, waits for completion, returns aggregated metrics and (if available) a presigned
  URL for `artifacts.zip`.
- `get_llamator_run`: returns aggregated metrics for a finished job and the optional artifacts archive URL.

Protocol notes, headers, and examples are documented in `DOCUMENTATION.md`.

## Security model

- If `LLAMATOR_MCP_API_KEY` is empty, authentication is disabled.
- If configured, protected HTTP routes and the MCP app require `X-API-Key: <value>`.

## Local development

Install dependencies:

```bash
poetry install
```

Run the API server:

```bash
uvicorn llamator_mcp_server.main:app --host 0.0.0.0 --port 8000
```

Run the worker:

```bash
arq llamator_mcp_server.worker_settings.WorkerSettings
```

## Tests

Integration tests are located in `llamator-mcp-server/tests` and rely on `tests/.env.test`.

Run:

```bash
pytest -q
```

## License 📜

This project is licensed under the terms of the **Creative Commons Attribution-ShareAlike 4.0 International** license.
See the [LICENSE](LICENSE) file for details.

[![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-sa/4.0/)