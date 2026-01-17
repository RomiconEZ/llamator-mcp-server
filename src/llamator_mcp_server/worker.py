from __future__ import annotations

from arq.worker import run_worker

from llamator_mcp_server.worker_settings import WorkerSettings


def main() -> None:
    """
    ARQ worker entrypoint (without using arq CLI).

    :return: None.
    """
    run_worker(WorkerSettings)
