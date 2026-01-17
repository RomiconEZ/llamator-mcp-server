from __future__ import annotations

import importlib
import inspect
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterator

import llamator
from llamator_mcp_server.domain.models import OpenAIClientConfig
from llamator_mcp_server.domain.models import TestParameter
from llamator_mcp_server.domain.models import TestPlan


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    """
    Fully resolved worker run configuration.

    :param job_id: Job identifier.
    :param attack_model: Attack model configuration.
    :param tested_model: Tested model configuration.
    :param judge_model: Judge model configuration.
    :param plan: Test plan.
    :param run_config: LLAMATOR config dict.
    :param artifacts_root: Local artifacts root for this job.
    """

    job_id: str
    attack_model: OpenAIClientConfig
    tested_model: OpenAIClientConfig
    judge_model: OpenAIClientConfig
    plan: TestPlan
    run_config: dict[str, Any]
    artifacts_root: Path


def _params_to_dict(params: tuple[TestParameter, ...]) -> dict[str, Any]:
    """
    Convert typed parameters into a dict.

    :param params: Parameters tuple.
    :return: Dict representation.
    """
    return {p.name: p.value for p in params}


def _build_client(cfg: OpenAIClientConfig) -> Any:
    """
    Build a LLAMATOR OpenAI client instance.

    :param cfg: OpenAIClientConfig.
    :return: LLAMATOR client instance.
    """
    api_key: str = cfg.api_key or ""
    temperature: float = cfg.temperature if cfg.temperature is not None else 0.1
    return llamator.ClientOpenAI(
        api_key=api_key,
        base_url=str(cfg.base_url),
        model=cfg.model,
        temperature=temperature,
        system_prompts=list(cfg.system_prompts) if cfg.system_prompts is not None else None,
        model_description=cfg.model_description,
    )


def _resolve_basic_tests(plan: TestPlan) -> list[tuple[str, dict[str, Any]]] | None:
    """
    Resolve built-in tests from preset_name and explicit basic tests.

    :param plan: Test plan.
    :return: List of (code_name, params) pairs or None.
    """
    tests: list[tuple[str, dict[str, Any]]] = []

    if plan.preset_name is not None:
        preset_tests: list[tuple[str, dict[str, Any]]] = llamator.get_test_preset(plan.preset_name.strip())
        tests.extend(preset_tests)

    if plan.basic_tests is not None:
        for spec in plan.basic_tests:
            tests.append((spec.code_name, _params_to_dict(spec.params)))

    return tests or None


def _import_custom_test(import_path: str) -> type:
    """
    Import a test class by a fully qualified import path.

    :param import_path: Module path + class name.
    :return: Imported class.
    :raises ValueError: If the path is invalid or does not point to a class.
    """
    module_name, _, class_name = import_path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError("import_path must be a fully-qualified path to a class.")
    module = importlib.import_module(module_name)
    obj = getattr(module, class_name, None)
    if obj is None or not inspect.isclass(obj):
        raise ValueError("import_path must point to an importable class.")
    return obj


def _resolve_custom_tests(plan: TestPlan) -> list[tuple[type, dict[str, Any]]] | None:
    """
    Resolve custom tests from import_path specs.

    :param plan: Test plan.
    :return: List of (class, params) pairs or None.
    """
    if plan.custom_tests is None:
        return None

    tests: list[tuple[type, dict[str, Any]]] = []
    for spec in plan.custom_tests:
        test_cls: type = _import_custom_test(spec.import_path)
        tests.append((test_cls, _params_to_dict(spec.params)))
    return tests or None


@dataclass(frozen=True, slots=True)
class _RootLoggerSnapshot:
    """
    Immutable snapshot of root logger configuration.

    :param level: Root logger level.
    :param handlers: Root logger handlers list.
    :param filters: Root logger filters list.
    :param disabled: Root logger disabled flag.
    """

    level: int
    handlers: tuple[logging.Handler, ...]
    filters: tuple[logging.Filter, ...]
    disabled: bool


def _snapshot_root_logger(root: logging.Logger) -> _RootLoggerSnapshot:
    """
    Capture current root logger configuration.

    :param root: Root logger.
    :return: _RootLoggerSnapshot.
    """
    return _RootLoggerSnapshot(
        level=int(root.level),
        handlers=tuple(root.handlers),
        filters=tuple(root.filters),
        disabled=bool(root.disabled),
    )


def _clear_root_logger(root: logging.Logger) -> None:
    """
    Remove all handlers and filters from root logger.

    :param root: Root logger.
    :return: None.
    """
    for h in list(root.handlers):
        root.removeHandler(h)
    for f in list(root.filters):
        root.removeFilter(f)
    root.setLevel(logging.NOTSET)
    root.disabled = False


def _restore_root_logger(root: logging.Logger, snap: _RootLoggerSnapshot) -> None:
    """
    Restore root logger configuration from a snapshot.

    :param root: Root logger.
    :param snap: Snapshot.
    :return: None.
    """
    _clear_root_logger(root)

    root.setLevel(int(snap.level))
    root.disabled = bool(snap.disabled)

    for f in snap.filters:
        root.addFilter(f)

    for h in snap.handlers:
        root.addHandler(h)


def _close_handlers(handlers: tuple[logging.Handler, ...]) -> None:
    """
    Flush and close handlers safely.

    :param handlers: Handlers tuple.
    :return: None.
    """
    for h in handlers:
        try:
            h.flush()
        except Exception:
            pass
        try:
            h.close()
        except Exception:
            pass


@contextmanager
def _llamator_root_logging_scope() -> Iterator[None]:
    """
    Provide a temporary "clean" root logger scope for LLAMATOR execution.

    LLAMATOR and its dependencies may configure logging via logging.basicConfig
    and expect the root logger to have no pre-existing handlers. In production,
    the root logger is often configured by the HTTP server / worker startup,
    which makes basicConfig a no-op and leads to missing logs in LLAMATOR artifacts.

    This context manager:
    - snapshots root logger config,
    - clears root handlers/filters before LLAMATOR runs,
    - after completion, closes any handlers added by LLAMATOR and restores the snapshot.

    :return: Iterator[None].
    """
    root: logging.Logger = logging.getLogger()
    snap: _RootLoggerSnapshot = _snapshot_root_logger(root)
    _clear_root_logger(root)
    try:
        yield
    finally:
        added_handlers: tuple[logging.Handler, ...] = tuple(root.handlers)
        _clear_root_logger(root)
        _close_handlers(added_handlers)
        _restore_root_logger(root, snap)


class LlamatorRunner:
    """
    LLAMATOR tests executor (called by the worker).

    :param logger: Logger.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger: logging.Logger = logger

    def run(self, resolved: ResolvedRun) -> dict[str, dict[str, int]]:
        """
        Run LLAMATOR tests.

        :param resolved: Fully resolved run configuration.
        :return: Aggregated LLAMATOR results.
        :raises Exception: Propagates any LLAMATOR/client errors not handled upstream.
        """
        resolved.artifacts_root.mkdir(parents=True, exist_ok=True)

        attack_model = _build_client(resolved.attack_model)
        tested_model = _build_client(resolved.tested_model)
        judge_model = _build_client(resolved.judge_model)

        basic_tests = _resolve_basic_tests(resolved.plan)
        custom_tests = _resolve_custom_tests(resolved.plan)

        num_threads: int | None = resolved.plan.num_threads
        self._logger.info(f"Starting LLAMATOR run for job_id={resolved.job_id}")

        with _llamator_root_logging_scope():
            return llamator.start_testing(
                attack_model=attack_model,
                tested_model=tested_model,
                config=resolved.run_config,
                judge_model=judge_model,
                num_threads=num_threads,
                basic_tests=basic_tests,
                custom_tests=custom_tests,
            )