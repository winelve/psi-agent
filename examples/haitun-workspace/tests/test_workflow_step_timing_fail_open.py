from __future__ import annotations

import sys
from collections.abc import Mapping
from datetime import datetime
from importlib import import_module
from pathlib import Path

import anyio
import pytest

WORKFLOW_SKILL_DIR = Path(__file__).parents[1] / "skills" / "workflow"
if str(WORKFLOW_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_SKILL_DIR))

workflow_execution = import_module("fusion_flow.workflow_execution")
from fusion_flow.step_timing import StepTiming, StepTimingMetadata  # noqa: E402
from fusion_flow.workflow_execution import DispatchContext, execute_plan, generate_plan  # noqa: E402
from fusion_flow.workflow_graph import ArtifactNode, ProducesEdge, StepNode, WorkflowGraph  # noqa: E402


class BusinessError(RuntimeError):
    pass


class TimingError(RuntimeError):
    pass


def _graph(*, max_attempts: int = 2) -> WorkflowGraph:
    return WorkflowGraph(
        workflow_id="timing_fail_open",
        steps=(
            StepNode(
                step_id="step",
                name_id="Run once",
                executor_id="worker",
                instruction_id="Run the operation",
                max_attempts=max_attempts,
            ),
        ),
        artifacts=(ArtifactNode("result", is_output=True),),
        edges=(ProducesEdge("step", "result"),),
    )


def _metadata() -> dict[str, StepTimingMetadata]:
    return {
        "step": StepTimingMetadata(
            step_name="Run once",
            executor_id="worker",
            executor_kind="Program",
        )
    }


def _leaf_exceptions(error: BaseException) -> tuple[BaseException, ...]:
    if isinstance(error, BaseExceptionGroup):
        return tuple(nested for child in error.exceptions for nested in _leaf_exceptions(child))
    return (error,)


@pytest.mark.anyio
async def test_utc_rollback_does_not_retry_successful_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter(
        (
            "2026-08-11T10:00:00Z",
            "2026-08-11T09:59:59Z",
        )
    )
    monkeypatch.setattr(workflow_execution, "_timing_now", lambda: next(clock))
    dispatch_calls = 0
    records: list[StepTiming] = []

    async def dispatch(
        _step: StepNode,
        _inputs: Mapping[str, object],
        _context: DispatchContext,
    ) -> dict[str, object]:
        nonlocal dispatch_calls
        dispatch_calls += 1
        return {"result": "ok"}

    outputs = await execute_plan(
        generate_plan(graph := _graph()),
        graph,
        inputs={},
        dispatch=dispatch,
        timing_recorder=records.append,
        timing_metadata=_metadata(),
    )

    assert outputs == {"result": "ok"}
    assert dispatch_calls == 1
    assert len(records) == 1
    assert len(records[0].attempts) == 1
    assert datetime.fromisoformat(records[0].finished_at) >= datetime.fromisoformat(records[0].started_at)
    assert datetime.fromisoformat(records[0].attempts[0].finished_at) >= datetime.fromisoformat(
        records[0].attempts[0].started_at
    )


@pytest.mark.anyio
async def test_attempt_timing_construction_failure_does_not_retry_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_attempt_timing(**_kwargs: object) -> None:
        raise TimingError("attempt timing validation failed")

    monkeypatch.setattr(workflow_execution, "AttemptTiming", fail_attempt_timing)
    dispatch_calls = 0
    records: list[StepTiming] = []

    async def dispatch(
        _step: StepNode,
        _inputs: Mapping[str, object],
        _context: DispatchContext,
    ) -> dict[str, object]:
        nonlocal dispatch_calls
        dispatch_calls += 1
        return {"result": "ok"}

    outputs = await execute_plan(
        generate_plan(graph := _graph(max_attempts=3)),
        graph,
        inputs={},
        dispatch=dispatch,
        timing_recorder=records.append,
        timing_metadata=_metadata(),
    )

    assert outputs == {"result": "ok"}
    assert dispatch_calls == 1
    assert len(records) == 1
    assert records[0].attempts == ()


@pytest.mark.anyio
@pytest.mark.parametrize("failure_site", ("step-constructor", "recorder"))
async def test_step_record_failures_do_not_change_success(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    recorder_calls = 0

    def fail_step_timing(**_kwargs: object) -> None:
        raise TimingError("step timing validation failed")

    def recorder(_record: StepTiming) -> None:
        nonlocal recorder_calls
        recorder_calls += 1
        raise TimingError("timing recorder failed")

    if failure_site == "step-constructor":
        monkeypatch.setattr(workflow_execution, "StepTiming", fail_step_timing)

    async def dispatch(
        _step: StepNode,
        _inputs: Mapping[str, object],
        _context: DispatchContext,
    ) -> dict[str, object]:
        return {"result": "ok"}

    outputs = await execute_plan(
        generate_plan(graph := _graph()),
        graph,
        inputs={},
        dispatch=dispatch,
        timing_recorder=recorder,
        timing_metadata=_metadata(),
    )

    assert outputs == {"result": "ok"}
    assert recorder_calls == (0 if failure_site == "step-constructor" else 1)


@pytest.mark.anyio
async def test_attempt_timing_failure_does_not_replace_dispatch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_attempt_timing(**_kwargs: object) -> None:
        raise TimingError("attempt timing validation failed")

    monkeypatch.setattr(workflow_execution, "AttemptTiming", fail_attempt_timing)
    authoritative = BusinessError("authoritative dispatcher failure")
    dispatch_calls = 0

    async def dispatch(
        _step: StepNode,
        _inputs: Mapping[str, object],
        _context: DispatchContext,
    ) -> dict[str, object]:
        nonlocal dispatch_calls
        dispatch_calls += 1
        raise authoritative

    with pytest.raises(ExceptionGroup) as raised:
        await execute_plan(
            generate_plan(graph := _graph(max_attempts=2)),
            graph,
            inputs={},
            dispatch=dispatch,
            timing_recorder=lambda _record: None,
            timing_metadata=_metadata(),
        )

    assert dispatch_calls == 2
    assert _leaf_exceptions(raised.value) == (authoritative,)


@pytest.mark.anyio
async def test_recorder_failure_does_not_replace_cancellation() -> None:
    cancelled = anyio.get_cancelled_exc_class()()
    authoritative = BaseExceptionGroup("authoritative cancellation", [cancelled])
    dispatch_calls = 0
    recorder_calls = 0

    async def dispatch(
        _step: StepNode,
        _inputs: Mapping[str, object],
        _context: DispatchContext,
    ) -> dict[str, object]:
        nonlocal dispatch_calls
        dispatch_calls += 1
        raise authoritative

    def recorder(_record: StepTiming) -> None:
        nonlocal recorder_calls
        recorder_calls += 1
        raise TimingError("timing recorder failed")

    with pytest.raises(BaseExceptionGroup) as raised:
        await execute_plan(
            generate_plan(graph := _graph(max_attempts=3)),
            graph,
            inputs={},
            dispatch=dispatch,
            timing_recorder=recorder,
            timing_metadata=_metadata(),
        )

    assert dispatch_calls == 1
    assert recorder_calls == 1
    assert _leaf_exceptions(raised.value) == (cancelled,)
