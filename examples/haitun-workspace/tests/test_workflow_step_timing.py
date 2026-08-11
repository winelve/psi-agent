from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Callable, Mapping
from importlib import import_module
from pathlib import Path
from typing import Literal, cast

import anyio
import pytest

WORKFLOW_SKILL_DIR = Path(__file__).parents[1] / "skills" / "workflow"
if str(WORKFLOW_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_SKILL_DIR))
TOOLS_DIR = Path(__file__).parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

workflow_tool = import_module("run_flow")
from fusion_flow.step_timing import (  # noqa: E402
    AttemptTiming,
    IterationTiming,
    StepTiming,
    StepTimingCollector,
    StepTimingMetadata,
    StepTimingReporter,
    StepTimingStore,
)
from fusion_flow.workflow_execution import DispatchContext, execute_plan, generate_plan  # noqa: E402
from fusion_flow.workflow_graph import (  # noqa: E402
    ArtifactNode,
    ForeachEdge,
    ProducesEdge,
    StepNode,
    WorkflowGraph,
    WorkflowPolicy,
)
from fusion_flow.workflow_runner import ProgramInvocation, execute_workflow  # noqa: E402


@pytest.mark.anyio
async def test_step_timing_sidecar_survives_reopen_and_finalizes(tmp_path: Path) -> None:
    run_dir = anyio.Path(str(tmp_path), "runs", "a" * 32)
    store = await StepTimingStore.open(
        run_dir,
        run_id="a" * 32,
        workflow_id="resume_approval",
        flow_path="flows/workflows/resume-approval/resume-approval.workflow",
    )
    store.collector.record(
        StepTiming(
            step_id="analyze_resume_step",
            step_name="Match and analyze one resume",
            executor_id="resume_analyzer",
            executor_kind="Agent",
            foreach=True,
            started_at="2026-08-08T01:00:00.000000Z",
            finished_at="2026-08-08T01:00:02.500000Z",
            duration_ms=2500.0,
            status="ok",
            iterations=(
                IterationTiming(
                    iteration_index=1,
                    started_at="2026-08-08T01:00:00.500000Z",
                    finished_at="2026-08-08T01:00:02.500000Z",
                    duration_ms=2000.0,
                    status="ok",
                    attempts=(
                        AttemptTiming(
                            attempt=1,
                            started_at="2026-08-08T01:00:00.500000Z",
                            finished_at="2026-08-08T01:00:02.500000Z",
                            duration_ms=2000.0,
                            status="ok",
                        ),
                    ),
                ),
            ),
        )
    )
    await store.persist()
    running_path = run_dir / "step-timings.json"
    running = json.loads(await running_path.read_text(encoding="utf-8"))
    assert running["status"] == "running"
    assert [step["step_id"] for step in running["steps"]] == ["analyze_resume_step"]
    assert not await (run_dir / ".step-timings.partial.json").exists()

    reopened = await StepTimingStore.open(
        run_dir,
        run_id="a" * 32,
        workflow_id="resume_approval",
        flow_path="flows/workflows/resume-approval/resume-approval.workflow",
    )
    assert reopened.collector.snapshot() == store.collector.snapshot()

    await reopened.finalize(status="completed", error_type=None)

    payload = json.loads(await running_path.read_text(encoding="utf-8"))
    assert payload == {
        "error_type": None,
        "flow_path": "flows/workflows/resume-approval/resume-approval.workflow",
        "run_id": "a" * 32,
        "status": "completed",
        "steps": [
            {
                "attempts": [],
                "duration_ms": 2500.0,
                "error_type": None,
                "executor_id": "resume_analyzer",
                "executor_kind": "Agent",
                "finished_at": "2026-08-08T01:00:02.500000Z",
                "foreach": True,
                "iterations": [
                    {
                        "attempts": [
                            {
                                "attempt": 1,
                                "duration_ms": 2000.0,
                                "error_type": None,
                                "finished_at": "2026-08-08T01:00:02.500000Z",
                                "started_at": "2026-08-08T01:00:00.500000Z",
                                "status": "ok",
                            }
                        ],
                        "duration_ms": 2000.0,
                        "error_type": None,
                        "finished_at": "2026-08-08T01:00:02.500000Z",
                        "iteration_index": 1,
                        "started_at": "2026-08-08T01:00:00.500000Z",
                        "status": "ok",
                    }
                ],
                "started_at": "2026-08-08T01:00:00.000000Z",
                "status": "ok",
                "step_id": "analyze_resume_step",
                "step_name": "Match and analyze one resume",
            }
        ],
        "version": 1,
        "workflow_id": "resume_approval",
    }
    assert not await (run_dir / ".step-timings.partial.json").exists()


@pytest.mark.anyio
@pytest.mark.parametrize("malformation", ("unexpected-field", "duplicate-step"))
async def test_step_timing_sidecar_rejects_non_exact_nested_shape(
    tmp_path: Path,
    malformation: str,
) -> None:
    run_dir = anyio.Path(str(tmp_path), malformation)
    store = await StepTimingStore.open(
        run_dir,
        run_id="d" * 32,
        workflow_id="strict_shape",
        flow_path="flows/workflows/strict-shape/strict-shape.workflow",
    )
    store.collector.record(
        StepTiming(
            step_id="step",
            step_name="Run",
            executor_id="worker",
            executor_kind="Program",
            foreach=False,
            started_at="2026-08-08T01:00:00Z",
            finished_at="2026-08-08T01:00:01Z",
            duration_ms=1000.0,
            status="ok",
        )
    )
    await store.persist()
    report_path = run_dir / "step-timings.json"
    payload = cast(dict[str, object], json.loads(await report_path.read_text(encoding="utf-8")))
    steps = cast(list[dict[str, object]], payload["steps"])
    if malformation == "unexpected-field":
        steps[0]["unexpected"] = True
    else:
        steps.append(dict(steps[0]))
    await report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        await StepTimingStore.open(
            run_dir,
            run_id="d" * 32,
            workflow_id="strict_shape",
            flow_path="flows/workflows/strict-shape/strict-shape.workflow",
        )


@pytest.mark.anyio
async def test_step_timing_store_migrates_legacy_partial_on_next_write(tmp_path: Path) -> None:
    run_dir = anyio.Path(str(tmp_path), "legacy")
    store = await StepTimingStore.open(
        run_dir,
        run_id="e" * 32,
        workflow_id="legacy_partial",
        flow_path="flows/workflows/legacy-partial/legacy-partial.workflow",
    )
    store.collector.record(
        StepTiming(
            step_id="step",
            step_name="Run",
            executor_id="worker",
            executor_kind="Program",
            foreach=False,
            started_at="2026-08-08T01:00:00Z",
            finished_at="2026-08-08T01:00:01Z",
            duration_ms=1000.0,
            status="ok",
        )
    )
    await store.persist()
    report_path = run_dir / "step-timings.json"
    legacy_path = run_dir / ".step-timings.partial.json"
    await report_path.replace(legacy_path)

    reopened = await StepTimingStore.open(
        run_dir,
        run_id="e" * 32,
        workflow_id="legacy_partial",
        flow_path="flows/workflows/legacy-partial/legacy-partial.workflow",
    )
    await reopened.persist()

    assert await report_path.exists()
    assert not await legacy_path.exists()
    migrated = json.loads(await report_path.read_text(encoding="utf-8"))
    assert migrated["status"] == "running"
    assert [step["step_id"] for step in migrated["steps"]] == ["step"]


@pytest.mark.anyio
async def test_retry_timing_records_failed_and_successful_attempts() -> None:
    graph = WorkflowGraph(
        workflow_id="retry_timing",
        steps=(
            StepNode(
                step_id="retry_step",
                name_id="Retry one operation",
                executor_id="worker",
                instruction_id="Run the operation",
                max_attempts=2,
            ),
        ),
        artifacts=(ArtifactNode("result", is_output=True),),
        edges=(ProducesEdge("retry_step", "result"),),
    )
    records: list[StepTiming] = []

    async def dispatch(
        _step: StepNode,
        _inputs: Mapping[str, object],
        context: DispatchContext,
    ) -> dict[str, object]:
        if context.attempt == 1:
            raise ValueError("first attempt fails")
        return {"result": "ok"}

    outputs = await execute_plan(
        generate_plan(graph),
        graph,
        inputs={},
        dispatch=dispatch,
        timing_recorder=records.append,
        timing_metadata={
            "retry_step": StepTimingMetadata(
                step_name="Retry one operation",
                executor_id="worker",
                executor_kind="Program",
            )
        },
    )

    assert outputs == {"result": "ok"}
    assert len(records) == 1
    record = records[0]
    assert record.step_id == "retry_step"
    assert record.status == "ok"
    assert record.error_type is None
    assert [attempt.attempt for attempt in record.attempts] == [1, 2]
    assert [attempt.status for attempt in record.attempts] == ["error", "ok"]
    assert [attempt.error_type for attempt in record.attempts] == ["ValueError", None]
    assert all(attempt.duration_ms >= 0 for attempt in record.attempts)
    assert record.duration_ms >= sum(attempt.duration_ms for attempt in record.attempts)


@pytest.mark.anyio
async def test_step_timing_is_recorded_before_checkpoint_persistence() -> None:
    graph = WorkflowGraph(
        workflow_id="checkpoint_boundary",
        steps=(StepNode("step", "Run", "worker", "Run"),),
        artifacts=(ArtifactNode("result", is_output=True),),
        edges=(ProducesEdge("step", "result"),),
    )
    records: list[StepTiming] = []
    checkpoint_started = anyio.Event()
    release_checkpoint = anyio.Event()

    async def dispatch(
        _step: StepNode,
        _inputs: Mapping[str, object],
        _context: DispatchContext,
    ) -> dict[str, object]:
        return {"result": "ok"}

    async def observe_checkpoint(_checkpoint: object) -> None:
        checkpoint_started.set()
        await release_checkpoint.wait()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            lambda: execute_plan(
                generate_plan(graph),
                graph,
                inputs={},
                dispatch=dispatch,
                checkpoint_observer=observe_checkpoint,
                timing_recorder=records.append,
                timing_metadata={
                    "step": StepTimingMetadata(
                        step_name="Run",
                        executor_id="worker",
                        executor_kind="Program",
                    )
                },
            )
        )
        await checkpoint_started.wait()
        assert len(records) == 1
        release_checkpoint.set()


@pytest.mark.anyio
async def test_foreach_group_timing_excludes_iteration_checkpoint_wait() -> None:
    graph = WorkflowGraph(
        workflow_id="foreach_checkpoint_boundary",
        steps=(StepNode("step", "Run", "worker", "Run"),),
        artifacts=(
            ArtifactNode("items", is_input=True),
            ArtifactNode("item", binding_step_id="step"),
            ArtifactNode("results", is_output=True),
        ),
        edges=(
            ForeachEdge("items", "step", "item"),
            ProducesEdge("step", "results"),
        ),
    )
    records: list[StepTiming] = []
    checkpoint_started = anyio.Event()
    release_checkpoint = anyio.Event()
    checkpoint_count = 0

    async def dispatch(
        _step: StepNode,
        _inputs: Mapping[str, object],
        _context: DispatchContext,
    ) -> dict[str, object]:
        return {"results": "ok"}

    async def observe_checkpoint(_checkpoint: object) -> None:
        nonlocal checkpoint_count
        checkpoint_count += 1
        if checkpoint_count == 1:
            checkpoint_started.set()
            await release_checkpoint.wait()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            lambda: execute_plan(
                generate_plan(graph),
                graph,
                inputs={"items": ["one"]},
                dispatch=dispatch,
                checkpoint_observer=observe_checkpoint,
                timing_recorder=records.append,
                timing_metadata={
                    "step": StepTimingMetadata(
                        step_name="Run",
                        executor_id="worker",
                        executor_kind="Program",
                    )
                },
            )
        )
        await checkpoint_started.wait()
        await anyio.sleep(0.2)
        release_checkpoint.set()

    assert len(records) == 1
    assert records[0].duration_ms < 100


def test_collector_merges_foreach_segments_across_resume() -> None:
    collector = StepTimingCollector(
        (
            StepTiming(
                step_id="foreach_step",
                step_name="Process every item",
                executor_id="worker",
                executor_kind="Agent",
                foreach=True,
                started_at="2026-08-08T01:00:00Z",
                finished_at="2026-08-08T01:00:01Z",
                duration_ms=1000.0,
                status="cancelled",
                error_type="CancelledError",
                iterations=(
                    IterationTiming(
                        iteration_index=0,
                        started_at="2026-08-08T01:00:00Z",
                        finished_at="2026-08-08T01:00:01Z",
                        duration_ms=1000.0,
                        status="ok",
                        attempts=(),
                    ),
                ),
            ),
        )
    )

    collector.record(
        StepTiming(
            step_id="foreach_step",
            step_name="Process every item",
            executor_id="worker",
            executor_kind="Agent",
            foreach=True,
            started_at="2026-08-08T02:00:00Z",
            finished_at="2026-08-08T02:00:02Z",
            duration_ms=2000.0,
            status="ok",
            iterations=(
                IterationTiming(
                    iteration_index=1,
                    started_at="2026-08-08T02:00:00Z",
                    finished_at="2026-08-08T02:00:02Z",
                    duration_ms=2000.0,
                    status="ok",
                    attempts=(),
                ),
            ),
        )
    )

    record = collector.snapshot()[0]
    assert record.status == "ok"
    assert record.error_type is None
    assert record.started_at == "2026-08-08T01:00:00Z"
    assert record.finished_at == "2026-08-08T02:00:02Z"
    assert record.duration_ms == 3000.0
    assert [iteration.iteration_index for iteration in record.iterations] == [0, 1]


@pytest.mark.anyio
async def test_foreach_timing_records_group_and_concurrent_iterations() -> None:
    graph = WorkflowGraph(
        workflow_id="foreach_timing",
        steps=(
            StepNode(
                step_id="foreach_step",
                name_id="Process every item",
                executor_id="worker",
                instruction_id="Process one item",
            ),
        ),
        artifacts=(
            ArtifactNode("items", is_input=True),
            ArtifactNode("item", binding_step_id="foreach_step"),
            ArtifactNode("results", is_output=True),
        ),
        edges=(
            ForeachEdge("items", "foreach_step", "item"),
            ProducesEdge("foreach_step", "results"),
        ),
        policy=WorkflowPolicy(max_concurrency=2),
    )
    records: list[StepTiming] = []
    both_started = anyio.Event()
    started_count = 0

    async def dispatch(
        _step: StepNode,
        _inputs: Mapping[str, object],
        context: DispatchContext,
    ) -> dict[str, object]:
        nonlocal started_count
        started_count += 1
        if started_count == 2:
            both_started.set()
        await both_started.wait()
        return {"results": context.iteration_index}

    outputs = await execute_plan(
        generate_plan(graph),
        graph,
        inputs={"items": ["first", "second"]},
        dispatch=dispatch,
        timing_recorder=records.append,
        timing_metadata={
            "foreach_step": StepTimingMetadata(
                step_name="Process every item",
                executor_id="worker",
                executor_kind="Agent",
            )
        },
    )

    assert outputs == {"results": [0, 1]}
    assert started_count == 2
    assert len(records) == 1
    record = records[0]
    assert record.foreach is True
    assert record.attempts == ()
    assert [iteration.iteration_index for iteration in record.iterations] == [0, 1]
    assert [iteration.status for iteration in record.iterations] == ["ok", "ok"]
    assert [attempt.attempt for iteration in record.iterations for attempt in iteration.attempts] == [1, 1]
    assert record.duration_ms >= max(iteration.duration_ms for iteration in record.iterations)


@pytest.mark.anyio
async def test_foreach_cancellation_records_each_started_iteration() -> None:
    graph = WorkflowGraph(
        workflow_id="foreach_cancel_timing",
        steps=(
            StepNode(
                step_id="foreach_step",
                name_id="Process every item",
                executor_id="worker",
                instruction_id="Process one item",
            ),
        ),
        artifacts=(
            ArtifactNode("items", is_input=True),
            ArtifactNode("item", binding_step_id="foreach_step"),
        ),
        edges=(ForeachEdge("items", "foreach_step", "item"),),
        policy=WorkflowPolicy(max_concurrency=2),
    )
    records: list[StepTiming] = []
    both_started = anyio.Event()
    started_count = 0

    async def dispatch(
        _step: StepNode,
        _inputs: Mapping[str, object],
        context: DispatchContext,
    ) -> dict[str, object]:
        nonlocal started_count
        started_count += 1
        if started_count == 2:
            both_started.set()
        await both_started.wait()
        if context.iteration_index == 0:
            raise BaseExceptionGroup(
                "cancelled iteration",
                [anyio.get_cancelled_exc_class()()],
            )
        await anyio.sleep_forever()
        raise AssertionError("unreachable")

    with pytest.raises(BaseExceptionGroup):
        await execute_plan(
            generate_plan(graph),
            graph,
            inputs={"items": ["first", "second"]},
            dispatch=dispatch,
            timing_recorder=records.append,
            timing_metadata={
                "foreach_step": StepTimingMetadata(
                    step_name="Process every item",
                    executor_id="worker",
                    executor_kind="Agent",
                )
            },
        )

    assert len(records) == 1
    assert records[0].status == "cancelled"
    assert [iteration.iteration_index for iteration in records[0].iterations] == [0, 1]
    assert [iteration.status for iteration in records[0].iterations] == ["cancelled", "cancelled"]


@pytest.mark.anyio
async def test_terminal_step_error_is_recorded_without_replacing_exception() -> None:
    graph = WorkflowGraph(
        workflow_id="failed_timing",
        steps=(
            StepNode(
                step_id="failed_step",
                name_id="Fail twice",
                executor_id="worker",
                instruction_id="Fail",
                max_attempts=2,
            ),
        ),
        artifacts=(),
    )
    records: list[StepTiming] = []

    async def dispatch(
        _step: StepNode,
        _inputs: Mapping[str, object],
        _context: DispatchContext,
    ) -> dict[str, object]:
        raise RuntimeError("authoritative failure")

    with pytest.raises(ExceptionGroup) as raised:
        await execute_plan(
            generate_plan(graph),
            graph,
            inputs={},
            dispatch=dispatch,
            timing_recorder=records.append,
            timing_metadata={
                "failed_step": StepTimingMetadata(
                    step_name="Fail twice",
                    executor_id="worker",
                    executor_kind="Program",
                )
            },
        )

    assert len(raised.value.exceptions) == 1
    assert isinstance(raised.value.exceptions[0], RuntimeError)
    assert str(raised.value.exceptions[0]) == "authoritative failure"
    assert len(records) == 1
    assert records[0].status == "error"
    assert records[0].error_type == "RuntimeError"
    assert [attempt.status for attempt in records[0].attempts] == ["error", "error"]


@pytest.mark.anyio
async def test_human_workflow_does_not_emit_timing_records() -> None:
    source = """
const approval:Workflow;
const review_step:Step;
const reviewer:Human,Executor;
const decision:Artifact;

workflow approval {
    input_workflow(approval) == [];
    output_workflow(approval) == [decision];
    step_name(review_step) == "Human review";
    step_instruction(review_step) == "Choose a decision";
    step_executor(review_step) == reviewer;
    produces(review_step) == [decision];
}
"""
    records: list[StepTiming] = []

    async def prepare_human(_prompt: str, _context: object) -> str:
        return "Choose a decision"

    async def request_human(_prepared: str, _context: object) -> object:
        return "approved"

    outputs = await execute_workflow(
        source,
        inputs={},
        prepare_human_instruction=prepare_human,
        request_human=request_human,
        timing_recorder=records.append,
    )

    assert outputs == {"decision": "approved"}
    assert records == []


@pytest.mark.anyio
async def test_run_flow_writes_report_below_resolved_workflow_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = anyio.Path(str(tmp_path))
    bundle_dir = workspace / "flows" / "workflows" / "timed-program"
    await bundle_dir.mkdir(parents=True)
    flow_path = bundle_dir / "timed-program.workflow"
    await flow_path.write_text(
        """
const timed_program:Workflow;
const program_step:Step;
const program_worker:Program,Executor;
const result:Artifact;

workflow timed_program {
    input_workflow(timed_program) == [];
    output_workflow(timed_program) == [result];
    program_path(program_worker) == "./program.py";
    step_name(program_step) == "Run one program";
    step_instruction(program_step) == "Return the result";
    step_executor(program_step) == program_worker;
    produces(program_step) == [result];
}
""",
        encoding="utf-8",
    )

    async def run_with_sessions(operation: object, **_kwargs: object) -> dict[str, object]:
        return await cast(Callable[[], Awaitable[dict[str, object]]], operation)()

    async def load_step_tools() -> object:
        return object()

    async def complete_program(_invocation: object, **_kwargs: object) -> dict[str, object]:
        return {"result": "ok"}

    monkeypatch.setattr(workflow_tool, "_workspace_dir", lambda: Path(str(workspace)))
    monkeypatch.setattr(workflow_tool, "current_tool_ai_socket", lambda: "test-ai")
    monkeypatch.setattr(workflow_tool, "_run_with_agent_sessions", run_with_sessions)
    monkeypatch.setattr(workflow_tool, "_load_step_tools", load_step_tools)
    monkeypatch.setattr(workflow_tool, "_complete_program_step", complete_program)

    result = await workflow_tool.run_flow(
        "flows/workflows/timed-program/timed-program.workflow",
    )

    assert json.loads(result) == {"result": "ok"}
    run_dirs = [path async for path in (bundle_dir / "runs").iterdir()]
    assert len(run_dirs) == 1
    report = json.loads(await (run_dirs[0] / "step-timings.json").read_text(encoding="utf-8"))
    assert report["workflow_id"] == "timed_program"
    assert report["flow_path"] == "flows/workflows/timed-program/timed-program.workflow"
    assert report["status"] == "completed"
    assert [step["step_id"] for step in report["steps"]] == ["program_step"]
    assert not await (run_dirs[0] / ".step-timings.partial.json").exists()


@pytest.mark.anyio
async def test_timing_reporter_contains_sidecar_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = await StepTimingReporter.open(
        anyio.Path(str(tmp_path), "run"),
        run_id="b" * 32,
        workflow_id="failure_isolation",
        flow_path="flows/workflows/failure-isolation/failure-isolation.workflow",
    )

    async def fail_persist(_store: StepTimingStore) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(StepTimingStore, "persist", fail_persist)

    await reporter.persist()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "error_type"),
    (("failed", "RuntimeError"), ("cancelled", "CancelledError")),
)
async def test_step_timing_sidecar_finalizes_non_success_terminal_status(
    tmp_path: Path,
    status: str,
    error_type: str,
) -> None:
    run_dir = anyio.Path(str(tmp_path), status)
    store = await StepTimingStore.open(
        run_dir,
        run_id="c" * 32,
        workflow_id="terminal_status",
        flow_path="flows/workflows/terminal-status/terminal-status.workflow",
    )
    await store.persist()

    await store.finalize(
        status=cast(Literal["failed", "cancelled"], status),
        error_type=error_type,
    )

    report = json.loads(await (run_dir / "step-timings.json").read_text(encoding="utf-8"))
    assert report["status"] == status
    assert report["error_type"] == error_type
    assert not await (run_dir / ".step-timings.partial.json").exists()


@pytest.mark.anyio
async def test_human_resume_updates_public_timings_and_excludes_human_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = anyio.Path(str(tmp_path))
    bundle_dir = workspace / "flows" / "workflows" / "timed-human"
    await bundle_dir.mkdir(parents=True)
    await (bundle_dir / "timed-human.workflow").write_text(
        """
const timed_human:Workflow;
const before_step:Step;
const human_step:Step;
const after_step:Step;
const before_worker:Program,Executor;
const reviewer:Human,Executor;
const after_worker:Program,Executor;
const before:Artifact;
const decision:Artifact;
const result:Artifact;

workflow timed_human {
    input_workflow(timed_human) == [];
    output_workflow(timed_human) == [result];
    program_path(before_worker) == "./before.py";
    program_path(after_worker) == "./after.py";
    step_name(before_step) == "Before review";
    step_instruction(before_step) == "Prepare review";
    step_executor(before_step) == before_worker;
    produces(before_step) == [before];
    step_name(human_step) == "Human review";
    step_instruction(human_step) == "Approve or reject";
    step_executor(human_step) == reviewer;
    consumes(human_step) == [before];
    produces(human_step) == [decision];
    step_name(after_step) == "After review";
    step_instruction(after_step) == "Persist decision";
    step_executor(after_step) == after_worker;
    consumes(after_step) == [decision];
    produces(after_step) == [result];
}
""",
        encoding="utf-8",
    )

    async def run_with_sessions(operation: object, **_kwargs: object) -> dict[str, object]:
        return await cast(Callable[[], Awaitable[dict[str, object]]], operation)()

    async def load_step_tools() -> object:
        return object()

    async def complete_program(invocation: ProgramInvocation, **_kwargs: object) -> dict[str, object]:
        return {"before": "ready"} if "before" in invocation.output_ids else {"result": "approved"}

    async def prepare_human(*_args: object, **_kwargs: object) -> str:
        return json.dumps(
            {
                "question": "Approve?",
                "options": ["approved", "rejected"],
                "recommended": 1,
                "default": "approved",
            }
        )

    monkeypatch.setattr(workflow_tool, "_workspace_dir", lambda: Path(str(workspace)))
    monkeypatch.setattr(workflow_tool, "current_tool_ai_socket", lambda: "test-ai")
    monkeypatch.setattr(workflow_tool, "_run_with_agent_sessions", run_with_sessions)
    monkeypatch.setattr(workflow_tool, "_load_step_tools", load_step_tools)
    monkeypatch.setattr(workflow_tool, "_build_human_preparer_tools", lambda _tools: object())
    monkeypatch.setattr(workflow_tool, "_complete_program_step", complete_program)
    monkeypatch.setattr(workflow_tool, "_prepare_human_step", prepare_human)

    waiting = json.loads(
        await workflow_tool.run_flow(
            "flows/workflows/timed-human/timed-human.workflow",
        )
    )
    control = waiting["$fusion_flow/control"]
    run_id = control["run_id"]
    request_id = control["request"]["request_id"]
    run_dir = bundle_dir / "runs" / run_id
    running = json.loads(await (run_dir / "step-timings.json").read_text(encoding="utf-8"))
    assert running["status"] == "running"
    assert [step["step_id"] for step in running["steps"]] == ["before_step"]
    assert not await (run_dir / ".step-timings.partial.json").exists()

    resumed = await workflow_tool.run_flow_resume(
        run_id,
        request_id,
        json.dumps("approved"),
    )

    assert json.loads(resumed) == {"result": "approved"}
    final = json.loads(await (run_dir / "step-timings.json").read_text(encoding="utf-8"))
    assert [step["step_id"] for step in final["steps"]] == ["after_step", "before_step"]
    assert "human_step" not in {step["step_id"] for step in final["steps"]}
    assert not await (run_dir / ".step-timings.partial.json").exists()
