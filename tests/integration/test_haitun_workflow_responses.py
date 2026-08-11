from __future__ import annotations

import sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from anyio import Path as AsyncPath
from anyio.lowlevel import checkpoint

TOOLS_DIR = Path(__file__).parents[2] / "examples" / "haitun-workspace" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

workflow_tool = import_module("run_flow")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"artifact":{"status":"ok"}}', {"artifact": {"status": "ok"}}),
        (
            '\r\n```JSON\r\n{"artifact":[1,true,null]}\r\n````\r\n',
            {"artifact": [1, True, None]},
        ),
    ],
)
def test_agent_step_result_accepts_only_strict_raw_or_single_fence(
    response: str,
    expected: dict[str, object],
) -> None:
    assert (
        workflow_tool._parse_agent_step_result(
            response,
            step_id="review",
            output_ids=("artifact",),
        )
        == expected
    )


@pytest.mark.parametrize(
    "response",
    [
        "{'artifact': 1}",
        '{"artifact": [1,]}',
        '{"artifact":}',
        '{"artifact": "unterminated}',
        '{"artifact": NaN}',
        '{"artifact": Infinity}',
        '{"artifact": 1e400}',
        '{"artifact": {}, "artifact": []}',
        '{"artifact": {"nested": 1, "nested": 2}}',
        '{"artifact": "approved" "rejected"}',
        '{"artifact": 1}{"artifact": 2}',
        '[{"artifact": 1}]',
        '```json\n{"artifact": NaN}\n```',
        '```json\n{"artifact": {}, "artifact": []}\n```',
        '```json\n{"artifact":}\n```',
        '```json\n{"artifact": 1}\n',
        'result:\n```json\n{"artifact": 1}\n```',
        '```json\n{"artifact": 1}\n```\nresult complete',
        '```json\n{"artifact": 1}\n```\n```json\n{"artifact": 2}\n```',
    ],
)
def test_agent_step_result_rejects_ambiguous_or_malformed_json(response: str) -> None:
    with pytest.raises(workflow_tool._AgentStepResultParseError):
        workflow_tool._parse_agent_step_result(
            response,
            step_id="review",
            output_ids=("artifact",),
        )


def test_agent_step_result_requires_exact_output_keys() -> None:
    with pytest.raises(ValueError, match="must match exactly"):
        workflow_tool._parse_agent_step_result(
            '{"unexpected": 1}',
            step_id="review",
            output_ids=("artifact",),
        )


def test_agent_step_result_accepts_an_empty_object_for_zero_outputs() -> None:
    assert (
        workflow_tool._parse_agent_step_result(
            "{}",
            step_id="notification",
            output_ids=(),
        )
        == {}
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('"42"', "42"),
        ("42", 42),
        ("true", True),
        ("null", None),
        ('{"decision":"approve"}', {"decision": "approve"}),
        ('["approve", 2]', ["approve", 2]),
    ],
)
def test_human_response_preserves_valid_json_types(response: str, expected: object) -> None:
    parsed = workflow_tool._parse_human_response(response)

    assert parsed == expected
    assert type(parsed) is type(expected)


@pytest.mark.parametrize(
    "response",
    [
        "同意",
        "  needs changes  ",
        "NaN is a label",
        "Infinity is a concept",
        "01",
    ],
)
def test_human_response_preserves_nonempty_plain_text_verbatim(response: str) -> None:
    assert workflow_tool._parse_human_response(response) == response


@pytest.mark.parametrize(
    "response",
    [
        "",
        "   ",
        "{broken",
        "[1,",
        '"unterminated',
        "NaN",
        "Infinity",
        "-Infinity",
        "1e400",
        "[1e400]",
        '{"value": NaN}',
        '{"decision":"approve","decision":"reject"}',
        '{"outer":{"value":1,"value":2}}',
        '{"value":1} trailing',
        '\ufeff{"decision":"approve"}',
        "\ufeffNaN",
    ],
)
def test_human_response_rejects_invalid_or_ambiguous_json(response: str) -> None:
    with pytest.raises(ValueError, match="human_response_json"):
        workflow_tool._parse_human_response(response)


def test_response_parsers_normalize_recursion_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_recursion_error(_value: str) -> object:
        raise RecursionError("nested too deeply")

    monkeypatch.setattr(workflow_tool, "_parse_strict_json_value", raise_recursion_error)
    with pytest.raises(ValueError, match="human_response_json"):
        workflow_tool._parse_human_response("plain")
    with pytest.raises(workflow_tool._AgentStepResultParseError):
        workflow_tool._parse_agent_step_result(
            '{"artifact": 1}',
            step_id="review",
            output_ids=("artifact",),
        )


def _human_request(*output_ids: str) -> Any:
    return workflow_tool.HumanRequestSpec(
        request_id="1" * 32,
        step_id="human_review",
        question="Review?",
        output_artifact_ids=output_ids,
    )


def _empty_checkpoint() -> Any:
    return workflow_tool.ExecutionCheckpoint(
        workflow_id="workflow",
        plan_digest="0" * 64,
        values={},
    )


def test_single_output_human_resume_binds_the_entire_plain_response() -> None:
    response = workflow_tool._parse_human_response("  needs changes  ")

    resumed = workflow_tool._checkpoint_human_response(
        _empty_checkpoint(),
        _human_request("decision"),
        response,
    )

    assert resumed.values == {"decision": "  needs changes  "}
    assert resumed.completed_step_ids == ("human_review",)


def test_multi_output_human_resume_requires_an_exact_json_object() -> None:
    response = workflow_tool._parse_human_response('{"decision":"approve","reason":"clear"}')

    resumed = workflow_tool._checkpoint_human_response(
        _empty_checkpoint(),
        _human_request("decision", "reason"),
        response,
    )

    assert resumed.values == {"decision": "approve", "reason": "clear"}

    with pytest.raises(ValueError, match="must receive a JSON object"):
        workflow_tool._checkpoint_human_response(
            _empty_checkpoint(),
            _human_request("decision", "reason"),
            workflow_tool._parse_human_response("approve"),
        )
    with pytest.raises(ValueError, match="must match exactly"):
        workflow_tool._checkpoint_human_response(
            _empty_checkpoint(),
            _human_request("decision", "reason"),
            workflow_tool._parse_human_response('{"decision":"approve"}'),
        )


@pytest.mark.anyio
async def test_reopened_artifact_store_republishes_managed_files_and_preserves_extras(
    tmp_path: Path,
) -> None:
    bundle = AsyncPath(tmp_path / "bundle")
    await bundle.mkdir()
    run_id = "5" * 32
    values = {"decision": "approved", "details": {"score": 1}}
    store = await workflow_tool.ArtifactStore.open(
        bundle,
        run_id,
        reuse_existing=False,
    )
    await store.persist(values)

    decision_file = store.artifacts_dir / "decision.md"
    details_file = store.artifacts_dir / "details.md"
    extra_file = store.artifacts_dir / "notes.md"
    await decision_file.write_text("manual edit", encoding="utf-8")
    await details_file.unlink()
    await extra_file.write_text("keep me", encoding="utf-8")

    reopened = await workflow_tool.ArtifactStore.open(
        bundle,
        run_id,
        reuse_existing=True,
    )
    await reopened.persist(values)

    assert await decision_file.read_text(encoding="utf-8") == "approved"
    assert await details_file.read_text(encoding="utf-8") == '```json\n{\n  "score": 1\n}\n```\n'
    assert await extra_file.read_text(encoding="utf-8") == "keep me"


class _ResumeStore:
    def __init__(self, run: Any) -> None:
        self.run = run
        self.saved: list[Any] = []
        self.acquire_count = 0

    def acquire(self, run_id: str) -> _ResumeLease:
        assert run_id == self.run.run_id
        return _ResumeLease(self)


class _ResumeLease:
    def __init__(self, store: _ResumeStore) -> None:
        self.store = store

    async def __aenter__(self) -> _ResumeLease:
        await checkpoint()
        self.store.acquire_count += 1
        return self

    async def __aexit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        await checkpoint()

    async def load(self) -> Any:
        await checkpoint()
        return self.store.run

    async def save(self, run: Any) -> None:
        await checkpoint()
        self.store.run = run
        self.store.saved.append(run)


class _RecordingArtifactStore:
    def __init__(self) -> None:
        self.persisted: list[dict[str, object]] = []

    async def persist(self, values: dict[str, object]) -> None:
        await checkpoint()
        self.persisted.append(dict(values))


@pytest.mark.anyio
async def test_run_flow_resume_plain_text_is_persisted_and_json_string_retry_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "2" * 32
    request_id = "3" * 32
    request = workflow_tool.HumanRequestSpec(
        request_id=request_id,
        step_id="human_review",
        question="Review?",
        output_artifact_ids=("decision",),
    )
    run = workflow_tool.HumanWorkflowRun(
        run_id=run_id,
        status="waiting_for_human",
        flow_path="flows/review.workflow",
        definition_digest="4" * 64,
        inputs={},
        resource_capacities={},
        checkpoint=_empty_checkpoint(),
        prepared_request=request,
    )
    store = _ResumeStore(run)
    artifacts = _RecordingArtifactStore()
    execution_calls: list[Any] = []

    async def read_flow_source(_flow_path: str) -> str:
        await checkpoint()
        return "workflow source"

    async def materialize_instruction_files(_compiled: object, _flow_path: str) -> dict[str, str]:
        await checkpoint()
        return {}

    async def artifact_store(
        _flow_path: str,
        _run_id: str,
        *,
        reuse_existing: bool,
    ) -> _RecordingArtifactStore:
        await checkpoint()
        assert reuse_existing is True
        return artifacts

    async def execute_persisted_run(
        _source: str,
        resumed: Any,
        lease: _ResumeLease,
        *,
        ai_socket: str,
        instruction_files: dict[str, str],
    ) -> str:
        await checkpoint()
        assert ai_socket == "socket"
        assert instruction_files == {}
        execution_calls.append(resumed)
        await lease.save(replace(resumed, status="completed", outputs={"final": "ok"}))
        return '{"final": "ok"}'

    monkeypatch.setattr(workflow_tool, "current_tool_ai_socket", lambda: "socket")
    monkeypatch.setattr(workflow_tool, "_job_store", lambda: store)
    monkeypatch.setattr(workflow_tool, "_read_flow_source", read_flow_source)
    monkeypatch.setattr(workflow_tool, "_compile_workflow_for_run", lambda _source, **_kwargs: object())
    monkeypatch.setattr(workflow_tool, "_materialize_instruction_files", materialize_instruction_files)
    monkeypatch.setattr(workflow_tool, "_workflow_definition_digest", lambda _source, _files: "4" * 64)
    monkeypatch.setattr(workflow_tool, "_artifact_store", artifact_store)
    monkeypatch.setattr(workflow_tool, "_execute_persisted_run", execute_persisted_run)

    first = await workflow_tool.run_flow_resume(run_id, request_id, "foo")
    second = await workflow_tool.run_flow_resume(run_id, request_id, '"foo"')

    assert first == second == '{"final": "ok"}'
    assert len(execution_calls) == 1
    assert artifacts.persisted == [{"decision": "foo"}]
    assert store.run.human_responses == {request_id: "foo"}
    assert store.run.checkpoint.values == {"decision": "foo"}

    saved_count = len(store.saved)
    with pytest.raises(ValueError, match="already has a different response"):
        await workflow_tool.run_flow_resume(run_id, request_id, '"bar"')
    assert len(store.saved) == saved_count

    acquired_before_invalid = store.acquire_count
    with pytest.raises(ValueError, match="human_response_json"):
        await workflow_tool.run_flow_resume(run_id, request_id, "{broken")
    assert store.acquire_count == acquired_before_invalid


class _EmptyToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def get(self, _name: str) -> None:
        return None


def _agent_context(*output_ids: str) -> Any:
    return SimpleNamespace(
        step_id="review",
        executor_id="reviewer",
        output_ids=output_ids,
        dispatch=SimpleNamespace(resource_lease=SimpleNamespace(grants=())),
    )


def _patch_agent_responses(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[str],
    prompts: list[str],
) -> None:
    async def create_step_agent(*_args: object, **_kwargs: object) -> tuple[object, SimpleNamespace]:
        await checkpoint()
        return object(), SimpleNamespace(messages=[])

    async def complete_step_agent(
        _agent: object,
        _conversation: object,
        message: str,
        **_kwargs: object,
    ) -> str:
        await checkpoint()
        prompts.append(message)
        return responses.pop(0)

    monkeypatch.setattr(workflow_tool, "_create_step_agent", create_step_agent)
    monkeypatch.setattr(workflow_tool, "_complete_step_agent", complete_step_agent)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid_response",
    ["{'artifact': 'unsafe'}", '{"unexpected":"unsafe"}'],
)
async def test_agent_step_retries_invalid_json_and_uses_the_next_strict_result(
    monkeypatch: pytest.MonkeyPatch,
    invalid_response: str,
) -> None:
    prompts: list[str] = []
    _patch_agent_responses(
        monkeypatch,
        [invalid_response, '{"artifact":"safe"}'],
        prompts,
    )

    result = await workflow_tool._complete_agent_step(
        "Review the input.",
        _agent_context("artifact"),
        ai_socket="unused",
        tool_registry=cast(Any, _EmptyToolRegistry()),
    )

    assert result == {"artifact": "safe"}
    assert len(prompts) == 2
    assert "exactly these output keys" in prompts[1]
    assert '["artifact"]' in prompts[1]
    assert "Do not add Markdown or prose" in prompts[1]


@pytest.mark.anyio
@pytest.mark.parametrize("output_ids", [(), ("artifact",), ("left", "right")])
async def test_agent_step_never_publishes_raw_fallback_after_three_invalid_attempts(
    monkeypatch: pytest.MonkeyPatch,
    output_ids: tuple[str, ...],
) -> None:
    prompts: list[str] = []
    _patch_agent_responses(
        monkeypatch,
        ['{"artifact":}', '{"artifact":}', '{"artifact":}'],
        prompts,
    )

    with pytest.raises(ValueError, match="remained invalid after 3 attempts"):
        await workflow_tool._complete_agent_step(
            "Review the input.",
            _agent_context(*output_ids),
            ai_socket="unused",
            tool_registry=cast(Any, _EmptyToolRegistry()),
        )

    assert len(prompts) == 3
