from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
import pytest

WORKFLOW_SKILL_DIR = Path(__file__).parents[1] / "skills" / "workflow"
if str(WORKFLOW_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_SKILL_DIR))

from fusion_flow import _atomic_io as atomic_io  # noqa: E402
from fusion_flow import step_timing  # noqa: E402
from fusion_flow.step_timing import StepTimingStore  # noqa: E402


def _utf16_code_units(path: str | Path | anyio.Path) -> int:
    return len(str(path).encode("utf-16-le")) // 2


def _run_dir_with_report_length(tmp_path: Path, target_length: int) -> Path:
    target_name = "step-timings.json"
    padding_length = target_length - _utf16_code_units(tmp_path) - _utf16_code_units(target_name) - 2
    assert padding_length >= 2
    padding = f"😀{'x' * (padding_length - 2)}"
    assert len(padding.encode("utf-8")) <= 255
    run_dir = tmp_path / padding
    report_path = run_dir / target_name
    assert report_path.is_absolute()
    assert _utf16_code_units(report_path) == target_length
    return run_dir


@pytest.mark.anyio
async def test_timing_store_delegates_exact_json_text_to_shared_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[anyio.Path, str, str | None]] = []

    async def capture_atomic_write_text(
        path: anyio.Path,
        value: str,
        *,
        newline: str | None = "",
    ) -> None:
        writes.append((path, value, newline))

    monkeypatch.setattr(step_timing, "atomic_write_text", capture_atomic_write_text)
    run_dir = anyio.Path(tmp_path / "run")
    store = await StepTimingStore.open(
        run_dir,
        run_id="a" * 32,
        workflow_id="atomic_sidecar",
        flow_path="flows/workflows/atomic-sidecar/atomic-sidecar.workflow",
    )

    await store.persist()
    await store.finalize(status="completed", error_type=None)

    assert len(writes) == 2
    assert [path for path, _value, _newline in writes] == [
        run_dir / "step-timings.json",
        run_dir / "step-timings.json",
    ]
    assert [newline for _path, _value, newline in writes] == ["", ""]
    assert all(value.endswith("\n") and not value.endswith("\n\n") for _path, value, _newline in writes)
    assert [json.loads(value)["status"] for _path, value, _newline in writes] == [
        "running",
        "completed",
    ]


@pytest.mark.anyio
async def test_timing_store_inherits_bounded_paths_and_ownership_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir_path = _run_dir_with_report_length(tmp_path, 259)
    run_dir = anyio.Path(run_dir_path)
    report_path = run_dir_path / "step-timings.json"
    async_report_path = anyio.Path(report_path)
    temporary_name = "f" * _utf16_code_units(report_path.name)
    temporary_path = run_dir_path / temporary_name
    replace_sources: list[Path] = []
    original_replace = anyio.Path.replace
    original_temporary_length = atomic_io._temporary_length_for_platform

    def windows_temporary_length(target: anyio.Path) -> int:
        return original_temporary_length(target, windows=True)

    def fixed_temporary_names(length: int):
        assert length == len(temporary_name)
        yield temporary_name

    async def replace_and_reuse_name(
        source: anyio.Path,
        destination: anyio.Path,
    ) -> anyio.Path:
        source_path = Path(str(source))
        replace_sources.append(source_path)
        if _utf16_code_units(source_path) >= 260:
            raise FileNotFoundError(2, "Windows path is too long", str(source_path))
        replaced = await original_replace(source, destination)
        await source.write_bytes(b"second writer")
        return replaced

    monkeypatch.setattr(
        atomic_io,
        "_temporary_length_for_platform",
        windows_temporary_length,
    )
    monkeypatch.setattr(atomic_io, "_temporary_names", fixed_temporary_names)
    monkeypatch.setattr(anyio.Path, "replace", replace_and_reuse_name)

    store = await StepTimingStore.open(
        run_dir,
        run_id="b" * 32,
        workflow_id="bounded_sidecar",
        flow_path="flows/workflows/bounded-sidecar/bounded-sidecar.workflow",
    )
    await store.persist()

    assert replace_sources == [temporary_path]
    assert _utf16_code_units(replace_sources[0]) < 260
    assert json.loads(await async_report_path.read_text(encoding="utf-8"))["status"] == "running"
    assert (await async_report_path.read_bytes()).endswith(b"\n")
    assert await anyio.Path(temporary_path).read_bytes() == b"second writer"
