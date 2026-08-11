from __future__ import annotations

import sys
from pathlib import Path

import anyio
import pytest

WORKFLOW_SKILL_DIR = Path(__file__).parents[1] / "skills" / "workflow"
if str(WORKFLOW_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_SKILL_DIR))

from fusion_flow.execution import runtime  # noqa: E402


@pytest.mark.anyio
async def test_atomic_write_uses_short_temporary_name_for_long_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_length = 245
    target_name = "metadata.json"
    padding_length = target_length - len(str(tmp_path)) - len(target_name) - 2
    assert 0 < padding_length <= 255
    parent = tmp_path / ("x" * padding_length)
    parent.mkdir()
    target = parent / target_name
    assert len(str(target)) == target_length

    original_write_bytes = anyio.Path.write_bytes

    async def windows_limited_write_bytes(path: anyio.Path, value: bytes) -> None:
        if len(str(path)) >= 260:
            raise FileNotFoundError(2, "Windows path is too long", str(path))
        await original_write_bytes(path, value)

    monkeypatch.setattr(anyio.Path, "write_bytes", windows_limited_write_bytes)

    await runtime._atomic_write_text(anyio.Path(target), "ok")

    assert target.read_text(encoding="utf-8") == "ok"
