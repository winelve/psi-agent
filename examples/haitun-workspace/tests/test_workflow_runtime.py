from __future__ import annotations

import errno
import sys
from pathlib import Path

import anyio
import pytest

WORKFLOW_SKILL_DIR = Path(__file__).parents[1] / "skills" / "workflow"
if str(WORKFLOW_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_SKILL_DIR))

from fusion_flow import _atomic_io as atomic_io  # noqa: E402
from fusion_flow.artifact_store import ArtifactStore  # noqa: E402
from fusion_flow.execution import runtime  # noqa: E402
from fusion_flow.job_store import JobStore  # noqa: E402


def _utf16_code_units(path: str | Path | anyio.Path) -> int:
    return len(str(path).encode("utf-16-le")) // 2


def _target_with_utf16_length(
    tmp_path: Path,
    *,
    target_name: str,
    target_length: int,
) -> Path:
    padding_length = target_length - _utf16_code_units(tmp_path) - _utf16_code_units(target_name) - 2
    assert padding_length >= 2
    padding = f"😀{'x' * (padding_length - 2)}"
    assert len(padding.encode("utf-8")) <= 255
    parent = tmp_path / padding
    parent.mkdir()
    target = parent / target_name
    assert target.is_absolute()
    assert _utf16_code_units(target) == target_length
    return target


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

    original_write_bytes = atomic_io._write_new_bytes
    temporary_paths: list[Path] = []

    def windows_limited_write_bytes(path: str, value: bytes) -> None:
        temporary_paths.append(Path(path))
        if _utf16_code_units(path) >= 260:
            raise FileNotFoundError(2, "Windows path is too long", str(path))
        original_write_bytes(path, value)

    monkeypatch.setattr(atomic_io, "_write_new_bytes", windows_limited_write_bytes)

    await runtime._atomic_write_text(anyio.Path(target), "ok")

    assert len(temporary_paths) == 1
    assert _utf16_code_units(temporary_paths[0]) < 260
    assert target.read_text(encoding="utf-8") == "ok"


def test_temporary_name_budget_keeps_the_absolute_path_below_max_path(
    tmp_path: Path,
) -> None:
    target = _target_with_utf16_length(
        tmp_path,
        target_name="metadata.json",
        target_length=259,
    )

    temporary_name_length = atomic_io._temporary_name_length(anyio.Path(target))
    temporary_name = next(atomic_io._temporary_names(temporary_name_length))
    temporary = target.parent / temporary_name

    assert temporary_name_length == _utf16_code_units("metadata.json")
    assert _utf16_code_units(temporary_name) == temporary_name_length
    assert temporary.is_absolute()
    assert _utf16_code_units(temporary) < 260
    assert len(str(temporary)) < _utf16_code_units(temporary)
    assert (
        atomic_io._temporary_length_for_platform(
            anyio.Path(target),
            windows=True,
        )
        == temporary_name_length
    )


def test_windows_long_target_still_uses_classic_budget_when_the_parent_fits(
    tmp_path: Path,
) -> None:
    target = _target_with_utf16_length(
        tmp_path,
        target_name="x" * 20,
        target_length=269,
    )

    temporary_name_length = atomic_io._temporary_length_for_platform(
        anyio.Path(target),
        windows=True,
    )
    temporary = target.parent / next(atomic_io._temporary_names(temporary_name_length))

    assert temporary_name_length == 10
    assert _utf16_code_units(temporary) < 260


@pytest.mark.parametrize("available_length", [1, 5, 20])
def test_temporary_name_budget_supports_short_hex_names(
    tmp_path: Path,
    available_length: int,
) -> None:
    target = _target_with_utf16_length(
        tmp_path,
        target_name="x" * available_length,
        target_length=259,
    )

    temporary_name_length = atomic_io._temporary_name_length(anyio.Path(target))
    temporary_name = next(atomic_io._temporary_names(temporary_name_length))
    temporary = target.parent / temporary_name

    assert temporary_name_length == available_length
    assert _utf16_code_units(temporary_name) == available_length
    assert set(temporary_name) <= set("0123456789abcdef")
    assert temporary_name.casefold() != target.name.casefold()
    assert _utf16_code_units(temporary) < 260


def test_temporary_name_budget_rejects_zero_available_length(
    tmp_path: Path,
) -> None:
    target = tmp_path / "metadata.json"
    parent_and_separator_length = _utf16_code_units(target.parent) + 1

    with pytest.raises(OSError) as error:
        atomic_io._temporary_name_length(
            anyio.Path(target),
            path_limit=parent_and_separator_length + 1,
        )

    assert error.value.errno == errno.ENAMETOOLONG


def test_one_character_temporary_namespace_is_finite_and_unique() -> None:
    names = list(atomic_io._temporary_names(1))

    assert len(names) == 16
    assert len(set(names)) == 16
    assert all(set(name) <= set("0123456789abcdef") for name in names)


@pytest.mark.anyio
async def test_atomic_write_preserves_every_file_when_short_namespace_is_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    occupied = {name: name.encode() for name in "0123456789abcdef"}
    for name, value in occupied.items():
        (tmp_path / name).write_bytes(value)
    monkeypatch.setattr(
        atomic_io,
        "_temporary_length_for_platform",
        lambda _target: 1,
    )

    with pytest.raises(FileExistsError):
        await atomic_io.atomic_write_bytes(anyio.Path(tmp_path / "target"), b"new")

    assert not (tmp_path / "target").exists()
    assert {name: (tmp_path / name).read_bytes() for name in occupied} == occupied


@pytest.mark.anyio
async def test_atomic_write_retries_without_overwriting_a_colliding_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / f".TMP-{'a' * 16}"
    target_alias = f".tmp-{'a' * 16}"
    colliding_name = f".tmp-{'b' * 16}"
    successful_name = f".tmp-{'c' * 16}"
    aliased = tmp_path / target_alias
    colliding = tmp_path / colliding_name
    successful = tmp_path / successful_name
    colliding.write_bytes(b"occupied")
    yielded_candidates: list[str] = []
    requested_lengths: list[int] = []

    def fixed_temporary_names(length: int):
        requested_lengths.append(length)
        for candidate in (target_alias, colliding_name, successful_name):
            assert _utf16_code_units(candidate) == length
            yielded_candidates.append(candidate)
            yield candidate

    monkeypatch.setattr(atomic_io, "_temporary_names", fixed_temporary_names)

    await atomic_io.atomic_write_bytes(anyio.Path(target), b"new")

    assert requested_lengths == [21]
    assert yielded_candidates == [target_alias, colliding_name, successful_name]
    assert not aliased.exists()
    assert colliding.read_bytes() == b"occupied"
    assert target.read_bytes() == b"new"
    assert not successful.exists()


@pytest.mark.anyio
async def test_atomic_write_cleans_the_temporary_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "metadata.json"
    target.write_bytes(b"old")
    temporary_name = f".tmp-{'d' * 16}"
    temporary = tmp_path / temporary_name

    def fixed_temporary_names(length: int):
        assert _utf16_code_units(temporary_name) == length
        yield temporary_name

    async def failing_replace(path: anyio.Path, destination: anyio.Path) -> None:
        assert Path(str(path)) == temporary
        assert Path(str(destination)) == target
        raise OSError("replace failed")

    monkeypatch.setattr(atomic_io, "_temporary_names", fixed_temporary_names)
    monkeypatch.setattr(anyio.Path, "replace", failing_replace)

    with pytest.raises(OSError, match="replace failed"):
        await atomic_io.atomic_write_bytes(anyio.Path(target), b"new")

    assert target.read_bytes() == b"old"
    assert not temporary.exists()


@pytest.mark.anyio
async def test_atomic_write_cleans_an_owned_file_when_writing_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "metadata.json"
    temporary_name = f".tmp-{'1' * 16}"
    temporary = tmp_path / temporary_name

    def fixed_temporary_names(length: int):
        assert _utf16_code_units(temporary_name) == length
        yield temporary_name

    real_open = open

    class FailingWriter:
        def __init__(self, path: str, mode: str) -> None:
            self._stream = real_open(path, mode)

        def __enter__(self) -> FailingWriter:
            return self

        def __exit__(self, *_args: object) -> None:
            self._stream.close()

        def write(self, _value: bytes) -> None:
            self._stream.write(b"partial")
            raise OSError("write failed")

    def failing_open(path: str, mode: str) -> FailingWriter:
        return FailingWriter(path, mode)

    monkeypatch.setattr(atomic_io, "_temporary_names", fixed_temporary_names)
    monkeypatch.setattr(atomic_io, "open", failing_open, raising=False)

    with pytest.raises(OSError, match="write failed"):
        await atomic_io.atomic_write_bytes(anyio.Path(target), b"new")

    assert not target.exists()
    assert not temporary.exists()


@pytest.mark.anyio
async def test_atomic_write_cleans_an_owned_file_when_cancelled_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "metadata.json"
    temporary_name = f".tmp-{'2' * 16}"
    temporary = tmp_path / temporary_name
    checkpoints = 0

    def fixed_temporary_names(_length: int):
        yield temporary_name

    async def cancel_at_commit() -> None:
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 2:
            raise anyio.get_cancelled_exc_class()

    monkeypatch.setattr(atomic_io, "_temporary_names", fixed_temporary_names)
    monkeypatch.setattr(atomic_io, "checkpoint_if_cancelled", cancel_at_commit)

    with pytest.raises(anyio.get_cancelled_exc_class()):
        await atomic_io.atomic_write_bytes(anyio.Path(target), b"new")

    assert checkpoints == 2
    assert not target.exists()
    assert not temporary.exists()


@pytest.mark.anyio
async def test_atomic_write_does_not_unlink_a_reused_name_after_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "metadata.json"
    temporary_name = f".tmp-{'e' * 16}"
    temporary = anyio.Path(tmp_path / temporary_name)
    original_replace = anyio.Path.replace

    def fixed_temporary_names(length: int):
        assert _utf16_code_units(temporary_name) == length
        yield temporary_name

    async def replace_and_reuse_name(path: anyio.Path, destination: anyio.Path) -> anyio.Path:
        replaced = await original_replace(path, destination)
        await path.write_bytes(b"second writer")
        return replaced

    monkeypatch.setattr(atomic_io, "_temporary_names", fixed_temporary_names)
    monkeypatch.setattr(anyio.Path, "replace", replace_and_reuse_name)

    await atomic_io.atomic_write_bytes(anyio.Path(target), b"first writer")

    assert target.read_bytes() == b"first writer"
    assert await temporary.read_bytes() == b"second writer"


@pytest.mark.anyio
async def test_artifact_and_job_stores_share_the_bounded_atomic_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_lengths: list[int] = []
    temporary_name = f".tmp-{'f' * 16}"

    def fixed_temporary_names(length: int):
        requested_lengths.append(length)
        assert _utf16_code_units(temporary_name) == length
        yield temporary_name

    monkeypatch.setattr(atomic_io, "_temporary_names", fixed_temporary_names)

    artifact_store = await ArtifactStore.open(
        anyio.Path(tmp_path / "bundle"),
        "a" * 32,
        reuse_existing=False,
    )
    await artifact_store.persist({"a": "artifact"})

    job_store = JobStore(anyio.Path(tmp_path / "jobs"))
    run = await job_store.create(
        flow_path="flow.py",
        definition_digest="b" * 64,
        inputs={},
    )

    assert requested_lengths == [21, 21]
    assert (tmp_path / "bundle" / "runs" / ("a" * 32) / "artifacts" / "a.md").read_text(
        encoding="utf-8",
    ) == "artifact"
    assert await job_store.load(run.run_id) == run
