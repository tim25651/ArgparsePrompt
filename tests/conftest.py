"""Fixtures for tests."""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable


@pytest.fixture
def ansi_escape() -> re.Pattern[str]:
    """Return an ANSI escape sequence."""
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


@pytest.fixture(autouse=True)
def empty_argv() -> Generator[None]:
    """Change sys.argv temporarily."""
    old_argv = sys.argv
    try:
        sys.argv = ["pytest-mock"]
        yield
    finally:
        sys.argv = old_argv


@pytest.fixture
def patch_argv() -> Generator[Callable[[Iterable[str]], None]]:
    """Patch sys.argv temporarily."""
    old_argv = sys.argv

    def patch_argv_(argv: Iterable[str]) -> None:
        sys.argv = list(argv)

    try:
        yield patch_argv_
    finally:
        sys.argv = old_argv


@pytest.fixture
def input_patcher(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[Callable[[Iterable[str]], None]]:
    _inputs: Iterable[str] | None = None

    def patch_input(inputs: Iterable[str]) -> None:
        nonlocal _inputs
        _inputs = iter(inputs)

        def _input(prompt: str) -> str:
            del prompt  # unused
            return next(_inputs)

        monkeypatch.setattr("builtins.input", _input)

    yield patch_input

    if _inputs is None or list(_inputs):  # pragma: no cover
        raise ValueError("Not all inputs or no inputs were used")
